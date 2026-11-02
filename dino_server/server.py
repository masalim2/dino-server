"""
Always-on DINOv3 inference server.

Keeps one (or more) fine-tuned checkpoints hot in GPU memory and exposes an HTTP
endpoint that runs DINOv3 segmentation, driven by input/output *filesystem paths*
in a POST body.

An external coordinator is expected to stage data in/out of the paths; this layer
only keeps the model hot and runs inference.

Run (single GPU, resident):
    CUDA_VISIBLE_DEVICES=0 dino-server \
        --uds /tmp/dino.sock \
        --default-checkpoint /path/to/best.ckpt

Request (the FIRST gateway authenticates the caller and sets X-First-User-Id):
    curl -s --unix-socket /tmp/dino.sock http://localhost/v1/segmentation \
        -H 'Content-Type: application/json' -H 'X-First-User-Id: <id>' -d '{
        "input_dir":  "/eagle/.../user-staging/<id>/job123",
        "project":    "moon",
        "save_overlay": true
    }' | jq
"""
import argparse
import threading
import time
from pathlib import Path
from typing import Annotated, Optional, Any

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, Header, HTTPException
from torch.utils.data import DataLoader

import lightly_train

# Reuse the shared inference building blocks.
from .inference import (
    PROJECTS,
    ImageDataset,
    predict_batch_correct,
    mask_to_color,
    normalize_image_percentile,
)

# ------------------------
# Hot model cache
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()
# Serialize GPU work: one model, one CUDA context -> concurrent forward passes
# would just contend. Scale by running more replicas.
_infer_lock = threading.Lock()


def get_model(checkpoint: str):
    """Load a checkpoint once and keep it resident, keyed by path."""
    with _cache_lock:
        model = _model_cache.get(checkpoint)
        if model is None:
            print(f"[serve] loading checkpoint (cold): {checkpoint}", flush=True)
            t0 = time.time()
            model = lightly_train.load_model(checkpoint)
            model.eval()
            model = model.to(DEVICE)
            _model_cache[checkpoint] = model
            print(f"[serve] loaded in {time.time() - t0:.1f}s", flush=True)
        return model


# ------------------------
# Request / response schema
# ------------------------
class InferRequest(BaseModel):
    input_dir: str
    project: str = "synaps-I"
    checkpoint: Optional[str] = None  # falls back to server default
    save_overlay: bool = False
    batch_size: int = 4
    num_workers: int = 4


class InferResponse(BaseModel):
    status: str
    num_images: int
    mask_dir: str
    overlay_dir: Optional[str] = None
    elapsed_sec: float


# ------------------------
# App
# ------------------------
app = FastAPI(title="DINOv3 inference server")
DEFAULT_CHECKPOINT: Optional[str] = None
# Results land beside the input, where the user can read them back through Globus.
OUTPUT_DIR_SUFFIX = ".dinov3.output"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "loaded_checkpoints": list(_model_cache.keys()),
        "default_checkpoint": DEFAULT_CHECKPOINT,
        "projects": list(PROJECTS.keys()),
    }


def _infer(req: InferRequest, model: Any, dataset: ImageDataset, mask_dir: Path, overlay_dir: Path):
    class_colors = PROJECTS[req.project]["colors"]
    loader = DataLoader(
        dataset,
        batch_size=req.batch_size,
        shuffle=False,
        num_workers=req.num_workers,
        pin_memory=True,
    )
    with _infer_lock, torch.inference_mode():
        for batch in loader:
            batch_images = batch["image"].to(DEVICE)
            pred_masks = predict_batch_correct(model, batch_images)

            for i in range(len(batch["img_name"])):
                img_name = batch["img_name"][i]
                pred_mask_np = pred_masks[i]

                Image.fromarray(pred_mask_np.astype(np.uint8), mode="L").save(mask_dir / img_name)

                if req.save_overlay:
                    img_np = batch["image_np"][i]
                    if isinstance(img_np, torch.Tensor):
                        img_np = img_np.cpu().numpy()
                    color_mask = mask_to_color(pred_mask_np, class_colors)
                    img_norm = normalize_image_percentile(img_np, 2, 98)
                    overlay = (0.6 * img_norm + 0.4 * color_mask).astype(np.uint8)
                    Image.fromarray(overlay).save(overlay_dir / img_name)


def resolve_input_dir(input_dir: str, user_id: str) -> Path:
    """Confine input_dir, and the output beside it, to the caller's staging area."""
    staging_segment = f"/user-staging/{user_id}/"
    resolved = str(Path(input_dir).resolve())
    marker = resolved.find(staging_segment)
    if marker == -1 or not resolved[marker + len(staging_segment):]:
        raise HTTPException(
            400,
            f"input_dir must be a subdirectory of your staging area "
            f"({staging_segment}); got: {input_dir!r}",
        )
    return Path(resolved)


@app.post("/v1/segmentation", response_model=InferResponse)
def segmentation(
    req: InferRequest,
    x_first_user_id: Annotated[str, Header()],
):
    checkpoint = req.checkpoint or DEFAULT_CHECKPOINT
    if checkpoint is None:
        raise HTTPException(400, "No checkpoint given and no --default-checkpoint set")
    if req.project not in PROJECTS:
        raise HTTPException(400, f"Unknown project '{req.project}'; choices: {list(PROJECTS)}")

    input_dir = resolve_input_dir(req.input_dir, x_first_user_id)
    if not input_dir.is_dir():
        raise HTTPException(400, f"input_dir does not exist: {input_dir}")

    output_dir = input_dir.with_name(input_dir.name + OUTPUT_DIR_SUFFIX)
    mask_dir = output_dir / "semantic_masks"
    overlay_dir = output_dir / "semantic_masks_overlay"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if req.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(checkpoint)
    dataset = ImageDataset(input_dir)
    if len(dataset) == 0:
        raise HTTPException(400, f"No images found in {input_dir}")

    t0 = time.time()
    _infer(req, model, dataset, mask_dir, overlay_dir)

    return InferResponse(
        status="ok",
        num_images=len(dataset),
        mask_dir=str(mask_dir),
        overlay_dir=str(overlay_dir) if req.save_overlay else None,
        elapsed_sec=round(time.time() - t0, 2),
    )


def main():
    global DEFAULT_CHECKPOINT
    parser = argparse.ArgumentParser(description="Always-on DINOv3 inference server")
    parser.add_argument("--uds", required=True, help="Unix domain socket to listen on")
    parser.add_argument("--default-checkpoint", type=str, default=None,
                        help="Checkpoint to preload and use when a request omits one")
    args = parser.parse_args()

    DEFAULT_CHECKPOINT = args.default_checkpoint

    if DEFAULT_CHECKPOINT:
        get_model(DEFAULT_CHECKPOINT)  # warm the model before accepting traffic

    import uvicorn
    # Single worker on purpose: keeps one hot model per process. Scale by
    # launching more replicas on other GPUs, not with --workers.
    uvicorn.run(app, uds=args.uds, workers=1, log_level="info")


if __name__ == "__main__":
    main()
