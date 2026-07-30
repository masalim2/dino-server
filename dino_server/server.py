"""
Always-on DINOv3 inference server.

Keeps one (or more) fine-tuned checkpoints hot in GPU memory and exposes an HTTP
endpoint that runs DINOv3 segmentation, driven by input/output *filesystem paths*
in a POST body.

An external coordinator is expected to stage data in/out of the paths; this layer
only keeps the model hot and runs inference.

The server does not take a fixed port: on startup it grabs an OS-assigned free
port and publishes a ``<pid>.json`` record under a well-known directory
(``/eagle/inference_service/.dino_server_registry`` by default). This lets Globus Compute drop one server per
GPU onto a node with no port coordination, and lets client functions discover the
live servers via ``dino_server.registry.discover_servers()``. See that module for
the race-safety rationale.

Run (single GPU, resident):
    CUDA_VISIBLE_DEVICES=0 dino-server \
        --host 0.0.0.0 \
        --default-checkpoint /path/to/best.ckpt

    # equivalently:
    CUDA_VISIBLE_DEVICES=0 python -m dino_server.server --default-checkpoint ...

Request (port is discovered, not fixed):
    curl -s http://localhost:$PORT/infer -H 'Content-Type: application/json' -d '{
        "input_dir":  "/scratch/job123/in",
        "output_dir": "/scratch/job123/out",
        "project":    "moon",
        "save_overlay": true
    }' | jq
"""
import argparse
import atexit
import os
import signal
import threading
import time
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
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
from . import registry

# ------------------------
# Hot model cache
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()
# Serialize GPU work: one model, one CUDA context -> concurrent forward passes
# would just contend. The coordinator can fan out across multiple server copies.
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
    output_dir: str
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
busy = False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "loaded_checkpoints": list(_model_cache.keys()),
        "default_checkpoint": DEFAULT_CHECKPOINT,
        "projects": list(PROJECTS.keys()),
        "busy": busy,
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

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    global busy
    checkpoint = req.checkpoint or DEFAULT_CHECKPOINT
    if checkpoint is None:
        raise HTTPException(400, "No checkpoint given and no --default-checkpoint set")
    if req.project not in PROJECTS:
        raise HTTPException(400, f"Unknown project '{req.project}'; choices: {list(PROJECTS)}")

    input_dir = Path(req.input_dir)
    if not input_dir.is_dir():
        raise HTTPException(400, f"input_dir does not exist: {input_dir}")

    output_dir = Path(req.output_dir)
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
    busy = True
    try:
        _infer(req, model, dataset, mask_dir, overlay_dir)
    finally:
        busy = False

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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--default-checkpoint", type=str, default=None,
                        help="Checkpoint to preload and use when a request omits one")
    args = parser.parse_args()

    DEFAULT_CHECKPOINT = args.default_checkpoint

    # Grab a free port *before* the slow model load, and keep the socket open so no
    # other server can take the port in the meantime. The OS guarantees the port is
    # unique across the concurrently-starting servers on this node; see registry.py.
    sock = registry.reserve_free_socket(args.host)
    port = registry.port_of(sock)
    print(f"[serve] reserved port {port} (pid={os.getpid()})", flush=True)

    if DEFAULT_CHECKPOINT:
        get_model(DEFAULT_CHECKPOINT)  # warm the model before accepting traffic

    # Publish our record only once we're about to serve. Clean it up on the way
    # out — atexit covers normal exit; the signal handlers cover SLURM/PBS/GC
    # sending SIGTERM/SIGINT so we don't leave a stale file behind.
    record_path = registry.register(
        port,
        host=args.host,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        default_checkpoint=DEFAULT_CHECKPOINT,
    )
    print(f"[serve] registered {record_path}", flush=True)

    atexit.register(registry.unregister, record_path)

    def _handle_signal(signum, _frame):
        registry.unregister(record_path)
        # Re-raise default behavior so uvicorn/Python shuts down normally.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handle_signal)

    import uvicorn
    # Single worker on purpose: keeps one hot model per process. Scale by
    # launching more copies on other GPUs, not with --workers. Hand uvicorn the
    # pre-bound socket so the reserved port passes straight through with no gap.
    config = uvicorn.Config(app, workers=1, log_level="info")
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[sock])
    finally:
        registry.unregister(record_path)


if __name__ == "__main__":
    main()
