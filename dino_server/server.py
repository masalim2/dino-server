"""
Always-on DINOv3 inference server.

Keeps one (or more) fine-tuned checkpoints hot in GPU memory and exposes an HTTP
endpoint that runs DINOv3 segmentation, driven by input/output *filesystem paths*
in a POST body.

Security model: everything under a user's staging directory is untrusted.
Checkpoints are code (pickle) and are therefore only configurable at launch,
never from a request, and may not live in a staging area. Staged images are
read, and results written, through ``safe_fs`` so that planted symlinks or
hardlinks can't redirect I/O outside the caller's staging directory.

An external coordinator is expected to stage data in/out of the paths; this layer
only keeps the model hot and runs inference.

Run (single GPU, resident):
    CUDA_VISIBLE_DEVICES=0 dino-server \
        --uds /tmp/dino.sock \
        --default-checkpoint /path/to/best.ckpt \
        --checkpoint other=/path/to/other.ckpt

Request (the FIRST gateway authenticates the caller and sets X-First-User-Id):
    curl -s --unix-socket /tmp/dino.sock http://localhost/v1/segmentation \
        -H 'Content-Type: application/json' -H 'X-First-User-Id: <id>' -d '{
        "input_dir":  "/eagle/.../user-staging/<id>/job123",
        "project":    "moon",
        "save_overlay": true
    }' | jq
"""
import argparse
import contextlib
import os
import re
import time
import threading
from pathlib import Path
from typing import Annotated, Optional, Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from fastapi import FastAPI, Header, HTTPException
from torch.utils.data import DataLoader

import lightly_train

from . import safe_fs
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

DEFAULT_CHECKPOINT_NAME = "default"
STAGING_DIRNAME = "user-staging"

# name -> loaded model. Populated once in main() from launch flags only; never
# modified while serving.
_models: dict[str, Any] = {}
_default_model: Optional[str] = None
# Optional canonical staging root (".../user-staging"); see --staging-root.
_staging_root: Optional[Path] = None

# Serialize GPU work: one model, one CUDA context -> concurrent forward passes
# would just contend. Scale by running more replicas.
_infer_lock = threading.Lock()


def _validate_checkpoint_path(path: str) -> Path:
    """Refuse checkpoints that don't exist or that live in a user staging area."""
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise SystemExit(f"checkpoint is not a regular file: {path}")
    for p in (Path(os.path.abspath(path)), resolved):
        if STAGING_DIRNAME in p.parts or (
            _staging_root is not None and p.is_relative_to(_staging_root)
        ):
            raise SystemExit(
                f"refusing to load checkpoint from a user staging area: {path}"
            )
    return resolved


def load_checkpoints(checkpoints: dict[str, str]) -> None:
    """Load every launch-configured checkpoint and keep it resident."""
    for name, path in checkpoints.items():
        resolved = _validate_checkpoint_path(path)
        print(f"[serve] loading checkpoint {name!r}: {resolved}", flush=True)
        t0 = time.time()
        model = lightly_train.load_model(str(resolved))
        model.eval()
        _models[name] = model.to(DEVICE)
        print(f"[serve] loaded in {time.time() - t0:.1f}s", flush=True)


# ------------------------
# Request / response schema
# ------------------------
class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_dir: str
    project: str = "synaps-I"
    # Name of a checkpoint configured at launch (not a path).
    checkpoint: Optional[str] = None
    save_overlay: bool = False
    batch_size: int = Field(4, ge=1, le=64)
    num_workers: int = Field(4, ge=0, le=16)


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
# Results land beside the input, where the user can read them back through Globus.
OUTPUT_DIR_SUFFIX = ".dinov3.output"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoints": list(_models.keys()),
        "default_checkpoint": _default_model,
        "projects": list(PROJECTS.keys()),
    }


def _save(img: Image.Image, dir_fd: int, name: str) -> None:
    safe_fs.write_file_atomic(dir_fd, name, safe_fs.encode_image(img, name))


def _infer(req: InferRequest, model: Any, dataset: ImageDataset,
           mask_fd: int, overlay_fd: Optional[int]):
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

                _save(Image.fromarray(pred_mask_np.astype(np.uint8), mode="L"),
                      mask_fd, img_name)

                if overlay_fd is not None:
                    img_np = batch["image_np"][i]
                    if isinstance(img_np, torch.Tensor):
                        img_np = img_np.cpu().numpy()
                    color_mask = mask_to_color(pred_mask_np, class_colors)
                    img_norm = normalize_image_percentile(img_np, 2, 98)
                    overlay = (0.6 * img_norm + 0.4 * color_mask).astype(np.uint8)
                    _save(Image.fromarray(overlay), overlay_fd, img_name)


# Must be usable as a single path component.
_USER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,127}")


def resolve_input_dir(input_dir: str, user_id: str) -> tuple[Path, tuple[str, ...]]:
    """Confine input_dir to the caller's staging area.

    Returns ``(anchor, parts)``: ``anchor`` is the caller's canonical
    ``.../user-staging/<user_id>`` directory and ``parts`` the path components
    of input_dir below it. All later I/O walks ``parts`` from ``anchor`` with
    O_NOFOLLOW, so swapping a component for a symlink after this check is caught.
    """
    if not _USER_ID_RE.fullmatch(user_id) or ".." in user_id:
        raise HTTPException(400, "invalid X-First-User-Id")

    bad = HTTPException(
        400,
        f"input_dir must be an absolute path to a subdirectory of your staging "
        f"area (/{STAGING_DIRNAME}/{user_id}/); got: {input_dir!r}",
    )
    if "\0" in input_dir or not os.path.isabs(input_dir):
        raise bad
    resolved = Path(input_dir).resolve()

    if _staging_root is not None:
        anchor = _staging_root / user_id
        if not resolved.is_relative_to(anchor):
            raise bad
        parts = resolved.relative_to(anchor).parts
    else:
        p = resolved.parts
        idx = next(
            (i for i in range(len(p) - 1)
             if p[i] == STAGING_DIRNAME and p[i + 1] == user_id),
            None,
        )
        if idx is None:
            raise bad
        anchor = Path(*p[: idx + 2])
        parts = p[idx + 2:]

    if not parts:
        raise bad
    return anchor, parts


@app.post("/v1/segmentation", response_model=InferResponse)
def segmentation(
    req: InferRequest,
    x_first_user_id: Annotated[str, Header()],
):
    name = req.checkpoint or _default_model
    if name is None:
        raise HTTPException(400, "No checkpoint given and the server has no default")
    model = _models.get(name)
    if model is None:
        raise HTTPException(
            400, f"Unknown checkpoint {name!r}; choices: {list(_models)}"
        )
    if req.project not in PROJECTS:
        raise HTTPException(400, f"Unknown project '{req.project}'; choices: {list(PROJECTS)}")

    anchor, parts = resolve_input_dir(req.input_dir, x_first_user_id)
    input_dir = anchor.joinpath(*parts)
    output_dir = input_dir.with_name(input_dir.name + OUTPUT_DIR_SUFFIX)
    mask_dir = output_dir / "semantic_masks"
    overlay_dir = output_dir / "semantic_masks_overlay"

    try:
        dataset = ImageDataset(anchor, parts)
    except (FileNotFoundError, NotADirectoryError, safe_fs.UnsafePathError) as e:
        raise HTTPException(400, f"input_dir is not a usable directory: {input_dir} ({e})")
    if len(dataset) == 0:
        raise HTTPException(400, f"No images found in {input_dir}")

    t0 = time.time()
    with contextlib.ExitStack() as stack:
        def _keep(fd: int) -> int:
            stack.callback(os.close, fd)
            return fd

        try:
            parent_fd = _keep(safe_fs.open_dir(str(anchor), parts[:-1]))
            out_fd = _keep(safe_fs.ensure_subdir(parent_fd, output_dir.name))
            mask_fd = _keep(safe_fs.ensure_subdir(out_fd, mask_dir.name))
            overlay_fd = (
                _keep(safe_fs.ensure_subdir(out_fd, overlay_dir.name))
                if req.save_overlay else None
            )
            _infer(req, model, dataset, mask_fd, overlay_fd)
        except (safe_fs.UnsafePathError, UnidentifiedImageError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
            raise HTTPException(400, f"Rejected staged data: {e}")

    return InferResponse(
        status="ok",
        num_images=len(dataset),
        mask_dir=str(mask_dir),
        overlay_dir=str(overlay_dir) if req.save_overlay else None,
        elapsed_sec=round(time.time() - t0, 2),
    )


def _parse_named_checkpoint(spec: str) -> tuple[str, str]:
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, path


def main():
    global _default_model, _staging_root
    parser = argparse.ArgumentParser(description="Always-on DINOv3 inference server")
    parser.add_argument("--uds", required=True, help="Unix domain socket to listen on")
    parser.add_argument("--default-checkpoint", type=str, default=None,
                        help=f"Checkpoint used when a request omits `checkpoint` "
                             f"(also selectable as {DEFAULT_CHECKPOINT_NAME!r})")
    parser.add_argument("--checkpoint", type=_parse_named_checkpoint, action="append",
                        default=[], metavar="NAME=PATH",
                        help="Additional checkpoint a request may select by NAME (repeatable)")
    parser.add_argument("--staging-root", type=str, default=None,
                        help=f"Canonical .../{STAGING_DIRNAME} directory. If set, "
                             f"input_dir must be under <staging-root>/<user id>/ "
                             f"(otherwise any /{STAGING_DIRNAME}/<user id>/ segment is accepted)")
    args = parser.parse_args()

    if args.staging_root:
        _staging_root = Path(args.staging_root).resolve(strict=True)

    checkpoints: dict[str, str] = {}
    if args.default_checkpoint:
        checkpoints[DEFAULT_CHECKPOINT_NAME] = args.default_checkpoint
        _default_model = DEFAULT_CHECKPOINT_NAME
    for name, path in args.checkpoint:
        if name in checkpoints:
            parser.error(f"duplicate checkpoint name {name!r}")
        checkpoints[name] = path
    if not checkpoints:
        parser.error("configure at least one checkpoint (--default-checkpoint or --checkpoint)")

    # Load everything before accepting traffic; nothing is loaded after this.
    load_checkpoints(checkpoints)

    import uvicorn
    # Single worker on purpose: keeps one hot model per process. Scale by
    # launching more replicas on other GPUs, not with --workers.
    uvicorn.run(app, uds=args.uds, workers=1, log_level="info")


if __name__ == "__main__":
    main()
