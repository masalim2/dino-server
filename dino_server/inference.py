"""
DINOv3 segmentation inference building blocks.

Pure, reusable pieces shared by the FastAPI server (``dino_server.server``):
project class definitions, the image dataset, batch prediction, and mask
visualization helpers. No distributed/argparse machinery — the server keeps a
single model hot on one GPU and drives it over HTTP.
"""
import io
import os
import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from . import safe_fs

# ------------------------
# Project definitions
# ------------------------
PROJECTS = {
    "synaps-I": {
        "colors": {
            0: [128, 128, 128],
            1: [0,   0,   255],
            2: [0,   255,   0],
            3: [128,   0, 128],
            4: [255,   0,   0],
            5: [255, 255,   0],
            6: [255, 165,   0],
        },
        "names": {
            0: "background",
            1: "Cortex",
            2: "Phloem Fibers",
            3: "Phloem",
            4: "Xylem vessels",
            5: "Air-based Pith cells",
            6: "Water-based Pith cells",
        },
    },
    "moon": {
        "colors": {
            0: [128, 128, 128],
            1: [0,   120, 255],
            2: [255,  80,  80],
            3: [ 50, 200,  50],
        },
        "names": {
            0: "background",
            1: "ice",
            2: "particles",
            3: "pores",
        },
    },
}

IGNORE_CLASSES = [255]


def mask_to_color(mask_array, class_colors):
    """
    Convert a grayscale mask to RGB using class colors.

    Args:
        mask_array: 2D numpy array with class indices
        class_colors: dict mapping class id -> [R, G, B]

    Returns:
        RGB image as numpy array
    """
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        color_mask[mask_array == class_id] = color

    return color_mask


def normalize_image_percentile(img_array, lower_percentile=2, upper_percentile=98):
    """
    Normalize image using percentiles for high contrast display.

    Args:
        img_array: numpy array (H, W, 3) or (H, W)
        lower_percentile: lower percentile to clip (default 2)
        upper_percentile: upper percentile to clip (default 98)

    Returns:
        Normalized image in range [0, 255] as uint8
    """
    lower = np.percentile(img_array, lower_percentile)
    upper = np.percentile(img_array, upper_percentile)

    img_clipped = np.clip(img_array, lower, upper)

    if upper > lower:
        img_normalized = ((img_clipped - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        img_normalized = img_array.astype(np.uint8)

    return img_normalized


# ------------------------
# Dataset class
# ------------------------
class ImageDataset(Dataset):
    """Images directly inside a user-staged directory, read symlink-safely.

    The directory is identified by a trusted ``anchor`` path plus the path
    components below it (see ``safe_fs.open_dir``). Each item re-opens the
    directory by walking those components rather than holding an fd, so the
    dataset works with any DataLoader worker start method.
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    # Only let PIL try the decoders matching IMG_EXTS, regardless of content.
    PIL_FORMATS = ("PNG", "JPEG", "TIFF")

    def __init__(self, anchor: str, parts: Sequence[str]):
        self.anchor = str(anchor)
        self.parts = tuple(parts)
        dir_fd = safe_fs.open_dir(self.anchor, self.parts)
        try:
            names = safe_fs.list_regular_files(dir_fd)
        finally:
            os.close(dir_fd)
        self.img_names = sorted(
            n for n in names
            if not n.startswith(".") and Path(n).suffix.lower() in self.IMG_EXTS
        )

    def __len__(self):
        return len(self.img_names)

    def _read(self, name: str) -> bytes:
        dir_fd = safe_fs.open_dir(self.anchor, self.parts)
        try:
            return safe_fs.read_file(dir_fd, name)
        finally:
            os.close(dir_fd)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        data = self._read(img_name)

        # Treat oversized images as errors rather than warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=self.PIL_FORMATS) as img:
                if img.mode not in ['RGB', 'L', 'RGBA']:
                    if img.mode in ['I', 'I;16', 'F']:
                        img_array = np.array(img, dtype=np.float64)
                        lo, hi = img_array.min(), img_array.max()
                        scale = 255.0 / (hi - lo) if hi > lo else 0.0
                        img_array = ((img_array - lo) * scale).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='L')
                    else:
                        img = img.convert('L')

                img = img.convert("RGB")
                img_np = np.array(img)

        img_tensor = transforms.ToTensor()(img)  # [0, 1] range

        return {
            'image': img_tensor,
            'image_np': img_np,
            'img_name': img_name,
        }


# ------------------------
# Batch prediction function
# ------------------------
def predict_batch_correct(model, batch_images):
    """
    Run batch inference matching model.predict() behavior exactly.

    Args:
        model: The loaded model
        batch_images: Tensor of shape (B, C, H, W) - raw images [0-1] range

    Returns:
        List of predicted masks (one per image) as numpy arrays
    """
    if model.training:
        model.eval()

    device = next(model.parameters()).device
    batch_size = batch_images.shape[0]
    pred_masks = []

    for i in range(batch_size):
        x = batch_images[i]  # (C, H, W)
        image_h, image_w = x.shape[-2:]

        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        x = transforms.functional.normalize(
            x, mean=model.image_normalize["mean"], std=model.image_normalize["std"]
        )

        crop_size = min(model.image_size)
        x = transforms.functional.resize(x, size=[crop_size])
        x = x.unsqueeze(0)  # (1, C, H', W')

        logits = model._forward_logits(x)  # (1, K+1, H', W')
        logits = logits[:, :-1]             # (1, K, H', W')

        logits = F.interpolate(
            logits, size=(image_h, image_w), mode="bilinear"
        )  # (1, K, H, W)

        masks = logits.argmax(dim=1)                      # (1, H, W)
        masks = model.internal_class_to_class[masks]      # (1, H, W)

        pred_masks.append(masks[0].cpu().numpy())

    return pred_masks
