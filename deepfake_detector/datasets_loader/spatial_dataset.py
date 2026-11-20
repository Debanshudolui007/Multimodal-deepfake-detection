"""
Milestone 4 — Spatial Branch Dataset Loader.

Provides a PyTorch Dataset that serves 224×224 face crops for the
Spatial (RGB) Branch.  All crops are loaded from the ``splits.json``
file produced by ``split_dataset.py`` in Milestone 3.

Augmentation is handled entirely by ``albumentations``:
  - **Train**: HorizontalFlip, ColorJitter, GaussNoise, RandomRotation,
    CoarseDropout, Normalize (ImageNet stats), ToTensorV2.
  - **Val/Test**: CenterCrop (224), Normalize, ToTensorV2.

Supports:
  - Weighted random sampling (``get_sampler()``) for imbalanced datasets.
  - Single-dataset subset via ``dataset_filter``.
  - Label mapping: ``real → 0``, ``fake → 1``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ── ImageNet normalisation constants ─────────────────────────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_INPUT_SIZE = 224

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP: dict[str, int] = {"real": 0, "fake": 1}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════


class SpatialFaceDataset(Dataset):
    """PyTorch Dataset for 224×224 PNG face crops (Spatial Branch).

    Args:
        splits_path: Path to ``splits.json`` (from ``split_dataset.py``).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        faces_root: Root directory of face crops (``processed/faces/``).
        dataset_filter: If set, only include videos from this dataset name
            (e.g. ``"celebdf"``).
        augment: If True, apply training augmentation. If False, apply
            deterministic val/test transforms.
        transform: Override albumentations transform (for testing).
    """

    def __init__(
        self,
        splits_path: Path,
        split: str,
        faces_root: Path,
        dataset_filter: str | None = None,
        augment: bool | None = None,
        transform: A.Compose | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train'|'val'|'test', got '{split}'")

        self.faces_root = faces_root
        self.split = split

        # ── Load splits.json ──────────────────────────────────────────────────
        if not splits_path.exists():
            raise FileNotFoundError(f"splits.json not found: {splits_path}")
        with splits_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        video_list: list[dict[str, Any]] = data["splits"][split]

        # ── Filter by dataset ──────────────────────────────────────────────────
        if dataset_filter:
            video_list = [v for v in video_list if v["dataset"] == dataset_filter]
            logger.info("[%s/%s] filtered to dataset='%s': %d videos",
                        split, dataset_filter, dataset_filter, len(video_list))

        # ── Expand video records → individual face files ───────────────────────
        self.samples: list[dict[str, Any]] = []
        for video in video_list:
            face_dir = Path(video["face_dir"])
            label_int = LABEL_MAP.get(video["label"], -1)
            if label_int < 0:
                logger.warning("Unknown label '%s' — skipping", video["label"])
                continue
            for png in sorted(face_dir.glob("face_*.png")):
                self.samples.append(
                    {
                        "path": png,
                        "label": label_int,
                        "video_id": video["video_id"],
                        "dataset": video["dataset"],
                    }
                )

        # ── Transforms ────────────────────────────────────────────────────────
        _augment = augment if augment is not None else (split == "train")
        self.transform = transform or (
            _train_transform() if _augment else _val_transform()
        )

        logger.info(
            "SpatialFaceDataset [%s]: %d samples | augment=%s",
            split, len(self.samples), _augment,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        """Return a single sample.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of ``(image_tensor, label, video_id, dataset_name)``.
            ``image_tensor`` shape: ``(3, 224, 224)`` float32.
        """
        sample = self.samples[idx]
        img_bgr = cv2.imread(str(sample["path"]))
        if img_bgr is None:
            # Return a zeroed tensor to avoid crashing the DataLoader
            logger.warning("Cannot read image: %s", sample["path"])
            image = torch.zeros(3, _INPUT_SIZE, _INPUT_SIZE, dtype=torch.float32)
            return image, sample["label"], sample["video_id"], sample["dataset"]

        # albumentations expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=img_rgb)
        image: torch.Tensor = augmented["image"]  # (3, H, W) float32

        return image, sample["label"], sample["video_id"], sample["dataset"]

    def get_labels(self) -> list[int]:
        """Return all labels (for computing class weights).

        Returns:
            List of integer labels in dataset order.
        """
        return [s["label"] for s in self.samples]


def get_sampler(dataset: SpatialFaceDataset) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that balances real/fake classes.

    Args:
        dataset: A ``SpatialFaceDataset`` instance.

    Returns:
        A ``WeightedRandomSampler`` that up-samples the minority class.
    """
    labels = dataset.get_labels()
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════════


def _train_transform() -> A.Compose:
    """Build the training augmentation pipeline.

    Returns:
        Albumentations ``Compose`` pipeline for training.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
            ),
            A.GaussNoise(std_range=(0.0, 0.04), p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=0.2,
            ),
            A.Resize(_INPUT_SIZE, _INPUT_SIZE),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _val_transform() -> A.Compose:
    """Build the deterministic val/test transform pipeline.

    Returns:
        Albumentations ``Compose`` pipeline for val/test.
    """
    return A.Compose(
        [
            A.Resize(_INPUT_SIZE, _INPUT_SIZE),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )
