"""
Milestone 5 — Frequency Branch Dataset Loader.

Serves pre-computed frequency tensors (``face_XXXXX_freq.npy``) paired
with their labels from the ``splits.json`` file produced by Milestone 3.

Each ``.npy`` file is a ``(3, H, W)`` float32 tensor:
  - CH0: FFT log-magnitude
  - CH1: DWT LL subband
  - CH2: DWT energy map

Augmentation for the frequency domain is intentionally minimal — only
RandomHorizontalFlip and normalization — because aggressive spatial
augmentations (rotation, crop, jitter) would destroy the frequency
artefact patterns we are trying to detect.

Supports:
  - ``get_sampler()`` for class-imbalance correction.
  - ``dataset_filter`` to restrict to one source dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str, int] = {"real": 0, "fake": 1}
_INPUT_SIZE = 224

# ImageNet-derived stats adapted for the 3-channel frequency representation
# CH0 (FFT): mean ~0.5, std ~0.2  |  CH1 (LL): ~0.5, 0.25  |  CH2 (Energy): ~0.3, 0.2
_FREQ_MEAN = (0.50, 0.50, 0.30)
_FREQ_STD  = (0.20, 0.25, 0.20)


class FrequencyDataset(Dataset):
    """PyTorch Dataset for (3, 224, 224) frequency tensors.

    Args:
        splits_path: Path to ``splits.json``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        faces_root: Root of face crops (``processed/faces/``).
        dataset_filter: Optional dataset name filter.
        augment: Apply horizontal flip for training if True.
    """

    def __init__(
        self,
        splits_path: Path,
        split: str,
        faces_root: Path,
        dataset_filter: str | None = None,
        augment: bool | None = None,
    ) -> None:
        import json

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got '{split}'")

        self.split = split
        self._augment = augment if augment is not None else (split == "train")

        if not splits_path.exists():
            raise FileNotFoundError(f"splits.json not found: {splits_path}")
        with splits_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        video_list: list[dict[str, Any]] = data["splits"][split]
        if dataset_filter:
            video_list = [v for v in video_list if v["dataset"] == dataset_filter]

        self.samples: list[dict[str, Any]] = []
        for video in video_list:
            face_dir = Path(video["face_dir"])
            label_int = LABEL_MAP.get(video["label"], -1)
            if label_int < 0:
                continue
            for npy_path in sorted(face_dir.glob("face_*_freq.npy")):
                self.samples.append(
                    {
                        "path": npy_path,
                        "label": label_int,
                        "video_id": video["video_id"],
                        "dataset": video["dataset"],
                    }
                )

        logger.info(
            "FrequencyDataset [%s]: %d samples | augment=%s",
            split, len(self.samples), self._augment,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        """Load one frequency tensor.

        Args:
            idx: Dataset index.

        Returns:
            Tuple ``(freq_tensor, label, video_id, dataset_name)``.
            ``freq_tensor`` shape: ``(3, 224, 224)`` float32.
        """
        sample = self.samples[idx]
        try:
            arr = np.load(str(sample["path"])).astype(np.float32)  # (3, H, W)
        except Exception:
            logger.warning("Cannot load: %s", sample["path"])
            arr = np.zeros((3, _INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)

        # ── Normalize each channel ────────────────────────────────────────────
        for c in range(3):
            arr[c] = (arr[c] - _FREQ_MEAN[c]) / (_FREQ_STD[c] + 1e-8)

        tensor = torch.from_numpy(arr)  # (3, H, W)

        # ── Augmentation: horizontal flip only ───────────────────────────────
        if self._augment and torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[2])

        return tensor, sample["label"], sample["video_id"], sample["dataset"]

    def get_labels(self) -> list[int]:
        """Return all labels for sampler construction.

        Returns:
            List of integer labels in dataset order.
        """
        return [s["label"] for s in self.samples]


def get_frequency_sampler(dataset: FrequencyDataset) -> WeightedRandomSampler:
    """Build a class-balanced WeightedRandomSampler.

    Args:
        dataset: ``FrequencyDataset`` instance.

    Returns:
        ``WeightedRandomSampler`` that balances real/fake classes.
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
