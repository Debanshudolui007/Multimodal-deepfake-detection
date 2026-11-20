"""
Milestone 6 — Audio Branch Dataset Loader.

Serves concatenated MFCC + Mel-spectrogram tensors for the Audio Branch.

Input per sample:
  - ``mfcc.npy``: ``(120, 300)`` float32
  - ``mel.npy``:  ``(128, 300)`` float32
  - Concatenated: ``(248, 300)`` float32

For videos without any audio stream (common in this dataset — all
source videos are video-only), the loader returns a zero tensor of
the correct shape so training still proceeds. The Audio Branch will
learn to output neutral predictions (near 0.5) for these, which is
correct behaviour — no audio means no audio evidence.

Augmentation:
  - **Train**: SpecAugment-style frequency masking and time masking.
  - **Val/Test**: No augmentation.

Supports ``get_sampler()`` for class-imbalance correction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str, int] = {"real": 0, "fake": 1}

_MFCC_DIM   = 40
_MEL_DIM    = 128
_INPUT_DIM  = _MFCC_DIM + _MEL_DIM   # 168
_TIME_STEPS = 300


class AudioDataset(Dataset):
    """PyTorch Dataset for MFCC + Mel audio features.

    Args:
        splits_path: Path to ``splits.json``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        audio_root: Root of extracted audio features (``processed/audio/``).
        dataset_filter: Optional dataset name filter.
        augment: Apply SpecAugment if True. Auto-set from split if None.
    """

    def __init__(
        self,
        splits_path: Path,
        split: str,
        audio_root: Path,
        dataset_filter: str | None = None,
        augment: bool | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got '{split}'")

        self.split = split
        self.audio_root = audio_root
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
            label_int = LABEL_MAP.get(video["label"], -1)
            if label_int < 0:
                continue
            audio_dir = (
                audio_root / video["dataset"] / video["label"] / video["video_id"]
            )
            self.samples.append(
                {
                    "audio_dir": audio_dir,
                    "label": label_int,
                    "video_id": video["video_id"],
                    "dataset": video["dataset"],
                    "has_audio": (audio_dir / "mfcc.npy").exists(),
                }
            )

        n_with_audio = sum(1 for s in self.samples if s["has_audio"])
        logger.info(
            "AudioDataset [%s]: %d samples | %d with audio | augment=%s",
            split, len(self.samples), n_with_audio, self._augment,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        """Load MFCC + Mel and concatenate.

        Args:
            idx: Dataset index.

        Returns:
            ``(audio_tensor, label, video_id, dataset_name)``.
            ``audio_tensor`` shape: ``(248, 300)`` float32.
        """
        sample = self.samples[idx]
        audio_dir = sample["audio_dir"]

        if sample["has_audio"]:
            try:
                mfcc = np.load(str(audio_dir / "mfcc.npy")).astype(np.float32)  # (120, 300)
                mel  = np.load(str(audio_dir / "mel.npy")).astype(np.float32)   # (128, 300)
                feat = np.concatenate([mfcc, mel], axis=0)                       # (248, 300)
            except Exception:
                logger.warning("Failed to load audio features: %s", audio_dir)
                feat = np.zeros((_INPUT_DIM, _TIME_STEPS), dtype=np.float32)
        else:
            # No audio stream: return zeros
            feat = np.zeros((_INPUT_DIM, _TIME_STEPS), dtype=np.float32)

        tensor = torch.from_numpy(feat)  # (168, 300)

        # ── Audio Augmentation (train only) ───────────────────────────────────
        if self._augment:
            # 1. White Noise (prob 0.3)
            if torch.rand(1).item() < 0.3:
                noise_amp = 0.005 * torch.rand(1).item() * torch.max(torch.abs(tensor))
                tensor = tensor + torch.randn_like(tensor) * noise_amp
                
            # 2. Time Shift (prob 0.3)
            if torch.rand(1).item() < 0.3:
                shift = torch.randint(-int(0.1 * _TIME_STEPS), int(0.1 * _TIME_STEPS), (1,)).item()
                tensor = torch.roll(tensor, shifts=shift, dims=1)
                
            # 3. Volume Scaling (prob 0.3)
            if torch.rand(1).item() < 0.3:
                scale = torch.empty(1).uniform_(0.8, 1.2).item()
                tensor = tensor * scale
                
            # 4. SpecAugment (Time + Freq Mask) (prob 0.3)
            if torch.rand(1).item() < 0.3:
                tensor = _spec_augment(tensor)

        return tensor, sample["label"], sample["video_id"], sample["dataset"]

    def get_labels(self) -> list[int]:
        """Return all labels for sampler construction.

        Returns:
            List of integer labels.
        """
        return [s["label"] for s in self.samples]


def get_audio_sampler(dataset: AudioDataset) -> WeightedRandomSampler:
    """Build a class-balanced WeightedRandomSampler.

    Args:
        dataset: ``AudioDataset`` instance.

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


# ── SpecAugment ───────────────────────────────────────────────────────────────


def _spec_augment(
    tensor: torch.Tensor,
    freq_mask_param: int = 20,
    time_mask_param: int = 30,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """Apply SpecAugment-style frequency and time masking.

    Args:
        tensor: Input tensor ``(C, T)`` where C = feature channels, T = time.
        freq_mask_param: Maximum frequency mask width.
        time_mask_param: Maximum time mask width.
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.

    Returns:
        Augmented tensor of the same shape.
    """
    C, T = tensor.shape
    tensor = tensor.clone()

    for _ in range(num_freq_masks):
        f = torch.randint(0, freq_mask_param + 1, (1,)).item()
        f0 = torch.randint(0, max(C - f, 1), (1,)).item()
        tensor[f0: f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t = torch.randint(0, time_mask_param + 1, (1,)).item()
        t0 = torch.randint(0, max(T - t, 1), (1,)).item()
        tensor[:, t0: t0 + t] = 0.0

    return tensor
