"""
Milestone 7 — Multimodal Dataset Loader.

Serves synchronized (spatial, frequency, audio) tensors for the Fusion Network.
Iterates over the face crops in `processed/faces/` and automatically discovers
the corresponding frequency tensor (`_freq.npy`) and audio feature (`_audio.npy`).
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

from deepfake_detector.datasets_loader.spatial_dataset import _train_transform, _val_transform, _INPUT_SIZE, LABEL_MAP
from deepfake_detector.datasets_loader.frequency_dataset import _FREQ_MEAN, _FREQ_STD

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """PyTorch Dataset returning synchronized (Spatial, Frequency, Audio) samples."""

    def __init__(
        self,
        splits_path: Path,
        split: str,
        faces_root: Path,
        audio_root: Path,
        dataset_filter: str | None = None,
        augment: bool | None = None,
        require_complete_modalities: bool = True,
    ) -> None:
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
            
            audio_dir = audio_root / video["dataset"] / video["label"] / video["video_id"]
            has_audio = (audio_dir / "mfcc.npy").exists()

            for png_path in sorted(face_dir.glob("face_*.png")):
                # Expect freq tensor to be face_XXXXX_freq.npy
                freq_name = png_path.stem + "_freq.npy"
                freq_path = png_path.parent / freq_name

                # Fusion must not learn whether a modality happened to be
                # missing.  Replacing absent features with zero tensors made
                # missingness a strong, label-correlated shortcut.
                if require_complete_modalities and (
                    not freq_path.is_file() or not has_audio
                ):
                    continue

                self.samples.append(
                    {
                        "spatial_path": png_path,
                        "freq_path": freq_path,
                        "audio_dir": audio_dir,
                        "has_audio": has_audio,
                        "label": label_int,
                        "video_id": video["video_id"],
                        "dataset": video["dataset"],
                    }
                )

        self.spatial_transform = _train_transform() if self._augment else _val_transform()

        logger.info(
            "MultimodalDataset [%s]: %d samples | augment=%s | complete_modalities=%s",
            split, len(self.samples), self._augment, require_complete_modalities,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, str, str]:
        """Load synced modalities."""
        sample = self.samples[idx]

        # 1. Spatial
        img_bgr = cv2.imread(str(sample["spatial_path"]))
        if img_bgr is None:
            spatial_tensor = torch.zeros(3, _INPUT_SIZE, _INPUT_SIZE, dtype=torch.float32)
            has_spatial = 0.0
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            spatial_tensor = self.spatial_transform(image=img_rgb)["image"]
            has_spatial = 1.0

        # 2. Frequency
        try:
            arr = np.load(str(sample["freq_path"])).astype(np.float32)
            for c in range(3):
                arr[c] = (arr[c] - _FREQ_MEAN[c]) / (_FREQ_STD[c] + 1e-8)
            freq_tensor = torch.from_numpy(arr)
            if self._augment and torch.rand(1).item() > 0.5:
                freq_tensor = torch.flip(freq_tensor, dims=[2])
            has_freq = 1.0
        except Exception:
            freq_tensor = torch.zeros(3, _INPUT_SIZE, _INPUT_SIZE, dtype=torch.float32)
            has_freq = 0.0

        # 3. Audio
        if sample["has_audio"]:
            try:
                mfcc = np.load(str(sample["audio_dir"] / "mfcc.npy")).astype(np.float32)
                mel  = np.load(str(sample["audio_dir"] / "mel.npy")).astype(np.float32)
                feat = np.concatenate([mfcc, mel], axis=0)
                audio_tensor = torch.from_numpy(feat)
                has_audio = 1.0
            except Exception:
                audio_tensor = torch.zeros(168, 300, dtype=torch.float32)
                has_audio = 0.0
        else:
            audio_tensor = torch.zeros(168, 300, dtype=torch.float32)
            has_audio = 0.0

        mask = torch.tensor([has_spatial, has_freq, has_audio], dtype=torch.float32)

        return (
            spatial_tensor,
            freq_tensor,
            audio_tensor,
            mask,
            sample["label"],
            sample["video_id"],
            sample["dataset"]
        )

    def get_labels(self) -> list[int]:
        return [s["label"] for s in self.samples]

def get_multimodal_sampler(dataset: MultimodalDataset) -> WeightedRandomSampler:
    labels = dataset.get_labels()
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
