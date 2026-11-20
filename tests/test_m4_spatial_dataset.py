"""
Tests for Milestone 4 — Spatial Dataset Loader.

Uses synthetic face-crop directories — no real data required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from deepfake_detector.datasets_loader.spatial_dataset import (
    LABEL_MAP,
    SpatialFaceDataset,
    _train_transform,
    _val_transform,
    get_sampler,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_splits_json(
    tmp_path: Path,
    n_real_train: int = 10,
    n_fake_train: int = 30,
    n_val: int = 5,
    n_test: int = 5,
    face_count: int = 3,
) -> tuple[Path, Path]:
    """Build a minimal splits.json and matching face-crop PNG files."""
    faces_root = tmp_path / "faces"

    def _make_video(dataset: str, label: str, idx: int) -> dict:
        vid_id = f"video_{dataset}_{label}_{idx:04d}"
        vid_dir = faces_root / dataset / label / vid_id
        vid_dir.mkdir(parents=True)
        for i in range(face_count):
            # Create 4-byte PNG header (enough for cv2.imread to reject gracefully)
            # Use a proper 1×1 green PNG instead
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            img[:, :] = [0, 200, 0]
            cv2.imwrite(str(vid_dir / f"face_{i:05d}.png"), img)
        return {
            "video_id": vid_id,
            "dataset": dataset,
            "label": label,
            "face_dir": str(vid_dir),
            "face_count": face_count,
        }

    videos_train = (
        [_make_video("celebdf", "real", i) for i in range(n_real_train)]
        + [_make_video("celebdf", "fake", i) for i in range(n_fake_train)]
    )
    videos_val = [_make_video("celebdf", "fake", 100 + i) for i in range(n_val)]
    videos_test = [_make_video("celebdf", "real", 200 + i) for i in range(n_test)]

    splits_data = {
        "summary": {"total_videos": len(videos_train) + len(videos_val) + len(videos_test)},
        "splits": {
            "train": videos_train,
            "val": videos_val,
            "test": videos_test,
        },
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits_data))
    return splits_path, faces_root


# ═══════════════════════════════════════════════════════════════════════════════
# Label map
# ═══════════════════════════════════════════════════════════════════════════════


class TestLabelMap:
    def test_real_is_zero(self) -> None:
        assert LABEL_MAP["real"] == 0

    def test_fake_is_one(self) -> None:
        assert LABEL_MAP["fake"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SpatialFaceDataset
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpatialFaceDataset:
    @pytest.fixture()
    def splits_and_root(self, tmp_path):
        return _make_splits_json(tmp_path, n_real_train=4, n_fake_train=8, n_val=2, n_test=2, face_count=3)

    def test_train_length_correct(self, splits_and_root) -> None:
        splits_path, faces_root = splits_and_root
        ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        # 4 real + 8 fake = 12 videos × 3 faces each = 36 samples
        assert len(ds) == 36

    def test_val_length_correct(self, splits_and_root) -> None:
        splits_path, faces_root = splits_and_root
        ds = SpatialFaceDataset(splits_path, "val", faces_root, augment=False)
        assert len(ds) == 6  # 2 videos × 3 faces

    def test_getitem_returns_correct_types(self, splits_and_root) -> None:
        splits_path, faces_root = splits_and_root
        ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        img, label, video_id, dataset_name = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(video_id, str)
        assert isinstance(dataset_name, str)

    def test_image_tensor_shape(self, splits_and_root) -> None:
        splits_path, faces_root = splits_and_root
        ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        img, _, _, _ = ds[0]
        assert img.shape == (3, 224, 224)

    def test_image_tensor_dtype_float32(self, splits_and_root) -> None:
        splits_path, faces_root = splits_and_root
        ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        img, _, _, _ = ds[0]
        assert img.dtype == torch.float32

    def test_labels_are_only_zero_or_one(self, splits_and_root) -> None:
        splits_path, faces_root = splits_and_root
        ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        labels = {ds[i][1] for i in range(len(ds))}
        assert labels <= {0, 1}

    def test_dataset_filter_reduces_count(self, splits_and_root) -> None:
        splits_path, faces_root = splits_and_root
        ds_filtered = SpatialFaceDataset(
            splits_path, "train", faces_root,
            dataset_filter="celebdf", augment=False,
        )
        ds_full = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        # Filtered should have ≤ full (all are celebdf in this fixture)
        assert len(ds_filtered) == len(ds_full)

    def test_invalid_split_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="split must be"):
            SpatialFaceDataset(tmp_path / "s.json", "invalid", tmp_path)

    def test_missing_splits_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            SpatialFaceDataset(tmp_path / "nonexistent.json", "train", tmp_path)

    def test_get_labels_length_matches_dataset(self, splits_and_root) -> None:
        splits_path, faces_root = splits_and_root
        ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        assert len(ds.get_labels()) == len(ds)


# ═══════════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransforms:
    def test_train_transform_output_shape(self) -> None:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        t = _train_transform()
        result = t(image=img)["image"]
        assert result.shape == (3, 224, 224)

    def test_val_transform_output_shape(self) -> None:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        t = _val_transform()
        result = t(image=img)["image"]
        assert result.shape == (3, 224, 224)

    def test_val_transform_deterministic(self) -> None:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        t = _val_transform()
        r1 = t(image=img.copy())["image"]
        r2 = t(image=img.copy())["image"]
        assert torch.allclose(r1, r2)


# ═══════════════════════════════════════════════════════════════════════════════
# WeightedRandomSampler
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetSampler:
    def test_sampler_length_matches_dataset(self, tmp_path) -> None:
        splits_path, faces_root = _make_splits_json(tmp_path, n_real_train=5, n_fake_train=15, face_count=2)
        ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        sampler = get_sampler(ds)
        assert sampler.num_samples == len(ds)

    def test_sampler_weights_are_positive(self, tmp_path) -> None:
        splits_path, faces_root = _make_splits_json(tmp_path, n_real_train=5, n_fake_train=15, face_count=2)
        ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=False)
        sampler = get_sampler(ds)
        assert all(w > 0 for w in sampler.weights.tolist())
