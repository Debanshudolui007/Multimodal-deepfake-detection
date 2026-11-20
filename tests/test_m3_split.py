"""
Tests for Milestone 3 — Dataset Splitting.

Covers: split_dataset.py
Uses a synthetic faces directory — no real data required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepfake_detector.preprocessing.split_dataset import (
    _allocate_split_counts,
    _check_no_leakage,
    _discover_videos,
    build_splits,
    load_splits,
)


# ── Synthetic faces directory fixture ────────────────────────────────────────


def _make_faces_dir(
    tmp_path: Path,
    datasets: dict[str, dict[str, int]],
) -> Path:
    """Create a synthetic faces directory structure with dummy PNG files.

    Args:
        tmp_path: pytest tmp_path fixture.
        datasets: ``{dataset: {label: n_videos}}``

    Returns:
        Path to the ``faces/`` root.
    """
    faces_root = tmp_path / "faces"
    for ds, labels in datasets.items():
        for label, n_videos in labels.items():
            for i in range(n_videos):
                vid_dir = faces_root / ds / label / f"video_{ds}_{label}_{i:04d}"
                vid_dir.mkdir(parents=True)
                # Minimum 1 PNG file per video dir (required by _discover_videos)
                (vid_dir / "face_00000.png").write_bytes(b"\x89PNG\r\n")
    return faces_root


# ═══════════════════════════════════════════════════════════════════════════════
# _discover_videos
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiscoverVideos:
    def test_discovers_correct_count(self, tmp_path: Path) -> None:
        faces = _make_faces_dir(tmp_path, {"celebdf": {"real": 10, "fake": 20}})
        videos = _discover_videos(faces)
        assert len(videos) == 30

    def test_skips_empty_directories(self, tmp_path: Path) -> None:
        faces_root = tmp_path / "faces"
        # Directory with no PNGs
        (faces_root / "ds" / "real" / "empty_vid").mkdir(parents=True)
        videos = _discover_videos(faces_root)
        assert len(videos) == 0

    def test_returns_correct_fields(self, tmp_path: Path) -> None:
        faces = _make_faces_dir(tmp_path, {"ff_plus": {"real": 1}})
        videos = _discover_videos(faces)
        assert len(videos) == 1
        v = videos[0]
        assert "video_id" in v
        assert "dataset" in v
        assert "label" in v
        assert "face_dir" in v
        assert "face_count" in v

    def test_nonexistent_root_returns_empty(self, tmp_path: Path) -> None:
        videos = _discover_videos(tmp_path / "nonexistent")
        assert videos == []


# ═══════════════════════════════════════════════════════════════════════════════
# _check_no_leakage
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckNoLeakage:
    def test_no_overlap_passes(self) -> None:
        splits = {
            "train": [{"video_id": "v1"}, {"video_id": "v2"}],
            "val": [{"video_id": "v3"}],
            "test": [{"video_id": "v4"}],
        }
        _check_no_leakage(splits)  # Should not raise

    def test_overlap_raises(self) -> None:
        splits = {
            "train": [{"video_id": "v1"}],
            "val": [{"video_id": "v1"}],  # Duplicate
        }
        with pytest.raises(AssertionError, match="Identity leakage"):
            _check_no_leakage(splits)


# ═══════════════════════════════════════════════════════════════════════════════
# build_splits
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildSplits:
    @pytest.fixture()
    def medium_faces(self, tmp_path: Path) -> Path:
        return _make_faces_dir(
            tmp_path,
            {
                "celebdf": {"real": 20, "fake": 100},
                "ff_plus": {"real": 30, "fake": 150},
                "fakeavceleb": {"real": 10, "fake": 80},
            },
        )

    def test_output_file_created(self, medium_faces, tmp_path) -> None:
        out = tmp_path / "splits.json"
        build_splits(medium_faces, out, train_ratio=0.7, val_ratio=0.15, seed=42)
        assert out.exists()

    def test_total_count_preserved(self, medium_faces, tmp_path) -> None:
        out = tmp_path / "splits.json"
        result = build_splits(medium_faces, out, train_ratio=0.7, val_ratio=0.15, seed=42)
        splits = result["splits"]
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == result["summary"]["total_videos"]

    def test_no_identity_leakage(self, medium_faces, tmp_path) -> None:
        out = tmp_path / "splits.json"
        result = build_splits(medium_faces, out, train_ratio=0.7, val_ratio=0.15, seed=42)
        _check_no_leakage(result["splits"])  # Must not raise

    def test_train_is_largest_split(self, medium_faces, tmp_path) -> None:
        out = tmp_path / "splits.json"
        result = build_splits(medium_faces, out, train_ratio=0.7, val_ratio=0.15, seed=42)
        splits = result["splits"]
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])

    def test_deterministic_with_same_seed(self, medium_faces, tmp_path) -> None:
        out1 = tmp_path / "s1.json"
        out2 = tmp_path / "s2.json"
        r1 = build_splits(medium_faces, out1, train_ratio=0.7, val_ratio=0.15, seed=42)
        r2 = build_splits(medium_faces, out2, train_ratio=0.7, val_ratio=0.15, seed=42)
        ids1 = sorted(v["video_id"] for v in r1["splits"]["train"])
        ids2 = sorted(v["video_id"] for v in r2["splits"]["train"])
        assert ids1 == ids2

    def test_different_seeds_produce_different_splits(
        self, medium_faces, tmp_path
    ) -> None:
        out1 = tmp_path / "s1.json"
        out2 = tmp_path / "s2.json"
        r1 = build_splits(medium_faces, out1, train_ratio=0.7, val_ratio=0.15, seed=1)
        r2 = build_splits(medium_faces, out2, train_ratio=0.7, val_ratio=0.15, seed=99)
        ids1 = sorted(v["video_id"] for v in r1["splits"]["train"])
        ids2 = sorted(v["video_id"] for v in r2["splits"]["train"])
        assert ids1 != ids2

    def test_invalid_ratios_raises(self, medium_faces, tmp_path) -> None:
        out = tmp_path / "s.json"
        with pytest.raises(ValueError, match="ratios"):
            build_splits(medium_faces, out, train_ratio=0.8, val_ratio=0.4)

    def test_empty_faces_dir_raises(self, tmp_path) -> None:
        empty = tmp_path / "empty_faces"
        empty.mkdir()
        out = tmp_path / "s.json"
        with pytest.raises(ValueError, match="No face-crop directories"):
            build_splits(empty, out)

    def test_minority_class_with_three_examples_reaches_every_split(self) -> None:
        assert _allocate_split_counts(3, 0.7, 0.15, 0.15) == (1, 1, 1)

    def test_minority_class_with_fourteen_examples_has_test_coverage(self) -> None:
        train, val, test = _allocate_split_counts(14, 0.7, 0.15, 0.15)
        assert train + val + test == 14
        assert min(train, val, test) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# load_splits
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadSplits:
    def test_loads_correct_structure(self, tmp_path: Path) -> None:
        faces = _make_faces_dir(tmp_path, {"ds": {"real": 5, "fake": 5}})
        out = tmp_path / "splits.json"
        build_splits(faces, out, train_ratio=0.7, val_ratio=0.15, seed=42)

        splits = load_splits(out)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_splits(tmp_path / "nonexistent.json")
