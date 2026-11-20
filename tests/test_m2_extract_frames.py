"""
Tests for preprocessing/extract_frames.py — Milestone 2.

Uses synthetic OpenCV-based video creation for fast, isolated testing.
No real dataset videos are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepfake_detector.preprocessing.extract_frames import (
    _adaptive_frame_count,
    _evenly_spaced_indices,
    _extract_video_frames,
    _FRAME_EXT,
    _INDEX_FILENAME,
    _MAX_FRAMES,
    _SHORT_VIDEO_FRAMES,
    _LONG_VIDEO_FRAMES,
)


# ── Synthetic video fixture ───────────────────────────────────────────────────


def _make_synthetic_video(
    path: Path,
    num_frames: int = 60,
    fps: float = 30.0,
    width: int = 64,
    height: int = 64,
) -> float:
    """Write a minimal valid MP4 using OpenCV's VideoWriter.

    Returns the video duration in seconds.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        # Alternating grey frames (fast to generate)
        frame = np.full((height, width, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return num_frames / fps


# ── _adaptive_frame_count ─────────────────────────────────────────────────────


class TestAdaptiveFrameCount:
    def test_short_video_returns_100(self) -> None:
        assert _adaptive_frame_count(10.0) == _SHORT_VIDEO_FRAMES

    def test_boundary_at_20s_returns_150(self) -> None:
        assert _adaptive_frame_count(20.0) == _LONG_VIDEO_FRAMES

    def test_long_video_returns_150(self) -> None:
        assert _adaptive_frame_count(60.0) == _LONG_VIDEO_FRAMES

    def test_never_exceeds_max(self) -> None:
        assert _adaptive_frame_count(9999.0) <= _MAX_FRAMES

    def test_zero_duration_returns_short_count(self) -> None:
        # 0.0 < 20.0 → short
        assert _adaptive_frame_count(0.0) == _SHORT_VIDEO_FRAMES

    def test_just_below_threshold(self) -> None:
        assert _adaptive_frame_count(19.99) == _SHORT_VIDEO_FRAMES

    def test_just_above_threshold(self) -> None:
        assert _adaptive_frame_count(20.01) == _LONG_VIDEO_FRAMES


# ── _evenly_spaced_indices ────────────────────────────────────────────────────


class TestEvenlySpacedIndices:
    def test_count_matches_n(self) -> None:
        idx = _evenly_spaced_indices(100, 10)
        assert len(idx) == 10

    def test_always_includes_first_frame(self) -> None:
        idx = _evenly_spaced_indices(50, 5)
        assert idx[0] == 0

    def test_always_includes_last_frame(self) -> None:
        idx = _evenly_spaced_indices(50, 5)
        assert idx[-1] == 49

    def test_all_within_bounds(self) -> None:
        total, n = 73, 15
        idx = _evenly_spaced_indices(total, n)
        assert all(0 <= i < total for i in idx)

    def test_indices_are_sorted(self) -> None:
        idx = _evenly_spaced_indices(200, 30)
        assert idx == sorted(idx)

    def test_single_frame_returns_zero(self) -> None:
        assert _evenly_spaced_indices(100, 1) == [0]

    def test_n_exceeds_total_clamped(self) -> None:
        # When n > total, return total frames (all of them)
        idx = _evenly_spaced_indices(5, 10)
        assert len(idx) == 5
        assert all(0 <= i < 5 for i in idx)

    def test_raises_on_zero_total(self) -> None:
        with pytest.raises(ValueError, match="total must be > 0"):
            _evenly_spaced_indices(0, 5)

    def test_raises_on_zero_n(self) -> None:
        with pytest.raises(ValueError, match="n must be > 0"):
            _evenly_spaced_indices(100, 0)


# ── _extract_video_frames (integration) ──────────────────────────────────────


class TestExtractVideoFrames:
    @pytest.fixture()
    def frames_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "frames"
        root.mkdir()
        return root

    @pytest.fixture()
    def short_video(self, tmp_path: Path) -> tuple[Path, float]:
        """A 2-second 60-frame synthetic video (< 20s → 60 frames capped)."""
        p = tmp_path / "short_video.mp4"
        duration = _make_synthetic_video(p, num_frames=60, fps=30.0)
        return p, duration

    @pytest.fixture()
    def long_video(self, tmp_path: Path) -> tuple[Path, float]:
        """A 25-second 750-frame synthetic video (≥ 20s → 150 frames)."""
        p = tmp_path / "long_video.mp4"
        duration = _make_synthetic_video(p, num_frames=750, fps=30.0)
        return p, duration

    # ── Extraction correctness ────────────────────────────────────────────────

    def test_extracts_correct_frame_count_short(
        self, short_video, frames_root
    ) -> None:
        path, duration = short_video
        result = _extract_video_frames(path, "test_ds", "fake", frames_root, duration)
        assert result["status"] == "ok"
        out_dir = frames_root / "test_ds" / "fake" / path.stem
        frames = sorted(out_dir.glob(f"*{_FRAME_EXT}"))
        # Duration is 2s → _adaptive_frame_count = 100, but only 60 frames exist
        assert len(frames) == min(60, _SHORT_VIDEO_FRAMES)

    def test_extracts_correct_frame_count_long(
        self, long_video, frames_root
    ) -> None:
        path, duration = long_video
        result = _extract_video_frames(path, "test_ds", "real", frames_root, duration)
        assert result["status"] == "ok"
        out_dir = frames_root / "test_ds" / "real" / path.stem
        frames = sorted(out_dir.glob(f"*{_FRAME_EXT}"))
        assert len(frames) == _LONG_VIDEO_FRAMES

    def test_extracted_files_are_valid_jpegs(
        self, short_video, frames_root
    ) -> None:
        path, duration = short_video
        _extract_video_frames(path, "test_ds", "real", frames_root, duration)
        out_dir = frames_root / "test_ds" / "real" / path.stem
        for frame_file in out_dir.glob(f"*{_FRAME_EXT}"):
            img = cv2.imread(str(frame_file))
            assert img is not None, f"Invalid JPEG: {frame_file.name}"
            assert img.shape[2] == 3  # BGR channels

    # ── Index sentinel ────────────────────────────────────────────────────────

    def test_frame_index_json_written(self, short_video, frames_root) -> None:
        path, duration = short_video
        _extract_video_frames(path, "test_ds", "fake", frames_root, duration)
        index_path = frames_root / "test_ds" / "fake" / path.stem / _INDEX_FILENAME
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert data["video_id"] == path.stem
        assert data["dataset"] == "test_ds"
        assert data["label"] == "fake"
        assert "frame_count" in data
        assert "frame_indices" in data
        assert "extracted_at" in data

    # ── Cache behaviour ───────────────────────────────────────────────────────

    def test_second_run_returns_cached(self, short_video, frames_root) -> None:
        path, duration = short_video
        # First run
        r1 = _extract_video_frames(path, "test_ds", "real", frames_root, duration)
        assert r1["status"] == "ok"
        # Second run — must be cached
        r2 = _extract_video_frames(path, "test_ds", "real", frames_root, duration)
        assert r2["status"] == "cached"
        assert r2["frame_count"] == 0  # Cached run extracts nothing

    def test_cached_run_preserves_frame_count(
        self, short_video, frames_root
    ) -> None:
        """Cache hit must leave exactly the same number of frames on disk."""
        path, duration = short_video
        _extract_video_frames(path, "test_ds", "real", frames_root, duration)
        out_dir = frames_root / "test_ds" / "real" / path.stem
        frames_before = sorted(f.name for f in out_dir.glob(f"*{_FRAME_EXT}"))

        # Second run (cache hit)
        _extract_video_frames(path, "test_ds", "real", frames_root, duration)
        frames_after = sorted(f.name for f in out_dir.glob(f"*{_FRAME_EXT}"))

        # Frame files must be identical — no new extractions, no deletions
        assert frames_before == frames_after

    def test_cached_run_index_json_consistent(
        self, short_video, frames_root
    ) -> None:
        """frame_index.json frame_count must be consistent across two runs."""
        path, duration = short_video
        _extract_video_frames(path, "test_ds", "real", frames_root, duration)
        index_path = frames_root / "test_ds" / "real" / path.stem / _INDEX_FILENAME
        count_run1 = json.loads(index_path.read_text())["frame_count"]

        # Second run is cached — index is NOT rewritten; count must match
        _extract_video_frames(path, "test_ds", "real", frames_root, duration)
        count_run2 = json.loads(index_path.read_text())["frame_count"]
        assert count_run1 == count_run2

    # ── Error handling ────────────────────────────────────────────────────────

    def test_missing_video_returns_failed(self, frames_root, tmp_path) -> None:
        result = _extract_video_frames(
            tmp_path / "nonexistent.mp4", "test_ds", "real", frames_root, 5.0
        )
        assert result["status"] == "failed"
        assert "error" in result

    def test_corrupted_video_returns_failed(
        self, tmp_path, frames_root
    ) -> None:
        bad = tmp_path / "corrupt.mp4"
        bad.write_bytes(b"not a real video file")
        result = _extract_video_frames(bad, "test_ds", "real", frames_root, 5.0)
        assert result["status"] == "failed"
