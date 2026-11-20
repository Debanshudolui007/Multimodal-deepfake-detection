"""
Tests for preprocessing/verify_dataset.py — dataset verification logic.

These tests use synthetic (mocked) data to avoid depending on the actual
35,000-video dataset, making the test suite fast and CI-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepfake_detector.preprocessing.verify_dataset import (
    _build_report,
    _evenly_spaced_indices,
    check_dataset_integrity,
)


# ── _evenly_spaced_indices ────────────────────────────────────────────────────


class TestEvenlySpacedIndices:
    def test_returns_correct_count(self) -> None:
        idx = _evenly_spaced_indices(100, 5)
        assert len(idx) == 5

    def test_starts_at_zero(self) -> None:
        idx = _evenly_spaced_indices(100, 5)
        assert idx[0] == 0

    def test_all_within_bounds(self) -> None:
        total, n = 47, 8
        idx = _evenly_spaced_indices(total, n)
        assert all(0 <= i < total for i in idx)

    def test_empty_on_zero_total(self) -> None:
        assert _evenly_spaced_indices(0, 5) == []

    def test_empty_on_zero_n(self) -> None:
        assert _evenly_spaced_indices(100, 0) == []

    def test_single_frame(self) -> None:
        assert _evenly_spaced_indices(1, 1) == [0]


# ── _build_report ─────────────────────────────────────────────────────────────


def _make_stat(
    dataset: str = "celebdf",
    label: str = "real",
    duration: float = 10.0,
    fps: float = 25.0,
    blur: float = 500.0,
    brightness: float = 128.0,
    has_audio: bool = True,
    resolution: str = "1920x1080",
    codec: str = "h264",
    size_bytes: int = 1_000_000,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "label": label,
        "duration_sec": duration,
        "fps": fps,
        "blur_score": blur,
        "avg_brightness": brightness,
        "has_audio": has_audio,
        "resolution": resolution,
        "codec": codec,
        "size_bytes": size_bytes,
        "frame_count": int(duration * fps),
        "width": int(resolution.split("x")[0]),
        "height": int(resolution.split("x")[1]),
    }


class TestBuildReport:
    @pytest.fixture()
    def sample_stats(self) -> list[dict[str, Any]]:
        return [
            _make_stat("celebdf", "real"),
            _make_stat("celebdf", "real"),
            _make_stat("celebdf", "fake", has_audio=False),
            _make_stat("ff_plus", "real"),
            _make_stat("ff_plus", "fake"),
        ]

    @pytest.fixture()
    def inventory(self) -> dict[str, Any]:
        return {
            "celebdf": {"real": [Path("a"), Path("b")], "fake": [Path("c")]},
            "ff_plus": {"real": [Path("d")], "fake": [Path("e")]},
        }

    def test_summary_counts_correct(self, sample_stats, inventory) -> None:
        report = _build_report(sample_stats, [], inventory, 1.0)
        assert report["summary"]["total_videos_valid"] == 5
        assert report["summary"]["total_videos_skipped"] == 0

    def test_per_dataset_real_fake_counts(self, sample_stats, inventory) -> None:
        report = _build_report(sample_stats, [], inventory, 1.0)
        celebdf = report["per_dataset"]["celebdf"]
        assert celebdf["real_count"] == 2
        assert celebdf["fake_count"] == 1
        assert celebdf["missing_audio_count"] == 1

    def test_skipped_recorded(self, sample_stats, inventory) -> None:
        skipped = [{"path": "bad.mp4", "dataset": "celebdf", "reason": "cannot_open"}]
        report = _build_report(sample_stats, skipped, inventory, 1.0)
        assert len(report["skipped"]) == 1

    def test_empty_stats_returns_empty_summary(self, inventory) -> None:
        report = _build_report([], [], inventory, 1.0)
        assert report["summary"] == {}


# ── check_dataset_integrity ───────────────────────────────────────────────────


class TestCheckDatasetIntegrity:
    def _make_report(self, real: int = 5, fake: int = 5, missing_audio: int = 0) -> dict[str, Any]:
        return {
            "summary": {"total_videos_valid": real + fake, "total_videos_skipped": 0},
            "per_dataset": {
                "celebdf": {
                    "real_count": real,
                    "fake_count": fake,
                    "total_videos": real + fake,
                    "missing_audio_count": missing_audio,
                }
            },
            "skipped": [],
        }

    def test_no_issues_on_balanced_dataset(self) -> None:
        report = self._make_report(real=100, fake=100)
        issues = check_dataset_integrity(report, fail_on_error=False)
        assert issues == []

    def test_warns_on_severe_imbalance(self) -> None:
        report = self._make_report(real=300, fake=5)
        issues = check_dataset_integrity(report, fail_on_error=False)
        assert any("imbalance" in m.lower() for m in issues)

    def test_critical_on_no_real(self) -> None:
        report = self._make_report(real=0, fake=50)
        issues = check_dataset_integrity(report, fail_on_error=False)
        assert any("CRITICAL" in m for m in issues)

    def test_critical_on_no_fake(self) -> None:
        report = self._make_report(real=50, fake=0)
        issues = check_dataset_integrity(report, fail_on_error=False)
        assert any("CRITICAL" in m for m in issues)

    def test_warns_on_majority_missing_audio(self) -> None:
        report = self._make_report(real=50, fake=50, missing_audio=60)
        issues = check_dataset_integrity(report, fail_on_error=False)
        assert any("audio" in m.lower() for m in issues)

    def test_raises_on_critical_with_fail_on_error(self) -> None:
        report = self._make_report(real=0, fake=50)
        with pytest.raises(RuntimeError, match="integrity check failed"):
            check_dataset_integrity(report, fail_on_error=True)
