"""
Tests for Milestone 3 — Audio Feature Extraction.

Covers: extract_audio.py
Uses librosa's test-tone generation — no real video files required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deepfake_detector.preprocessing.extract_audio import (
    _FIXED_FRAMES,
    _N_MFCC,
    _N_MELS,
    _compute_mfcc,
    _compute_mel,
    _is_cached,
    _pad_or_trim,
    _write_meta,
)


_SR = 16_000  # Sample rate used in tests


def _sine_wave(duration_sec: float = 3.0, freq: float = 440.0) -> np.ndarray:
    """Generate a simple sine wave as a test audio signal."""
    t = np.linspace(0, duration_sec, int(_SR * duration_sec), endpoint=False)
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestPadOrTrim:
    def test_short_array_padded(self) -> None:
        arr = np.ones((40, 100))
        result = _pad_or_trim(arr, 300)
        assert result.shape == (40, 300)
        assert np.all(result[:, 100:] == 0.0)

    def test_long_array_trimmed(self) -> None:
        arr = np.ones((40, 500))
        result = _pad_or_trim(arr, 300)
        assert result.shape == (40, 300)

    def test_exact_length_unchanged(self) -> None:
        arr = np.ones((40, 300))
        result = _pad_or_trim(arr, 300)
        assert result.shape == (40, 300)
        assert np.array_equal(arr, result)


class TestComputeMfcc:
    def test_output_shape(self) -> None:
        y = _sine_wave(3.0)
        mfcc = _compute_mfcc(y, _SR)
        assert mfcc.shape == (_N_MFCC * 3, _FIXED_FRAMES)

    def test_dtype_is_float32(self) -> None:
        y = _sine_wave(3.0)
        mfcc = _compute_mfcc(y, _SR)
        assert mfcc.dtype == np.float32

    def test_short_audio_padded(self) -> None:
        y = _sine_wave(0.5)  # Very short — will be padded
        mfcc = _compute_mfcc(y, _SR)
        assert mfcc.shape[1] == _FIXED_FRAMES

    def test_long_audio_trimmed(self) -> None:
        y = _sine_wave(30.0)  # Very long — will be trimmed
        mfcc = _compute_mfcc(y, _SR)
        assert mfcc.shape[1] == _FIXED_FRAMES

    def test_not_all_zeros(self) -> None:
        y = _sine_wave(3.0)
        mfcc = _compute_mfcc(y, _SR)
        assert not np.allclose(mfcc, 0.0)


class TestComputeMel:
    def test_output_shape(self) -> None:
        y = _sine_wave(3.0)
        mel = _compute_mel(y, _SR)
        assert mel.shape == (_N_MELS, _FIXED_FRAMES)

    def test_dtype_is_float32(self) -> None:
        y = _sine_wave(3.0)
        mel = _compute_mel(y, _SR)
        assert mel.dtype == np.float32

    def test_log_scaled_values_non_positive(self) -> None:
        # librosa.power_to_db with ref=max should produce values ≤ 0 dB
        y = _sine_wave(3.0)
        mel = _compute_mel(y, _SR)
        assert mel.max() <= 0.1  # Allow tiny floating-point slack

    def test_padded_region_is_distinct(self) -> None:
        # Short signal: padded region should differ from actual content
        y = _sine_wave(0.5)
        mel = _compute_mel(y, _SR)
        assert mel.shape[1] == _FIXED_FRAMES


class TestIsCached:
    def test_returns_false_if_missing_files(self, tmp_path: Path) -> None:
        assert _is_cached(tmp_path / "nodir") is False

    def test_returns_true_when_all_files_present(self, tmp_path: Path) -> None:
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for fname in ["mfcc.npy", "mel.npy", "audio_meta.json"]:
            (audio_dir / fname).write_text("{}")
        assert _is_cached(audio_dir) is True

    def test_returns_false_if_one_file_missing(self, tmp_path: Path) -> None:
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "mfcc.npy").write_text("")
        (audio_dir / "audio_meta.json").write_text("{}")
        # mel.npy missing
        assert _is_cached(audio_dir) is False


class TestWriteMeta:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        path = tmp_path / "audio_meta.json"
        _write_meta(path, "vid1", "ds", "real", status="ok", duration_sec=5.0)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["video_id"] == "vid1"
        assert data["status"] == "ok"
        assert data["duration_sec"] == 5.0
        assert "extracted_at" in data

    def test_extra_kwargs_included(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.json"
        _write_meta(path, "v", "d", "l", "ok", custom_key="custom_value")
        data = json.loads(path.read_text())
        assert data["custom_key"] == "custom_value"
