"""
Tests for Milestone 3 — Frequency Feature Generation.

Covers: generate_frequency.py
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from deepfake_detector.preprocessing.generate_frequency import (
    _fft_magnitude,
    _dwt_channels,
    _normalise,
    _to_grayscale_float,
    generate_frequency_tensor,
)


def _random_face(h: int = 224, w: int = 224) -> np.ndarray:
    """Create a random BGR face image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestToGrayscaleFloat:
    def test_bgr_to_float_in_range(self) -> None:
        img = _random_face(64, 64)
        gray = _to_grayscale_float(img)
        assert gray.ndim == 2
        assert gray.min() >= 0.0
        assert gray.max() <= 1.0

    def test_already_grayscale_unchanged_shape(self) -> None:
        gray_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        gray = _to_grayscale_float(gray_img)
        assert gray.shape == (64, 64)


class TestNormalise:
    def test_output_range_zero_to_one(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n = _normalise(arr)
        assert abs(n.min()) < 1e-6
        assert abs(n.max() - 1.0) < 1e-6

    def test_constant_array_returns_zeros(self) -> None:
        arr = np.ones((10, 10)) * 5.0
        n = _normalise(arr)
        assert np.allclose(n, 0.0)


class TestFftMagnitude:
    def test_output_shape_matches_input(self) -> None:
        gray = np.random.rand(128, 128)
        result = _fft_magnitude(gray)
        assert result.shape == (128, 128)

    def test_output_in_zero_one_range(self) -> None:
        gray = np.random.rand(64, 64)
        result = _fft_magnitude(gray)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_dtype_is_float32(self) -> None:
        gray = np.random.rand(64, 64)
        result = _fft_magnitude(gray)
        assert result.dtype == np.float32


class TestDwtChannels:
    def test_output_shapes_match_input(self) -> None:
        gray = np.random.rand(224, 224)
        ll, energy = _dwt_channels(gray)
        assert ll.shape == (224, 224)
        assert energy.shape == (224, 224)

    def test_ll_in_zero_one_range(self) -> None:
        gray = np.random.rand(64, 64)
        ll, _ = _dwt_channels(gray)
        assert ll.min() >= 0.0
        assert ll.max() <= 1.0 + 1e-6

    def test_energy_in_zero_one_range(self) -> None:
        gray = np.random.rand(64, 64)
        _, energy = _dwt_channels(gray)
        assert energy.min() >= 0.0
        assert energy.max() <= 1.0 + 1e-6

    def test_dtype_is_float32(self) -> None:
        gray = np.random.rand(64, 64)
        ll, energy = _dwt_channels(gray)
        assert ll.dtype == np.float32
        assert energy.dtype == np.float32

    def test_odd_dimension_image(self) -> None:
        """DWT should work on odd-dimensional inputs."""
        gray = np.random.rand(65, 65)
        ll, energy = _dwt_channels(gray)
        assert ll.shape == (65, 65)


class TestGenerateFrequencyTensor:
    def test_output_shape_is_3_h_w(self) -> None:
        face = _random_face(224, 224)
        tensor = generate_frequency_tensor(face)
        assert tensor.shape == (3, 224, 224)

    def test_output_dtype_float32(self) -> None:
        face = _random_face(224, 224)
        tensor = generate_frequency_tensor(face)
        assert tensor.dtype == np.float32

    def test_all_channels_in_zero_one(self) -> None:
        face = _random_face(224, 224)
        tensor = generate_frequency_tensor(face)
        for i in range(3):
            assert tensor[i].min() >= 0.0 - 1e-6
            assert tensor[i].max() <= 1.0 + 1e-6

    def test_three_channels_are_distinct(self) -> None:
        face = _random_face(224, 224)
        tensor = generate_frequency_tensor(face)
        # The three channels should not be identical
        assert not np.allclose(tensor[0], tensor[1])
        assert not np.allclose(tensor[1], tensor[2])

    def test_raises_on_none_input(self) -> None:
        with pytest.raises((ValueError, AttributeError)):
            generate_frequency_tensor(None)  # type: ignore

    def test_non_square_input_works(self) -> None:
        face = _random_face(128, 96)
        tensor = generate_frequency_tensor(face)
        assert tensor.shape == (3, 128, 96)

    def test_grayscale_input_works(self) -> None:
        gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        tensor = generate_frequency_tensor(gray)
        assert tensor.shape == (3, 224, 224)
