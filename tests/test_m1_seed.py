"""
Tests for utils/seed.py — deterministic seed management.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from deepfake_detector.utils.seed import get_device, set_all_seeds


class TestSetAllSeeds:
    def test_python_random_reproducible(self) -> None:
        set_all_seeds(42)
        a = [random.random() for _ in range(10)]
        set_all_seeds(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_random_reproducible(self) -> None:
        set_all_seeds(42)
        a = np.random.rand(10).tolist()
        set_all_seeds(42)
        b = np.random.rand(10).tolist()
        assert a == b

    def test_torch_random_reproducible(self) -> None:
        set_all_seeds(42)
        a = torch.rand(10).tolist()
        set_all_seeds(42)
        b = torch.rand(10).tolist()
        assert a == b

    def test_different_seeds_produce_different_values(self) -> None:
        set_all_seeds(1)
        a = torch.rand(5).tolist()
        set_all_seeds(99)
        b = torch.rand(5).tolist()
        assert a != b

    def test_cudnn_deterministic_set(self) -> None:
        set_all_seeds(0)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


class TestGetDevice:
    def test_auto_returns_device(self) -> None:
        d = get_device("auto")
        assert isinstance(d, torch.device)
        assert d.type in {"cuda", "cpu"}

    def test_cpu_preference(self) -> None:
        d = get_device("cpu")
        assert d.type == "cpu"

    def test_invalid_preference_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid device preference"):
            get_device("tpu")
