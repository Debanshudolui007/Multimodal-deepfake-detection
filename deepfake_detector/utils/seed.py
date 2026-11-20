"""
Deterministic random seed management.

Provides a single entry point ``set_all_seeds`` that seeds Python's
built-in random module, NumPy, PyTorch (CPU and CUDA), and configures
cuDNN for full determinism.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Set deterministic seeds across all relevant RNG sources.

    Args:
        seed: Integer seed value. Default is 42.

    Example:
        >>> from deepfake_detector.utils.seed import set_all_seeds
        >>> set_all_seeds(42)
    """
    # Python built-in random
    random.seed(seed)

    # Environment variable for hash randomisation (Python 3.3+)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (no-op when no GPU is present)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # cuDNN determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("All RNG seeds set to %d", seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device.

    Args:
        preference: One of ``"auto"``, ``"cuda"``, or ``"cpu"``.
            ``"auto"`` selects CUDA if available, else CPU.

    Returns:
        A ``torch.device`` object.

    Raises:
        ValueError: If an invalid preference string is given.
    """
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preference in {"cuda", "cpu"}:
        device = torch.device(preference)
    else:
        raise ValueError(
            f"Invalid device preference '{preference}'. "
            "Choose from 'auto', 'cuda', 'cpu'."
        )

    logger.info("Using device: %s", device)
    return device
