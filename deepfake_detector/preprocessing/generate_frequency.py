"""
Milestone 3 — Frequency Feature Generation Module.

Generates a 3-channel frequency tensor from each 224×224 face crop:

  Channel 0 — FFT Magnitude (log-scaled):
    Captures global spectral content. Deepfake GAN upsampling leaves
    characteristic grid artefacts in the FFT magnitude spectrum.

  Channel 1 — DWT Low-Low (LL) Sub-band (db4, level 1):
    Captures low-frequency structural information and serves as an
    approximation coefficient for the Swin-T frequency backbone.

  Channel 2 — DWT Energy Map (normalised):
    Sum of squared coefficients across all DWT sub-bands (LL, LH, HL, HH)
    at level 1 — highlights localised frequency energy anomalies.

Output is a ``(3, 224, 224)`` float32 NumPy array saved as a ``.npy``
file alongside each face crop.

Optional debug PNG is saved when ``save_debug_images=True`` — useful for
visual inspection of frequency content during development.

References:
  - Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake
    Image Recognition". arXiv:2012.12035.
  - Qian et al. (2020) "Thinking in Frequency: Face Forgery Detection
    by Mining Frequency-aware Clues". ECCV 2020.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pywt

from deepfake_detector.utils.logger import write_live_status

logger = logging.getLogger(__name__)

# ── DWT settings ──────────────────────────────────────────────────────────────
_WAVELET = "db4"
_DWT_LEVEL = 1

# ── Output filenames ──────────────────────────────────────────────────────────
_FREQ_SUFFIX = "_freq.npy"
_DEBUG_SUFFIX = "_freq_debug.png"

# ── Expected input resolution ─────────────────────────────────────────────────
_INPUT_SIZE = 224


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def generate_frequency_tensor(
    face_crop: np.ndarray,
) -> np.ndarray:
    """Compute the 3-channel frequency tensor from a BGR face crop.

    Args:
        face_crop: BGR uint8 image of shape ``(H, W, 3)``. Should be
            224×224 but arbitrary sizes are accepted (centre-cropped or
            padded internally to ensure DWT level-1 works).

    Returns:
        Float32 NumPy array of shape ``(3, H, W)`` with channels:
          [0] FFT log-magnitude (normalised to [0, 1]).
          [1] DWT LL sub-band (normalised to [0, 1]).
          [2] DWT Energy map (normalised to [0, 1]).

    Raises:
        ValueError: If ``face_crop`` is None or has fewer than 2 dimensions.
    """
    if face_crop is None or face_crop.ndim < 2:
        raise ValueError("face_crop must be a valid 2D or 3D NumPy array.")

    gray = _to_grayscale_float(face_crop)

    ch_fft = _fft_magnitude(gray)
    ch_ll, ch_energy = _dwt_channels(gray)

    freq_tensor = np.stack([ch_fft, ch_ll, ch_energy], axis=0).astype(np.float32)
    return freq_tensor


def process_face_directory(
    face_dir: Path,
    save_debug_images: bool = False,
) -> dict[str, Any]:
    """Generate and save frequency tensors for all face PNGs in a directory.

    Args:
        face_dir: Directory containing ``face_XXXXX.png`` files.
        save_debug_images: If True, also write a ``*_freq_debug.png``
            side-by-side visualisation for inspection.

    Returns:
        Summary: ``{processed, skipped, errors, output_dir}``.
    """
    face_files = sorted(face_dir.glob("face_*.png"))
    processed = 0
    skipped = 0
    errors = 0
    
    total_jobs = len(face_files)
    completed_count = 0

    for face_path in face_files:
        npy_path = face_dir / (face_path.stem + _FREQ_SUFFIX)

        # Cache check — skip if .npy already exists
        if npy_path.exists():
            skipped += 1
            completed_count += 1
            continue

        img = cv2.imread(str(face_path))
        if img is None:
            logger.warning("Cannot read face crop: %s", face_path.name)
            errors += 1
            completed_count += 1
            continue

        try:
            tensor = generate_frequency_tensor(img)
            np.save(str(npy_path), tensor)
            processed += 1

            if save_debug_images:
                debug_path = face_dir / (face_path.stem + _DEBUG_SUFFIX)
                _save_debug_image(tensor, debug_path)

        except Exception as exc:
            logger.warning("Frequency generation failed for %s: %s", face_path.name, exc)
            errors += 1
        
        completed_count += 1
        if completed_count % max(1, total_jobs // 100) == 0 or completed_count == total_jobs:
            write_live_status("FFT", completed_count, total_jobs)

    logger.debug(
        "%s: freq processed=%d skipped=%d errors=%d",
        face_dir.name, processed, skipped, errors,
    )
    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "output_dir": str(face_dir),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Channel computations
# ═══════════════════════════════════════════════════════════════════════════════


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or greyscale uint8 image to a float64 greyscale array.

    Args:
        image: Input image (2D greyscale or 3D BGR uint8).

    Returns:
        Float64 greyscale array in [0, 1].
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return gray.astype(np.float64) / 255.0


def _fft_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute the log-scaled FFT magnitude spectrum.

    The DC component is shifted to the image centre. The log transform
    compresses the dynamic range to make artefacts visible.

    Args:
        gray: Float64 greyscale image in [0, 1].

    Returns:
        Normalised FFT magnitude as float32 in [0, 1].
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))  # log(1 + |F|) for numerical stability
    return _normalise(magnitude).astype(np.float32)


def _dwt_channels(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute DWT LL sub-band and per-pixel energy map.

    Applies a single-level 2D DWT (Daubechies-4) to the greyscale image.

    Args:
        gray: Float64 greyscale image in [0, 1].

    Returns:
        Tuple of ``(ll_channel, energy_channel)``, both float32 in [0, 1]
        resized to match the input spatial dimensions.
    """
    h, w = gray.shape
    coeffs = pywt.dwt2(gray, wavelet=_WAVELET)
    # pywt.dwt2 at level 1 returns (cA, (cH, cV, cD))
    ca, (ch, cv, cd) = coeffs  # type: ignore[misc]

    # LL (approximation) sub-band
    ll = _normalise(np.abs(ca)).astype(np.float32)
    ll_resized = cv2.resize(ll, (w, h), interpolation=cv2.INTER_LINEAR)

    # Energy map: per-pixel L2 across all sub-bands
    energy = np.sqrt(ca**2 + ch**2 + cv**2 + cd**2)
    energy = _normalise(energy).astype(np.float32)
    energy_resized = cv2.resize(energy, (w, h), interpolation=cv2.INTER_LINEAR)

    return ll_resized, energy_resized


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1].

    Args:
        arr: Input array of any shape.

    Returns:
        Normalised float64 array in [0, 1]. If min == max, returns zeros.
    """
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / (mx - mn)


def _save_debug_image(tensor: np.ndarray, path: Path) -> None:
    """Save a side-by-side visualisation of all three frequency channels.

    Args:
        tensor: ``(3, H, W)`` float32 tensor.
        path: Destination PNG path.
    """
    channels = []
    for i in range(3):
        ch = (tensor[i] * 255).clip(0, 255).astype(np.uint8)
        channels.append(cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR))
    debug = np.concatenate(channels, axis=1)
    cv2.imwrite(str(path), debug)
