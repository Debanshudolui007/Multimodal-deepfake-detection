"""
Milestone 3 — Quality Filter Module.

Discards low-quality face crops before they enter the training pipeline.
Quality is assessed on four independent axes:

1. **Blur score** — Laplacian variance of the greyscale crop.
   Below ``min_blur`` → reject (over-blurred).

2. **Size** — Face bounding-box side length relative to the original
   frame's shortest dimension.
   Below ``min_face_fraction`` → reject (too small to be informative).

3. **Confidence** — Detection confidence from the face detector.
   Below ``min_confidence`` → reject (uncertain detection).

4. **Aspect ratio** — Width-to-height ratio of the bounding box.
   Outside ``[min_aspect, max_aspect]`` → reject (implausibly shaped).

A ``FilterReport`` is returned for every crop, enabling downstream
analysis of which filters fire most often.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Default thresholds ────────────────────────────────────────────────────────
_DEFAULT_MIN_BLUR: float = 50.0        # Laplacian variance floor
_DEFAULT_MIN_FACE_FRACTION: float = 0.05  # Min fraction of frame shortest dim
_DEFAULT_MIN_CONFIDENCE: float = 0.85
_DEFAULT_MIN_ASPECT: float = 0.5       # width / height
_DEFAULT_MAX_ASPECT: float = 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# Data class
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FilterReport:
    """Holds quality assessment results for one face crop.

    Attributes:
        passed: True if the crop passed all filters.
        blur_score: Laplacian variance of the greyscale crop.
        face_fraction: Detected face area as a fraction of the frame.
        confidence: Detector confidence score.
        aspect_ratio: Bounding-box width / height.
        rejection_reasons: List of filter names that triggered rejection.
    """

    passed: bool = True
    blur_score: float = 0.0
    face_fraction: float = 0.0
    confidence: float = 0.0
    aspect_ratio: float = 1.0
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of this report.
        """
        return {
            "passed": self.passed,
            "blur_score": round(self.blur_score, 2),
            "face_fraction": round(self.face_fraction, 4),
            "confidence": round(self.confidence, 6),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "rejection_reasons": self.rejection_reasons,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def filter_face(
    face_crop: np.ndarray,
    detection: dict[str, Any],
    frame_shape: tuple[int, int],
    min_blur: float = _DEFAULT_MIN_BLUR,
    min_face_fraction: float = _DEFAULT_MIN_FACE_FRACTION,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    min_aspect: float = _DEFAULT_MIN_ASPECT,
    max_aspect: float = _DEFAULT_MAX_ASPECT,
) -> FilterReport:
    """Apply all quality filters to a single face crop.

    Args:
        face_crop: The cropped and resized face BGR image (any size).
        detection: DetectionResult dict (must contain ``bbox`` and
            ``confidence``).
        frame_shape: ``(height, width)`` of the original frame.
        min_blur: Minimum Laplacian variance. Lower → blurrier.
        min_face_fraction: Minimum ratio of face size to shortest frame dim.
        min_confidence: Minimum detector confidence.
        min_aspect: Minimum bounding-box aspect ratio (width/height).
        max_aspect: Maximum bounding-box aspect ratio (width/height).

    Returns:
        A ``FilterReport`` instance describing all filter outcomes.
    """
    report = FilterReport()
    frame_h, frame_w = frame_shape
    bbox = detection.get("bbox", [0, 0, 0, 0])
    confidence = float(detection.get("confidence", 0.0))

    # ── Blur ──────────────────────────────────────────────────────────────────
    blur_score = _laplacian_variance(face_crop)
    report.blur_score = blur_score
    if blur_score < min_blur:
        report.passed = False
        report.rejection_reasons.append(f"blur:{blur_score:.1f}<{min_blur}")

    # ── Face size fraction ────────────────────────────────────────────────────
    x1, y1, x2, y2 = bbox
    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)
    min_dim = min(frame_w, frame_h)
    face_fraction = max(face_w, face_h) / max(min_dim, 1)
    report.face_fraction = face_fraction
    if face_fraction < min_face_fraction:
        report.passed = False
        report.rejection_reasons.append(
            f"size:{face_fraction:.3f}<{min_face_fraction}"
        )

    # ── Confidence ────────────────────────────────────────────────────────────
    report.confidence = confidence
    if confidence < min_confidence:
        report.passed = False
        report.rejection_reasons.append(
            f"confidence:{confidence:.3f}<{min_confidence}"
        )

    # ── Aspect ratio ─────────────────────────────────────────────────────────
    aspect_ratio = face_w / max(face_h, 1)
    report.aspect_ratio = aspect_ratio
    if not (min_aspect <= aspect_ratio <= max_aspect):
        report.passed = False
        report.rejection_reasons.append(
            f"aspect:{aspect_ratio:.2f} not in [{min_aspect},{max_aspect}]"
        )

    return report


def summarise_filter_stats(
    reports: list[FilterReport],
) -> dict[str, Any]:
    """Aggregate filter outcomes across multiple crops into summary stats.

    Args:
        reports: List of FilterReport instances from ``filter_face``.

    Returns:
        Dictionary with pass rate, average metrics, and per-filter counts.
    """
    if not reports:
        return {"total": 0, "passed": 0, "pass_rate": 0.0}

    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    rejection_counts: dict[str, int] = {}
    for r in reports:
        for reason in r.rejection_reasons:
            key = reason.split(":")[0]
            rejection_counts[key] = rejection_counts.get(key, 0) + 1

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4),
        "avg_blur_score": round(float(np.mean([r.blur_score for r in reports])), 2),
        "avg_confidence": round(float(np.mean([r.confidence for r in reports])), 4),
        "avg_face_fraction": round(float(np.mean([r.face_fraction for r in reports])), 4),
        "rejection_counts": rejection_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _laplacian_variance(image: np.ndarray) -> float:
    """Compute the Laplacian variance as a blur sharpness metric.

    Higher values → sharper image.

    Args:
        image: BGR or greyscale uint8 image.

    Returns:
        Laplacian variance as a float. Returns 0.0 if image is invalid.
    """
    if image is None or image.size == 0:
        return 0.0
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
