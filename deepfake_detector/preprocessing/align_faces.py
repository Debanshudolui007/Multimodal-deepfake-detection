"""
Milestone 3 — Face Alignment Module.

Rotates detected faces to a canonical upright orientation using the
eye-landmark positions, then saves per-frame alignment metadata to JSON.

Alignment algorithm:
  1. Compute the angle between the left and right eye centres.
  2. Rotate the image around the midpoint of the eyes.
  3. The corrected image is returned for downstream cropping.

Metadata saved per frame (``<faces_dir>/<video_id>/metadata.json``):
  ``{ frame_index, bbox, landmarks, rotation_angle, confidence, detector }``

This metadata is valuable for:
  - Debugging face pipeline failures.
  - Paper figures showing alignment quality.
  - Reproducibility audits.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Index into 5-point landmark array
_LEFT_EYE_IDX = 0
_RIGHT_EYE_IDX = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def align_face(
    image: np.ndarray,
    detection: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Rotate an image so the detected face is horizontally aligned.

    Uses the left and right eye landmarks to compute the roll angle and
    applies a 2D affine rotation around the eye midpoint.

    Args:
        image: Full BGR frame as a NumPy uint8 array (H × W × 3).
        detection: DetectionResult dict from ``detect_faces.py``,
            must contain ``landmarks`` (list of 5 ``[x, y]`` points).

    Returns:
        Tuple of (rotated_image, rotation_angle_degrees).
        If landmarks are missing or degenerate, returns (original_image, 0.0).
    """
    landmarks = detection.get("landmarks", [])

    if len(landmarks) < 2:
        logger.debug("Insufficient landmarks for alignment — returning original")
        return image, 0.0

    left_eye = landmarks[_LEFT_EYE_IDX]
    right_eye = landmarks[_RIGHT_EYE_IDX]

    angle = _compute_rotation_angle(left_eye, right_eye)
    if abs(angle) < 0.5:
        # Already essentially horizontal — skip rotation to avoid artefacts
        return image, angle

    center = _eye_midpoint(left_eye, right_eye)
    rotated = _rotate_image(image, center, angle)
    return rotated, angle


def build_alignment_metadata(
    frame_index: int,
    detection: dict[str, Any],
    rotation_angle: float,
) -> dict[str, Any]:
    """Construct the per-frame alignment metadata dictionary.

    Args:
        frame_index: Sequential frame index (0-based).
        detection: DetectionResult from ``detect_faces.detect_largest_face``.
        rotation_angle: Rotation angle applied (degrees, CCW positive).

    Returns:
        Metadata dictionary ready for JSON serialisation.
    """
    return {
        "frame_index": frame_index,
        "bbox": detection.get("bbox", []),
        "landmarks": detection.get("landmarks", []),
        "rotation_angle": round(rotation_angle, 4),
        "confidence": round(float(detection.get("confidence", 0.0)), 6),
        "detector": detection.get("detector", "unknown"),
    }


def save_metadata(
    metadata_list: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write alignment metadata for all frames of a video to JSON.

    Args:
        metadata_list: List of per-frame metadata dicts.
        output_path: Destination file path (``metadata.json``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)
    logger.debug("Metadata saved: %s (%d frames)", output_path, len(metadata_list))


# ═══════════════════════════════════════════════════════════════════════════════
# Geometric helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_rotation_angle(
    left_eye: list[int],
    right_eye: list[int],
) -> float:
    """Compute the clockwise rotation angle to level the eye line.

    Args:
        left_eye: ``[x, y]`` pixel coordinates of the left eye.
        right_eye: ``[x, y]`` pixel coordinates of the right eye.

    Returns:
        Rotation angle in degrees (negative = rotate clockwise).
    """
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    return -angle  # Negate so that positive = left-eye-low rotates up


def _eye_midpoint(
    left_eye: list[int],
    right_eye: list[int],
) -> tuple[float, float]:
    """Return the midpoint between two eye coordinates.

    Args:
        left_eye: Left eye ``[x, y]``.
        right_eye: Right eye ``[x, y]``.

    Returns:
        ``(cx, cy)`` float tuple.
    """
    cx = (left_eye[0] + right_eye[0]) / 2.0
    cy = (left_eye[1] + right_eye[1]) / 2.0
    return cx, cy


def _rotate_image(
    image: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    """Apply a 2D affine rotation to an image.

    Args:
        image: BGR uint8 image array.
        center: ``(cx, cy)`` rotation centre in pixel coordinates.
        angle_deg: Counter-clockwise rotation angle in degrees.

    Returns:
        Rotated image (same size as input, borders filled with black).
    """
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rotated


def rotate_bbox(
    bbox: list[int],
    center: tuple[float, float],
    angle_deg: float,
    image_shape: tuple[int, int],
) -> list[int]:
    """Rotate a bounding box to match a rotated image.

    Transforms the four corners of the bounding box through the same
    rotation matrix used for the image, then re-computes the axis-aligned
    bounding box that contains all four corners.

    Args:
        bbox: ``[x1, y1, x2, y2]`` original bounding box.
        center: Rotation centre used for the image rotation.
        angle_deg: Counter-clockwise rotation in degrees.
        image_shape: ``(height, width)`` of the image.

    Returns:
        New ``[x1, y1, x2, y2]`` axis-aligned bounding box, clamped to
        the image dimensions.
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )
    M = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
    rotated_corners = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    new_x1 = int(np.clip(rotated_corners[:, 0].min(), 0, w - 1))
    new_y1 = int(np.clip(rotated_corners[:, 1].min(), 0, h - 1))
    new_x2 = int(np.clip(rotated_corners[:, 0].max(), 0, w))
    new_y2 = int(np.clip(rotated_corners[:, 1].max(), 0, h))
    return [new_x1, new_y1, new_x2, new_y2]
