"""
Milestone 3 — Face Crop Module.

Crops the detected face region from an aligned frame with a configurable
margin, resizes to 224×224, and saves the result as a PNG file.

Output structure::

    processed/faces/
    └── <dataset>/
        └── <label>/
            └── <video_id>/
                ├── face_00000.png
                ├── face_00001.png
                └── face_index.json

The ``face_index.json`` sentinel file is the Milestone 3 cache token,
analogous to ``frame_index.json`` from Milestone 2.

Design:
  - Default margin: 20% of bounding-box dimensions on each side.
  - Output resolution: 224×224 pixels (configurable).
  - Format: PNG (lossless — required for frequency analysis accuracy).
  - Bounding box clamped to image edges before cropping.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Default constants ─────────────────────────────────────────────────────────
_DEFAULT_MARGIN: float = 0.20     # 20% margin on each side
_DEFAULT_SIZE: int = 224          # Target pixel size (H = W = 224)
_FACE_PREFIX = "face_"
_FACE_EXT = ".png"
_INDEX_FILENAME = "face_index.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def crop_face(
    image: np.ndarray,
    bbox: list[int],
    margin: float = _DEFAULT_MARGIN,
    target_size: int = _DEFAULT_SIZE,
) -> np.ndarray | None:
    """Crop and resize a single face from an image.

    Args:
        image: BGR uint8 image array (H × W × 3).
        bbox: ``[x1, y1, x2, y2]`` face bounding box in pixel coordinates.
        margin: Fractional margin to add around the bounding box on each
            side. E.g. ``0.20`` expands the crop by 20% of the box
            dimensions in each direction.
        target_size: Output face image size (square: H = W = target_size).

    Returns:
        Cropped and resized face image as a ``(target_size, target_size, 3)``
        uint8 BGR array, or ``None`` if the crop region is invalid (zero
        area after clamping).
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        logger.debug("Invalid bbox dimensions: %s", bbox)
        return None

    # Apply margin
    pad_x = int(box_w * margin)
    pad_y = int(box_h * margin)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    if cx2 <= cx1 or cy2 <= cy1:
        logger.debug("Crop region collapsed after clamping: (%d,%d,%d,%d)", cx1, cy1, cx2, cy2)
        return None

    crop = image[cy1:cy2, cx1:cx2]
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def save_face_crops(
    frames_dir: Path,
    detection_results: list[dict[str, Any]],
    output_dir: Path,
    video_id: str,
    dataset: str,
    label: str,
    margin: float = _DEFAULT_MARGIN,
    target_size: int = _DEFAULT_SIZE,
) -> dict[str, Any]:
    """Save face crops for all detected frames of a video.

    Applies crop_face() to each frame that has a valid detection, saves
    the PNG, and writes a ``face_index.json`` sentinel file.

    Args:
        frames_dir: Directory containing the raw extracted JPEG frames.
        detection_results: Per-frame detection dicts from
            ``detect_faces.detect_faces_in_video_frames``.
        output_dir: Root faces directory (``processed/faces/``).
        video_id: Stem of the source video filename.
        dataset: Dataset name (e.g. ``"celebdf"``).
        label: ``"real"`` or ``"fake"``.
        margin: Bounding-box margin fraction.
        target_size: Output face resolution.

    Returns:
        Summary dict: ``{video_id, saved, skipped, output_dir}``.
    """
    face_dir = output_dir / dataset / label / video_id
    face_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    face_records: list[dict[str, Any]] = []
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    for result in detection_results:
        frame_file = result["frame_file"]
        detection = result["detection"]

        if detection is None:
            skipped += 1
            continue

        frame_path = frames_dir / frame_file
        image = cv2.imread(str(frame_path))
        if image is None:
            logger.warning("Cannot read frame for crop: %s", frame_path)
            skipped += 1
            continue

        bbox = detection["bbox"]
        face_img = crop_face(image, bbox, margin=margin, target_size=target_size)
        if face_img is None:
            skipped += 1
            continue

        out_path = face_dir / f"{_FACE_PREFIX}{saved:05d}{_FACE_EXT}"
        cv2.imwrite(str(out_path), face_img, encode_params)

        face_records.append(
            {
                "face_file": out_path.name,
                "source_frame": frame_file,
                "bbox": bbox,
                "confidence": detection.get("confidence", 0.0),
                "detector": detection.get("detector", "unknown"),
            }
        )
        saved += 1

    # ── Write sentinel ────────────────────────────────────────────────────────
    index_data: dict[str, Any] = {
        "video_id": video_id,
        "dataset": dataset,
        "label": label,
        "face_count": saved,
        "skipped_frames": skipped,
        "margin": margin,
        "target_size": target_size,
        "faces": face_records,
        "extracted_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    index_path = face_dir / _INDEX_FILENAME
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    logger.debug(
        "[%s/%s/%s] crops saved=%d skipped=%d",
        dataset, label, video_id, saved, skipped,
    )
    return {
        "video_id": video_id,
        "saved": saved,
        "skipped": skipped,
        "output_dir": str(face_dir),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cache check utility
# ═══════════════════════════════════════════════════════════════════════════════


def is_cached(
    output_dir: Path,
    dataset: str,
    label: str,
    video_id: str,
) -> bool:
    """Check whether face crops for a video are already cached.

    A cache hit requires the ``face_index.json`` sentinel to exist *and*
    its recorded ``face_count`` to match the number of ``.png`` files
    actually present in the directory.

    Args:
        output_dir: Root faces directory (``processed/faces/``).
        dataset: Dataset name.
        label: ``"real"`` or ``"fake"``.
        video_id: Video stem name.

    Returns:
        True if the cache is valid and complete.
    """
    face_dir = output_dir / dataset / label / video_id
    index_path = face_dir / _INDEX_FILENAME

    if not index_path.exists():
        return False

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        saved_n = index.get("face_count", -1)
        actual_n = len(list(face_dir.glob(f"*{_FACE_EXT}")))
        return saved_n > 0 and actual_n == saved_n
    except Exception:
        return False
