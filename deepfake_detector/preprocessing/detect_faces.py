"""
Milestone 3 — Face Detection Module.

Implements a robust, hierarchical face detection pipeline per frame:

  1. **RetinaFace** (via ``retina-face`` package) — primary detector.
  2. **MTCNN** (via ``facenet-pytorch``) — fallback when RetinaFace fails.
  3. **Skip** — frame is logged and discarded if all detectors fail.

For each detected face the module returns:
  - Bounding box ``[x1, y1, x2, y2]``
  - Facial landmarks (5-point: eyes, nose, mouth corners)
  - Detection confidence score

Only the **largest face** (by bounding-box area) is retained per frame,
matching the architecture specification.

Design principles:
  - Detectors are lazy-imported so the module can be imported without
    triggering TensorFlow/PyTorch initialisation unless detection runs.
  - Every detector failure is caught individually; the next detector
    in the chain is attempted automatically.
  - All public functions are type-annotated and Google-style docstrings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Type alias ────────────────────────────────────────────────────────────────
DetectionResult = dict[str, Any]
# Keys: bbox [x1,y1,x2,y2], landmarks [[x,y]×5], confidence, detector


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def detect_largest_face(
    image: np.ndarray,
    min_confidence: float = 0.85,
) -> DetectionResult | None:
    """Detect the largest face in an image using a fallback detector chain.

    Tries RetinaFace, then MTCNN. Returns ``None`` if all detectors fail
    or no face meets the confidence threshold.

    Args:
        image: BGR image as a NumPy uint8 array (H × W × 3).
        min_confidence: Minimum detection confidence to accept. Detections
            below this threshold are discarded.

    Returns:
        Dictionary with keys ``bbox``, ``landmarks``, ``confidence``,
        ``detector``; or ``None`` if no face is detected.
    """
    if image is None or image.size == 0:
        logger.debug("detect_largest_face received empty image")
        return None

    # Detector chain
    for detector_fn in (_try_retinaface, _try_mtcnn):
        try:
            result = detector_fn(image, min_confidence)
            if result is not None:
                return result
        except Exception as exc:
            logger.debug("Detector %s raised: %s", detector_fn.__name__, exc)

    logger.debug("All detectors failed — no face returned")
    return None


def detect_faces_in_video_frames(
    frames_dir: Path,
    min_confidence: float = 0.85,
) -> list[dict[str, Any]]:
    """Run face detection on all frames in a video's frame directory.

    Args:
        frames_dir: Directory containing ``frame_XXXXX.jpg`` files
            (written by Milestone 2 ``extract_frames``).
        min_confidence: Minimum detection confidence threshold.

    Returns:
        List of per-frame result dicts, each containing:
          - ``frame_file``: filename string
          - ``detection``: DetectionResult dict or None
    """
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        logger.warning("No frame files found in: %s", frames_dir)
        return []

    results: list[dict[str, Any]] = []
    for frame_path in frame_files:
        image = cv2.imread(str(frame_path))
        if image is None:
            logger.warning("Could not read frame: %s", frame_path.name)
            results.append({"frame_file": frame_path.name, "detection": None})
            continue

        detection = detect_largest_face(image, min_confidence)
        results.append({"frame_file": frame_path.name, "detection": detection})

    detected = sum(1 for r in results if r["detection"] is not None)
    logger.debug(
        "%s: %d/%d frames with faces", frames_dir.name, detected, len(results)
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Detector implementations (lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════════


def _try_retinaface(
    image: np.ndarray,
    min_confidence: float,
) -> DetectionResult | None:
    """Attempt face detection using the RetinaFace detector.

    Args:
        image: BGR uint8 image array.
        min_confidence: Minimum confidence threshold.

    Returns:
        DetectionResult for the largest face, or None.

    Raises:
        ImportError: If the ``retina-face`` package is not installed.
        Exception: On any internal RetinaFace error.
    """
    from RetinaFace import RetinaFace  # type: ignore

    # RetinaFace.detect_faces returns dict or 0 (no face)
    detections = RetinaFace.detect_faces(image, threshold=min_confidence)

    if not isinstance(detections, dict) or not detections:
        return None

    best = _select_largest(
        [
            {
                "bbox": list(map(int, face["facial_area"])),
                "landmarks": _retinaface_landmarks(face["landmarks"]),
                "confidence": float(face["score"]),
                "detector": "retinaface",
            }
            for face in detections.values()
            if face["score"] >= min_confidence
        ]
    )
    return best


def _try_mtcnn(
    image: np.ndarray,
    min_confidence: float,
) -> DetectionResult | None:
    """Attempt face detection using MTCNN from facenet-pytorch.

    Args:
        image: BGR uint8 image array.
        min_confidence: Minimum confidence threshold.

    Returns:
        DetectionResult for the largest face, or None.

    Raises:
        ImportError: If ``facenet-pytorch`` is not installed.
        Exception: On any internal MTCNN error.
    """
    import torch
    from facenet_pytorch import MTCNN  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lazy singleton — avoid re-instantiating MTCNN on every call
    if not hasattr(_try_mtcnn, "_instance"):
        _try_mtcnn._instance = MTCNN(  # type: ignore[attr-defined]
            keep_all=True,
            device=device,
            thresholds=[0.6, 0.7, min_confidence],
            min_face_size=20,
        )
    mtcnn: MTCNN = _try_mtcnn._instance  # type: ignore[attr-defined]

    # MTCNN expects RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

    if boxes is None or probs is None:
        return None

    candidates = []
    for box, prob, lmk in zip(boxes, probs, landmarks):
        if prob is None or float(prob) < min_confidence:
            continue
        x1, y1, x2, y2 = map(int, box)
        lmk_list = [[int(p[0]), int(p[1])] for p in lmk] if lmk is not None else []
        candidates.append(
            {
                "bbox": [x1, y1, x2, y2],
                "landmarks": lmk_list,
                "confidence": float(prob),
                "detector": "mtcnn",
            }
        )

    return _select_largest(candidates)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _select_largest(
    candidates: list[DetectionResult],
) -> DetectionResult | None:
    """Return the detection with the largest bounding-box area.

    Args:
        candidates: List of DetectionResult dicts (may be empty).

    Returns:
        The largest-area candidate, or None if the list is empty.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda d: _bbox_area(d["bbox"]))


def _bbox_area(bbox: list[int]) -> int:
    """Compute the pixel area of a bounding box.

    Args:
        bbox: ``[x1, y1, x2, y2]`` bounding box coordinates.

    Returns:
        Width × Height in pixels.
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _retinaface_landmarks(lmk_dict: dict[str, list[float]]) -> list[list[int]]:
    """Convert RetinaFace landmark dict to a flat list of [x, y] points.

    RetinaFace returns landmarks as:
    ``{"left_eye": [x,y], "right_eye": [x,y], "nose": [x,y],
       "mouth_left": [x,y], "mouth_right": [x,y]}``

    Args:
        lmk_dict: RetinaFace landmark dictionary.

    Returns:
        5-element list of ``[x, y]`` integer pairs in the standard
        MTCNN order: left_eye, right_eye, nose, mouth_left, mouth_right.
    """
    order = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    result = []
    for key in order:
        pt = lmk_dict.get(key, [0.0, 0.0])
        result.append([int(pt[0]), int(pt[1])])
    return result
