"""
Tests for Milestone 3 — Face Pipeline.

Covers: detect_faces.py, align_faces.py, crop_faces.py, quality_filter.py.
Uses synthetic images — no real dataset required.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepfake_detector.preprocessing.align_faces import (
    align_face,
    build_alignment_metadata,
    _compute_rotation_angle,
    _eye_midpoint,
    rotate_bbox,
)
from deepfake_detector.preprocessing.crop_faces import (
    crop_face,
    is_cached,
    _DEFAULT_MARGIN,
    _DEFAULT_SIZE,
)
from deepfake_detector.preprocessing.detect_faces import (
    _bbox_area,
    _select_largest,
    _retinaface_landmarks,
)
from deepfake_detector.preprocessing.quality_filter import (
    FilterReport,
    _laplacian_variance,
    filter_face,
    summarise_filter_stats,
)


# ── Synthetic image helpers ───────────────────────────────────────────────────


def _make_face_image(size: int = 256, draw_eyes: bool = False) -> np.ndarray:
    """Create a synthetic BGR face-like image."""
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    # Draw a circle (face)
    cx, cy, r = size // 2, size // 2, size // 3
    cv2.circle(img, (cx, cy), r, (180, 160, 140), -1)
    if draw_eyes:
        eye_y = cy - r // 4
        cv2.circle(img, (cx - r // 3, eye_y), 8, (50, 50, 50), -1)
        cv2.circle(img, (cx + r // 3, eye_y), 8, (50, 50, 50), -1)
    return img


def _make_detection(
    x1=50, y1=50, x2=150, y2=150,
    confidence=0.95,
    detector="retinaface",
) -> dict:
    return {
        "bbox": [x1, y1, x2, y2],
        "landmarks": [
            [70, 80],   # left eye
            [130, 80],  # right eye
            [100, 100], # nose
            [80, 130],  # mouth left
            [120, 130], # mouth right
        ],
        "confidence": confidence,
        "detector": detector,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# detect_faces helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestBboxArea:
    def test_correct_area(self) -> None:
        assert _bbox_area([10, 10, 110, 60]) == 5000

    def test_zero_width(self) -> None:
        assert _bbox_area([0, 0, 0, 100]) == 0

    def test_inverted_coords_returns_zero(self) -> None:
        assert _bbox_area([100, 100, 50, 50]) == 0


class TestSelectLargest:
    def test_returns_largest(self) -> None:
        candidates = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.9},
            {"bbox": [0, 0, 50, 50], "confidence": 0.8},
            {"bbox": [0, 0, 30, 30], "confidence": 0.95},
        ]
        best = _select_largest(candidates)
        assert best["bbox"] == [0, 0, 50, 50]

    def test_empty_returns_none(self) -> None:
        assert _select_largest([]) is None

    def test_single_candidate_returned(self) -> None:
        c = [{"bbox": [1, 2, 3, 4], "confidence": 0.7}]
        assert _select_largest(c) is c[0]


class TestRetinaFaceLandmarks:
    def test_correct_order(self) -> None:
        lmk_dict = {
            "left_eye": [10.0, 20.0],
            "right_eye": [30.0, 20.0],
            "nose": [20.0, 30.0],
            "mouth_left": [12.0, 40.0],
            "mouth_right": [28.0, 40.0],
        }
        result = _retinaface_landmarks(lmk_dict)
        assert result[0] == [10, 20]
        assert result[1] == [30, 20]
        assert result[2] == [20, 30]
        assert len(result) == 5

    def test_missing_key_defaults_to_zero(self) -> None:
        result = _retinaface_landmarks({})
        assert all(pt == [0, 0] for pt in result)


# ═══════════════════════════════════════════════════════════════════════════════
# align_faces
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeRotationAngle:
    def test_horizontal_eyes_angle_near_zero(self) -> None:
        angle = _compute_rotation_angle([50, 100], [150, 100])
        assert abs(angle) < 1e-6

    def test_tilted_eyes_nonzero_angle(self) -> None:
        angle = _compute_rotation_angle([50, 90], [150, 110])
        assert abs(angle) > 0.1

    def test_angle_sign_convention(self) -> None:
        # Right eye lower → rotate clockwise (negative angle from our convention)
        angle = _compute_rotation_angle([50, 80], [150, 100])
        # left-eye high, right-eye low: dy > 0 → atan2 positive → -angle negative
        assert angle < 0


class TestEyeMidpoint:
    def test_midpoint_correct(self) -> None:
        cx, cy = _eye_midpoint([40, 60], [80, 100])
        assert cx == 60.0
        assert cy == 80.0


class TestAlignFace:
    def test_returns_image_of_same_size(self) -> None:
        img = _make_face_image()
        det = _make_detection()
        aligned, _ = align_face(img, det)
        assert aligned.shape == img.shape

    def test_near_zero_angle_returns_original(self) -> None:
        img = _make_face_image()
        det = _make_detection()
        det["landmarks"][0] = [70, 80]
        det["landmarks"][1] = [130, 80]   # Same height → ~0° angle
        aligned, angle = align_face(img, det)
        assert abs(angle) < 1.0

    def test_empty_landmarks_returns_original(self) -> None:
        img = _make_face_image()
        det = {"landmarks": [], "confidence": 0.9, "bbox": [0, 0, 100, 100]}
        aligned, angle = align_face(img, det)
        assert np.array_equal(aligned, img)
        assert angle == 0.0


class TestBuildAlignmentMetadata:
    def test_keys_present(self) -> None:
        det = _make_detection()
        meta = build_alignment_metadata(3, det, 5.2)
        assert meta["frame_index"] == 3
        assert abs(meta["rotation_angle"] - 5.2) < 1e-4
        assert "bbox" in meta
        assert "landmarks" in meta
        assert "confidence" in meta


class TestRotateBbox:
    def test_zero_rotation_unchanged(self) -> None:
        bbox = [50, 50, 150, 150]
        result = rotate_bbox(bbox, (100.0, 100.0), 0.0, (200, 200))
        assert result == bbox


# ═══════════════════════════════════════════════════════════════════════════════
# crop_faces
# ═══════════════════════════════════════════════════════════════════════════════


class TestCropFace:
    def test_output_is_target_size(self) -> None:
        img = _make_face_image(256)
        bbox = [50, 50, 200, 200]
        crop = crop_face(img, bbox, margin=0.20, target_size=224)
        assert crop is not None
        assert crop.shape == (224, 224, 3)

    def test_invalid_bbox_returns_none(self) -> None:
        img = _make_face_image(256)
        crop = crop_face(img, [100, 100, 50, 50])  # Inverted
        assert crop is None

    def test_zero_margin_does_not_crash(self) -> None:
        img = _make_face_image(256)
        crop = crop_face(img, [50, 50, 200, 200], margin=0.0, target_size=224)
        assert crop is not None
        assert crop.shape == (224, 224, 3)

    def test_margin_clamped_to_image_edge(self) -> None:
        img = _make_face_image(64)
        bbox = [0, 0, 60, 60]
        crop = crop_face(img, bbox, margin=0.5, target_size=224)
        assert crop is not None


class TestIsCached:
    def test_returns_false_if_no_directory(self, tmp_path) -> None:
        assert is_cached(tmp_path / "faces", "ds", "real", "vid1") is False

    def test_returns_true_when_index_and_files_match(self, tmp_path) -> None:
        face_dir = tmp_path / "faces" / "ds" / "real" / "vid1"
        face_dir.mkdir(parents=True)
        # Write 3 PNG files
        for i in range(3):
            (face_dir / f"face_{i:05d}.png").write_bytes(b"\x89PNG")
        # Write valid index
        index = {"face_count": 3}
        (face_dir / "face_index.json").write_text(json.dumps(index))
        assert is_cached(tmp_path / "faces", "ds", "real", "vid1") is True

    def test_returns_false_if_count_mismatch(self, tmp_path) -> None:
        face_dir = tmp_path / "faces" / "ds" / "real" / "vid1"
        face_dir.mkdir(parents=True)
        (face_dir / "face_00000.png").write_bytes(b"\x89PNG")
        (face_dir / "face_index.json").write_text(json.dumps({"face_count": 5}))
        assert is_cached(tmp_path / "faces", "ds", "real", "vid1") is False


# ═══════════════════════════════════════════════════════════════════════════════
# quality_filter
# ═══════════════════════════════════════════════════════════════════════════════


class TestLaplacianVariance:
    def test_sharp_image_higher_than_blurry(self) -> None:
        sharp = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        blurry = cv2.GaussianBlur(sharp, (31, 31), 5)
        assert _laplacian_variance(sharp) > _laplacian_variance(blurry)

    def test_constant_image_returns_near_zero(self) -> None:
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert _laplacian_variance(img) < 1.0

    def test_none_image_returns_zero(self) -> None:
        assert _laplacian_variance(None) == 0.0


class TestFilterFace:
    def test_good_face_passes_all_filters(self) -> None:
        img = _make_face_image(224, draw_eyes=True)
        det = _make_detection(x1=20, y1=20, x2=204, y2=204, confidence=0.96)
        report = filter_face(img, det, frame_shape=(224, 224))
        assert report.passed

    def test_blurry_face_rejected(self) -> None:
        img = cv2.GaussianBlur(_make_face_image(224), (61, 61), 20)
        det = _make_detection(x1=20, y1=20, x2=200, y2=200, confidence=0.96)
        report = filter_face(img, det, frame_shape=(224, 224), min_blur=500.0)
        assert not report.passed
        assert any("blur" in r for r in report.rejection_reasons)

    def test_low_confidence_rejected(self) -> None:
        img = _make_face_image(224, draw_eyes=True)
        det = _make_detection(confidence=0.3)
        report = filter_face(img, det, frame_shape=(224, 224), min_confidence=0.85)
        assert not report.passed
        assert any("confidence" in r for r in report.rejection_reasons)

    def test_small_face_rejected(self) -> None:
        img = _make_face_image(224)
        det = _make_detection(x1=100, y1=100, x2=110, y2=110, confidence=0.96)
        report = filter_face(
            img, det, frame_shape=(224, 224), min_face_fraction=0.5
        )
        assert not report.passed
        assert any("size" in r for r in report.rejection_reasons)

    def test_bad_aspect_ratio_rejected(self) -> None:
        img = _make_face_image(224)
        # Very wide bounding box (aspect ratio >> 2)
        det = _make_detection(x1=0, y1=100, x2=224, y2=110, confidence=0.96)
        report = filter_face(
            img, det, frame_shape=(224, 224), max_aspect=2.0
        )
        assert not report.passed
        assert any("aspect" in r for r in report.rejection_reasons)


class TestSummariseFilterStats:
    def test_pass_rate_correct(self) -> None:
        reports = [FilterReport(passed=True), FilterReport(passed=False), FilterReport(passed=True)]
        stats = summarise_filter_stats(reports)
        assert stats["total"] == 3
        assert stats["passed"] == 2
        assert abs(stats["pass_rate"] - 2 / 3) < 1e-4

    def test_empty_list(self) -> None:
        stats = summarise_filter_stats([])
        assert stats["total"] == 0

    def test_rejection_counts_aggregated(self) -> None:
        r = FilterReport(passed=False, rejection_reasons=["blur:10.0<50.0"])
        stats = summarise_filter_stats([r, FilterReport(passed=True)])
        assert stats["rejection_counts"]["blur"] == 1
