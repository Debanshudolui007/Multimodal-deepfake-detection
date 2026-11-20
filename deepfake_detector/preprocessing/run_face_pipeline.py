"""
M3 Preprocessing Pipeline Orchestrator.

Runs the complete face + frequency + audio preprocessing pipeline
on all verified videos from the Milestone 2 frame extraction output.

Pipeline per video:
  1. Load frames from ``processed/frames/<dataset>/<label>/<video_id>/``
  2. Detect faces in every frame (RetinaFace → MTCNN fallback)
  3. Align each detected face (eye-landmark rotation)
  4. Crop + resize to 224×224 PNG (20% margin)
  5. Quality filter (blur / size / confidence / aspect)
  6. Generate frequency tensor (.npy) for each passing crop
  7. Extract audio (WAV + MFCC + Mel) from source video

After all videos are processed:
  8. Build 70/15/15 video-level train/val/test split → splits.json

Everything is cache-aware: a video whose ``face_index.json`` sentinel
already exists with matching file count is skipped entirely.

Outputs:
  processed/faces/<dataset>/<label>/<video_id>/face_XXXXX.png
  processed/faces/<dataset>/<label>/<video_id>/face_XXXXX_freq.npy
  processed/faces/<dataset>/<label>/<video_id>/face_index.json
  processed/audio/<dataset>/<label>/<video_id>/mfcc.npy
  processed/audio/<dataset>/<label>/<video_id>/mel.npy
  processed/audio/<dataset>/<label>/<video_id>/audio_meta.json
  processed/splits/splits.json
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FACE_EXT = ".png"
_FACE_INDEX = "face_index.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════


def run_face_pipeline(
    cfg: dict[str, Any],
    project_root: Path,
    num_workers: int = 1,
) -> dict[str, Any]:
    """Run the complete M3 preprocessing pipeline on all extracted frames.

    Args:
        cfg: Merged project configuration dict.
        project_root: Root of the ``deepfake_detector`` project.
        num_workers: Number of parallel worker processes for face detection.
            Keep at 1 on CPU to avoid memory pressure; increase on GPU systems.

    Returns:
        Summary dict: ``{processed, cached, failed, total_faces, elapsed_sec}``.
    """
    frames_root = project_root / "deepfake_detector" / "processed" / "frames"
    faces_root  = project_root / "deepfake_detector" / "processed" / "faces"
    audio_root  = project_root / "deepfake_detector" / "processed" / "audio"
    splits_root = project_root / "deepfake_detector" / "processed" / "splits"
    # Raw video roots — needed for audio extraction
    data_cfg = cfg.get("dataset", {})

    # Collect all verified video frame dirs
    video_tasks = _collect_video_tasks(frames_root, data_cfg, project_root)
    if not video_tasks:
        logger.warning("No frame directories found under: %s", frames_root)
        return {"processed": 0, "cached": 0, "failed": 0, "total_faces": 0, "elapsed_sec": 0}

    logger.info(
        "Face pipeline: %d videos | workers=%d", len(video_tasks), num_workers
    )

    # ── Face detection + crop + frequency (per-video) ─────────────────────────
    t0 = time.time()
    processed = cached = failed = total_faces = 0
    total_tasks = len(video_tasks)

    from deepfake_detector.utils.logger import write_live_status

    # Use workers=1 on CPU (multi-process causes TF/MTCNN init overhead per proc)
    for i, task in enumerate(video_tasks):
        # Write live status for EVERY video so the monitor updates smoothly
        write_live_status("Face+Freq", i + 1, total_tasks, extra={"video_current": i + 1, "video_total": total_tasks})
            
        try:
            result = _process_one_video(task, faces_root)
            if result["status"] == "cached":
                cached += 1
            elif result["status"] == "ok":
                processed += 1
                total_faces += result.get("faces_saved", 0)
            else:
                failed += 1
        except Exception as exc:
            logger.error("Pipeline error for %s: %s", task.get("video_id"), exc)
            failed += 1

    # ── Audio extraction (per-video, uses ffmpeg subprocess) ──────────────────
    audio_ok = audio_skip = audio_fail = 0
    from deepfake_detector.preprocessing.extract_audio import extract_audio_features

    for i, task in enumerate(video_tasks):
        # Write live status
        if i % max(1, total_tasks // 100) == 0 or i == total_tasks - 1:
            write_live_status("Audio", i + 1, total_tasks)
            
        if task["dataset"] != "fakeavceleb":
            audio_skip += 1
            continue

        video_path = Path(task["video_path"])
        try:
            r = extract_audio_features(
                video_path=video_path,
                output_dir=audio_root,
                dataset=task["dataset"],
                label=task["label"],
                video_id=task["video_id"],
            )
            if r["status"] == "cached":
                audio_skip += 1
            elif r["status"] == "ok":
                audio_ok += 1
            else:
                audio_fail += 1
        except Exception as exc:
            logger.warning("Audio extraction failed for %s: %s", task["video_id"], exc)
            audio_fail += 1

    logger.info(
        "Audio: ok=%d cached=%d no_audio/failed=%d",
        audio_ok, audio_skip, audio_fail,
    )

    # ── Dataset split ─────────────────────────────────────────────────────────
    from deepfake_detector.preprocessing.split_dataset import build_splits

    splits_path = splits_root / "splits.json"
    splits_root.mkdir(parents=True, exist_ok=True)

    train_r = cfg.get("dataset", {}).get("split", {}).get("train", 0.70)
    val_r   = cfg.get("dataset", {}).get("split", {}).get("val", 0.15)
    seed    = cfg.get("dataset", {}).get("split", {}).get("seed", 42)

    split_result = build_splits(faces_root, splits_path, train_ratio=train_r, val_ratio=val_r, seed=seed)
    logger.info(
        "Split: train=%d val=%d test=%d",
        split_result["summary"]["train"],
        split_result["summary"]["val"],
        split_result["summary"]["test"],
    )

    elapsed = time.time() - t0
    summary = {
        "processed": processed,
        "cached": cached,
        "failed": failed,
        "total_faces": total_faces,
        "audio_ok": audio_ok,
        "audio_skip": audio_skip,
        "audio_fail": audio_fail,
        "elapsed_sec": round(elapsed, 1),
        "splits_path": str(splits_path),
    }

    # Save pipeline summary
    summary_path = splits_root / "face_pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Face pipeline complete | faces_processed=%d cached=%d failed=%d "
        "total_faces=%d elapsed=%.1fs",
        processed, cached, failed, total_faces, elapsed,
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Per-video processing
# ═══════════════════════════════════════════════════════════════════════════════


def _process_one_video(
    task: dict[str, Any],
    faces_root: Path,
) -> dict[str, Any]:
    """Run detect → align → crop → quality filter → frequency for one video.

    Args:
        task: Dict with ``video_id``, ``dataset``, ``label``, ``frames_dir``.
        faces_root: Root output directory for face crops.

    Returns:
        Status dict: ``{status, video_id, faces_saved, faces_skipped}``.
    """
    from deepfake_detector.preprocessing.crop_faces import (
        is_cached, save_face_crops,
    )
    from deepfake_detector.preprocessing.detect_faces import detect_faces_in_video_frames
    from deepfake_detector.preprocessing.generate_frequency import process_face_directory
    from deepfake_detector.preprocessing.align_faces import align_face, build_alignment_metadata, save_metadata
    from deepfake_detector.preprocessing.quality_filter import filter_face

    import cv2

    video_id = task["video_id"]
    dataset  = task["dataset"]
    label    = task["label"]
    frames_dir = Path(task["frames_dir"])

    # ── Cache check ────────────────────────────────────────────────────────────
    if is_cached(faces_root, dataset, label, video_id):
        # Still regenerate frequency tensors if any .npy are missing
        face_dir = faces_root / dataset / label / video_id
        process_face_directory(face_dir, save_debug_images=False)
        return {"status": "cached", "video_id": video_id, "faces_saved": 0}

    # ── Step 1: Face detection ─────────────────────────────────────────────────
    detection_results = detect_faces_in_video_frames(frames_dir, min_confidence=0.85)
    if not detection_results:
        return {"status": "failed", "video_id": video_id, "error": "no frames"}

    # ── Step 2: Align + filter detections ─────────────────────────────────────
    # We need to build the detection result list that save_face_crops expects,
    # but with aligned images written to a temp-like structure.
    # Strategy: align in-memory and pass directly to crop.

    aligned_results: list[dict[str, Any]] = []
    metadata_records: list[dict[str, Any]] = []

    for idx, result in enumerate(detection_results):
        frame_path = frames_dir / result["frame_file"]
        image = cv2.imread(str(frame_path))
        if image is None:
            aligned_results.append({"frame_file": result["frame_file"], "detection": None})
            continue

        h, w = image.shape[:2]
        detection = result["detection"]

        if detection is None:
            # KEEP THE WHOLE FRAME IF NO FACE DETECTED
            result_copy = dict(result)
            result_copy["detection"] = {
                "bbox": [0, 0, w, h],
                "confidence": 0.0,
                "detector": "none (full frame)",
                "landmarks": []
            }
            result_copy["_aligned_image"] = image
            aligned_results.append(result_copy)
            continue

        # Align
        aligned_img, angle = align_face(image, detection)

        # Quality filter (on the full frame, using original bbox)
        report = filter_face(
            face_crop=aligned_img,
            detection=detection,
            frame_shape=(h, w),
        )

        if not report.passed:
            # KEEP THE WHOLE FRAME IF QUALITY FILTER FAILS
            result_copy = dict(result)
            result_copy["detection"] = {
                "bbox": [0, 0, w, h],
                "confidence": detection.get("confidence", 0.0),
                "detector": "none (filtered full frame)",
                "landmarks": []
            }
            result_copy["_aligned_image"] = image
            aligned_results.append(result_copy)
            continue

        # Write aligned frame temporarily back so save_face_crops can read it
        # Instead: pass the aligned image directly through the detection dict
        result_copy = dict(result)
        result_copy["_aligned_image"] = aligned_img  # Custom key for aligned path
        aligned_results.append(result_copy)

        metadata_records.append(
            build_alignment_metadata(idx, detection, angle)
        )

    # ── Step 3: Crop + save ────────────────────────────────────────────────────
    # save_face_crops normally reads from frames_dir, but we have aligned images
    # in memory. Use a custom in-memory cropper:
    face_dir = faces_root / dataset / label / video_id
    faces_saved, faces_skipped = _crop_and_save_aligned(
        aligned_results, face_dir, video_id, dataset, label
    )

    # Save alignment metadata
    if metadata_records:
        meta_path = face_dir / "alignment_metadata.json"
        save_metadata(metadata_records, meta_path)

    # ── Step 4: Frequency tensors ──────────────────────────────────────────────
    if faces_saved > 0:
        process_face_directory(face_dir, save_debug_images=False)

    return {
        "status": "ok",
        "video_id": video_id,
        "faces_saved": faces_saved,
        "faces_skipped": faces_skipped,
    }


def _crop_and_save_aligned(
    aligned_results: list[dict[str, Any]],
    face_dir: Path,
    video_id: str,
    dataset: str,
    label: str,
    margin: float = 0.20,
    target_size: int = 224,
) -> tuple[int, int]:
    """Crop faces from in-memory aligned images and save as PNG.

    Args:
        aligned_results: Per-frame dicts with optional ``_aligned_image`` key.
        face_dir: Output directory for this video's face crops.
        video_id: Source video ID.
        dataset: Dataset name.
        label: Ground-truth label.
        margin: Bounding box margin fraction.
        target_size: Output face image size.

    Returns:
        ``(faces_saved, faces_skipped)`` counts.
    """
    import json
    from datetime import datetime, timezone

    import cv2

    from deepfake_detector.preprocessing.crop_faces import crop_face

    face_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    face_records = []
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    for result in aligned_results:
        detection = result.get("detection")
        aligned_image = result.get("_aligned_image")

        if detection is None or aligned_image is None:
            skipped += 1
            continue

        bbox = detection["bbox"]
        face_img = crop_face(aligned_image, bbox, margin=margin, target_size=target_size)
        if face_img is None:
            skipped += 1
            continue

        out_path = face_dir / f"face_{saved:05d}.png"
        cv2.imwrite(str(out_path), face_img, encode_params)

        face_records.append({
            "face_file": out_path.name,
            "source_frame": result["frame_file"],
            "bbox": bbox,
            "confidence": detection.get("confidence", 0.0),
            "detector": detection.get("detector", "unknown"),
        })
        saved += 1

    # Write sentinel
    index_data = {
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
    index_path = face_dir / _FACE_INDEX
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    logger.debug("[%s/%s/%s] saved=%d skipped=%d", dataset, label, video_id, saved, skipped)
    return saved, skipped


# ═══════════════════════════════════════════════════════════════════════════════
# Video task discovery
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_video_tasks(
    frames_root: Path,
    data_cfg: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]]:
    """Enumerate all video frame directories and resolve source video paths.

    Args:
        frames_root: Root of extracted frames (``processed/frames/``).
        data_cfg: Dataset config dict (contains root paths per dataset).
        project_root: Project root directory.

    Returns:
        List of task dicts: ``{video_id, dataset, label, frames_dir, video_path}``.
    """
    tasks: list[dict[str, Any]] = []
    if not frames_root.exists():
        logger.error("Frames root not found: %s", frames_root)
        return tasks

    # Map dataset name → source video root from config
    dataset_roots: dict[str, Path] = {}
    for ds_name, ds_cfg in data_cfg.get("datasets", {}).items():
        root = Path(ds_cfg.get("root", ""))
        if root.exists():
            dataset_roots[ds_name] = root

    for dataset_dir in sorted(frames_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        ds_name = dataset_dir.name

        for label_dir in sorted(dataset_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name

            for video_dir in sorted(label_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                # Must have at least one frame
                if not any(video_dir.glob("frame_*.jpg")):
                    continue

                # Resolve source video path
                video_path = _find_source_video(
                    video_dir.name, label, ds_name,
                    dataset_roots.get(ds_name), data_cfg
                )

                tasks.append({
                    "video_id": video_dir.name,
                    "dataset": ds_name,
                    "label": label,
                    "frames_dir": str(video_dir),
                    "video_path": str(video_path) if video_path else "",
                })

    logger.info("Collected %d video tasks from frames root.", len(tasks))
    return tasks


def _find_source_video(
    video_id: str,
    label: str,
    dataset: str,
    dataset_root: Path | None,
    data_cfg: dict[str, Any],
) -> Path | None:
    """Locate the original video file for a given video_id.

    Args:
        video_id: Stem name of the video.
        label: ``"real"`` or ``"fake"``.
        dataset: Dataset name.
        dataset_root: Root directory of this dataset's videos.
        data_cfg: Full dataset config.

    Returns:
        Path to the source video file, or None if not found.
    """
    if dataset_root is None:
        return None

    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        candidate = dataset_root / label / (video_id + ext)
        if candidate.exists():
            return candidate
        # Also search subdirectories one level deep
        for sub in dataset_root.iterdir():
            if sub.is_dir():
                c2 = sub / (video_id + ext)
                if c2.exists():
                    return c2
    return None
