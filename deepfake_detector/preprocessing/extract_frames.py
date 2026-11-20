"""
Milestone 2 — Adaptive, Cache-Aware Frame Extraction.

Extracts frames from all verified videos at an adaptive rate:
  - Videos < 20 seconds  → 100 frames (evenly spaced)
  - Videos ≥ 20 seconds  → 150 frames (evenly spaced)
  - Hard maximum         → 150 frames

Key design properties:
  - **Cache-aware**: checks for a per-video ``frame_index.json`` sentinel
    file before processing. If it exists and the expected frame count
    matches, the video is skipped entirely.
  - **Resume-aware**: if only some frames were extracted (partial run),
    the module re-extracts all frames for that video from scratch to
    guarantee consistency.
  - **Multiprocessing**: uses ``ProcessPoolExecutor`` for parallelism.
  - **Logging**: every skip and failure is logged at the appropriate level.
  - **Deterministic**: frames are always extracted at the same indices
    for a given video, regardless of run order.

Output structure::

    processed/frames/
    └── <dataset>/
        └── <label>/
            └── <video_id>/
                ├── frame_00000.jpg
                ├── frame_00001.jpg
                ├── ...
                └── frame_index.json

Usage (via CLI):
    python deepfake_detector/main.py --preprocess
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)

# ── Adaptive frame count thresholds ───────────────────────────────────────────
_SHORT_VIDEO_THRESHOLD_SEC: float = 20.0
_SHORT_VIDEO_FRAMES: int = 25
_LONG_VIDEO_FRAMES: int = 37
_MAX_FRAMES: int = 37

# ── Frame file settings ────────────────────────────────────────────────────────
_FRAME_EXT = ".jpg"
_FRAME_QUALITY = 95          # JPEG quality (0-100)
_FRAME_PREFIX = "frame_"
_INDEX_FILENAME = "frame_index.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════


def run_frame_extraction(
    cfg: dict[str, Any],
    project_root: Path,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Extract frames from all videos in all discovered datasets.

    Reads the dataset inventory from ``processed/splits/dataset_report_full.json``
    (written by Milestone 1 verification). This ensures frame extraction only
    runs on videos that have already been verified as readable.

    Args:
        cfg: Merged config dict (must contain ``cfg["dataset"]``).
        project_root: Absolute path to the project root directory.
        num_workers: Number of parallel worker processes.

    Returns:
        Summary dictionary: total processed, skipped (cached), failed.
    """
    dataset_cfg = cfg["dataset"]
    splits_dir = project_root / dataset_cfg["processed"]["splits"]
    frames_root = project_root / dataset_cfg["processed"]["frames"]
    frames_root.mkdir(parents=True, exist_ok=True)

    # ── Load verified video inventory from M1 report ──────────────────────────
    full_report_path = splits_dir / "dataset_report_full.json"
    if not full_report_path.exists():
        raise FileNotFoundError(
            f"dataset_report_full.json not found at {full_report_path}. "
            "Please run --verify (Milestone 1) first."
        )

    with full_report_path.open("r", encoding="utf-8") as f:
        full_report: dict[str, Any] = json.load(f)

    video_stats: list[dict[str, Any]] = full_report.get("video_stats", [])
    if not video_stats:
        raise ValueError("No video records found in dataset_report_full.json.")

    logger.info("Extracting frames for %d verified videos…", len(video_stats))
    t_start = time.time()

    # ── Dispatch extraction jobs ───────────────────────────────────────────────
    results = {"processed": 0, "cached": 0, "failed": 0, "total_frames": 0}
    jobs = [
        (
            Path(stat["path"]),
            stat["dataset"],
            stat["label"],
            frames_root,
            stat.get("duration_sec", 0.0),
        )
        for stat in video_stats
        if not stat.get("error")
    ]

    from deepfake_detector.utils.logger import write_live_status
    
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_extract_video_frames, path, ds, label, frames_root, dur): path
            for path, ds, label, frames_root, dur in jobs
        }
        
        completed_count = 0
        total_jobs = len(jobs)
        
        for fut in as_completed(futures):
            src_path = futures[fut]
            completed_count += 1
            
            # Write live status for monitor_training.py
            if completed_count % max(1, total_jobs // 100) == 0 or completed_count == total_jobs:
                write_live_status("ExtF", completed_count, total_jobs)
                
            try:
                result = fut.result()
                if result["status"] == "cached":
                    results["cached"] += 1
                elif result["status"] == "ok":
                    results["processed"] += 1
                    results["total_frames"] += result["frame_count"]
                else:
                    results["failed"] += 1
                    logger.warning(
                        "Frame extraction failed for %s: %s",
                        src_path.name,
                        result.get("error"),
                    )
            except Exception as exc:
                results["failed"] += 1
                logger.error("Unhandled error for %s: %s", src_path.name, exc)

    elapsed = time.time() - t_start
    logger.info(
        "Frame extraction complete in %.1fs | processed=%d cached=%d failed=%d total_frames=%d",
        elapsed,
        results["processed"],
        results["cached"],
        results["failed"],
        results["total_frames"],
    )
    results["elapsed_sec"] = round(elapsed, 1)

    # ── Write extraction summary ───────────────────────────────────────────────
    summary_path = splits_dir / "frame_extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Frame extraction summary saved: %s", summary_path)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Per-video extraction (runs in worker process)
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_video_frames(
    video_path: Path,
    dataset: str,
    label: str,
    frames_root: Path,
    duration_sec: float,
) -> dict[str, Any]:
    """Extract evenly-spaced frames from a single video.

    This function is designed to be called inside a worker process.

    Args:
        video_path: Absolute path to the video file.
        dataset: Parent dataset name (e.g. ``"celebdf"``).
        label: Ground-truth label (``"real"`` or ``"fake"``).
        frames_root: Root directory for all extracted frames.
        duration_sec: Pre-computed video duration in seconds.

    Returns:
        Dictionary with keys:
          - ``status``: ``"ok"`` | ``"cached"`` | ``"failed"``
          - ``frame_count``: number of frames extracted (0 if cached/failed)
          - ``error``: error message string (only present on failure)
    """
    video_id = video_path.stem
    out_dir = frames_root / dataset / label / video_id
    index_path = out_dir / _INDEX_FILENAME

    # ── Determine target frame count ──────────────────────────────────────────
    target_n = _adaptive_frame_count(duration_sec)

    # ── Cache check ───────────────────────────────────────────────────────────
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            saved_n = idx.get("frame_count", -1)
            frame_files = sorted(out_dir.glob(f"*{_FRAME_EXT}"))
            # Cache hit: the index says N frames were saved, and N files exist
            if saved_n > 0 and len(frame_files) == saved_n:
                return {"status": "cached", "frame_count": 0}
        except Exception:
            pass  # Index corrupted — re-extract

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "failed", "frame_count": 0, "error": "cannot_open"}

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0 or fps <= 0:
            return {
                "status": "failed",
                "frame_count": 0,
                "error": f"invalid_metadata fps={fps} frames={total_frames}",
            }

        # Clamp target to available frames
        # Extract exactly 1/5th of the total frames dynamically, as requested by the user
        # Added a hard maximum cap of 150 frames to prevent extracting thousands of frames for very long videos.
        target_n = min(150, max(1, total_frames // 5))
        n = min(target_n, total_frames)
        frame_indices = _evenly_spaced_indices(total_frames, n)

        # ── Extract ───────────────────────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)

        # Remove any stale frames from a partial previous run
        for stale in out_dir.glob(f"*{_FRAME_EXT}"):
            stale.unlink()

        extracted = 0
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, _FRAME_QUALITY]

        for seq_idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.debug(
                    "Skipped missing frame %d in %s", frame_idx, video_path.name
                )
                continue

            out_path = out_dir / f"{_FRAME_PREFIX}{seq_idx:05d}{_FRAME_EXT}"
            cv2.imwrite(str(out_path), frame, encode_params)
            extracted += 1

        # ── Write index sentinel ──────────────────────────────────────────────
        index_data: dict[str, Any] = {
            "video_id": video_id,
            "video_path": str(video_path),
            "dataset": dataset,
            "label": label,
            "frame_count": extracted,
            "target_frame_count": n,
            "frame_indices": frame_indices,
            "duration_sec": duration_sec,
            "extracted_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        index_path.write_text(
            json.dumps(index_data, indent=2), encoding="utf-8"
        )

        return {"status": "ok", "frame_count": extracted}

    except Exception as exc:
        return {"status": "failed", "frame_count": 0, "error": str(exc)}
    finally:
        cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities (also used by tests)
# ═══════════════════════════════════════════════════════════════════════════════


def _adaptive_frame_count(duration_sec: float) -> int:
    """Return the target frame count based on video duration.

    Implements the adaptive strategy from the architecture specification:
      - Duration < 20s  → 100 frames
      - Duration ≥ 20s  → 150 frames
      - Hard cap        → 150 frames

    Args:
        duration_sec: Video duration in seconds.

    Returns:
        Target number of frames to extract.
    """
    if duration_sec < _SHORT_VIDEO_THRESHOLD_SEC:
        count = _SHORT_VIDEO_FRAMES
    else:
        count = _LONG_VIDEO_FRAMES
    return min(count, _MAX_FRAMES)


def _evenly_spaced_indices(total: int, n: int) -> list[int]:
    """Return ``n`` evenly-spaced integer indices in ``[0, total)``.

    Uses linspace-style spacing to ensure full temporal coverage.

    Args:
        total: Total number of frames in the video.
        n: Number of frames to sample.

    Returns:
        Sorted list of integer frame indices.

    Raises:
        ValueError: If ``total <= 0`` or ``n <= 0``.
    """
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    n = min(n, total)
    if n == 1:
        return [0]

    # Compute evenly spaced float positions, then round to integers
    step = (total - 1) / (n - 1)
    indices = sorted({round(i * step) for i in range(n)})
    # Clamp to valid range (rounding may push the last index to `total`)
    indices = [min(idx, total - 1) for idx in indices]
    return indices[:n]
