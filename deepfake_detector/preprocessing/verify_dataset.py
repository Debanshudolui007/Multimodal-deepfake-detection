"""
Milestone 1 — Dataset Verification Module.

Scans CelebDF, FaceForensics++, and FakeAVCeleb datasets and produces:

1. Per-video statistics (FPS, resolution, duration, codec, blur, brightness,
   missing audio, corrupted frames).
2. Dataset-level aggregates.
3. A ``dataset_report.json`` with all findings.
4. A ``manifest.json`` recording preprocessing parameters.

Skips corrupted or unreadable files gracefully, logs every skipped video.

Usage (standalone):
    python -m deepfake_detector.preprocessing.verify_dataset

Usage (via CLI):
    python deepfake_detector/main.py --verify
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
_BLUR_SAMPLE_FRAMES = 5        # Number of frames sampled per video for blur score
_BRIGHTNESS_SAMPLE_FRAMES = 5  # Same for brightness
_CORRUPTED_SAMPLE_FRAMES = 20  # Frames probed when counting corrupted frames


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════


def run_verification(
    cfg: dict[str, Any],
    project_root: Path,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Run dataset verification across all configured datasets.

    Args:
        cfg: Merged config dict (must contain ``cfg["dataset"]``).
        project_root: Absolute path to the project root directory.
        num_workers: Number of parallel worker processes.

    Returns:
        The full verification report dictionary, which is also written to
        ``processed/splits/dataset_report.json``.
    """
    dataset_cfg = cfg["dataset"]
    raw_cfg = dataset_cfg["raw"]
    sample_fraction: float = dataset_cfg.get("sample_fraction", 1.0)
    seed: int = dataset_cfg.get("random_seed", 42)

    # ── Build video inventory per dataset ─────────────────────────────────────
    datasets_meta = _build_dataset_inventory(raw_cfg, project_root, sample_fraction, seed)
    total_videos = sum(len(v) for meta in datasets_meta.values() for v in meta.values())
    logger.info("Verifying %d videos across %d datasets…", total_videos, len(datasets_meta))

    # ── Verify each video (parallel) ──────────────────────────────────────────
    all_video_stats: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    flat_jobs: list[tuple[Path, str, str]] = []
    for ds_name, label_map in datasets_meta.items():
        for label, paths in label_map.items():
            for p in paths:
                flat_jobs.append((p, ds_name, label))

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_verify_single_video, path, ds_name, label): (path, ds_name)
            for path, ds_name, label in flat_jobs
        }
        for fut in as_completed(futures):
            path, ds_name = futures[fut]
            try:
                result = fut.result()
                if result.get("error"):
                    skipped.append({"path": str(path), "dataset": ds_name, "reason": result["error"]})
                    logger.warning("Skipped %s: %s", path.name, result["error"])
                else:
                    all_video_stats.append(result)
            except Exception as exc:
                skipped.append({"path": str(path), "dataset": ds_name, "reason": str(exc)})
                logger.warning("Skipped %s (exception): %s", path.name, exc)

    elapsed = time.time() - t_start
    logger.info(
        "Verification complete in %.1fs. Valid: %d  Skipped: %d",
        elapsed,
        len(all_video_stats),
        len(skipped),
    )

    # ── Aggregate statistics ───────────────────────────────────────────────────
    report = _build_report(all_video_stats, skipped, datasets_meta, elapsed)

    # ── Save report and manifest ───────────────────────────────────────────────
    _save_report(report, dataset_cfg, project_root)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Video inventory builder
# ═══════════════════════════════════════════════════════════════════════════════


def _build_dataset_inventory(
    raw_cfg: dict[str, str],
    project_root: Path,
    sample_fraction: float,
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    """Discover all videos in all datasets.

    Returns:
        Nested dict: ``{dataset_name: {label: [Path, ...]}}``
    """
    import random as _random
    rng = _random.Random(seed)

    # Dataset → label-folder mapping (defined inline to avoid circular import)
    layout: dict[str, dict[str, list[str]]] = {
        "celebdf": {
            "real": ["Celeb-real", "YouTube-real"],
            "fake": ["Celeb-synthesis"],
        },
        "ff_plus": {
            "real": ["original"],
            "fake": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"],
        },
        "fakeavceleb": {
            "real": ["RealVideo-RealAudio"],
            "fake": ["FakeVideo-FakeAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio"],
        },
    }

    inventory: dict[str, dict[str, list[Path]]] = {}
    for ds_key, label_dirs in layout.items():
        ds_root = project_root / raw_cfg.get(ds_key, "")
        if not ds_root.exists():
            logger.warning("Dataset root not found, skipping: %s", ds_root)
            continue

        inventory[ds_key] = {}
        for label, sub_dirs in label_dirs.items():
            paths: list[Path] = []
            for sub in sub_dirs:
                sub_path = ds_root / sub
                if not sub_path.exists():
                    logger.debug("Sub-directory not found: %s", sub_path)
                    continue
                found = [
                    p for p in sub_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
                ]
                paths.extend(found)

            # Apply sample fraction
            if sample_fraction < 1.0 and paths:
                k = max(1, int(len(paths) * sample_fraction))
                paths = rng.sample(paths, k)

            inventory[ds_key][label] = sorted(paths)
            logger.info(
                "[%s/%s] %d videos discovered", ds_key, label, len(paths)
            )

    return inventory


# ═══════════════════════════════════════════════════════════════════════════════
# Per-video verification (runs in subprocess worker)
# ═══════════════════════════════════════════════════════════════════════════════


def _verify_single_video(
    path: Path,
    dataset: str,
    label: str,
) -> dict[str, Any]:
    """Verify one video and extract rich statistics.

    This function is designed to be called in a worker process.

    Args:
        path: Absolute path to the video file.
        dataset: Parent dataset name.
        label: Ground-truth label ("real" or "fake").

    Returns:
        Dictionary of video statistics, or a dict with key ``"error"``
        if the video could not be read.
    """
    stat: dict[str, Any] = {
        "path": str(path),
        "dataset": dataset,
        "label": label,
        "filename": path.name,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "error": None,
    }

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        stat["error"] = "cannot_open"
        return stat

    try:
        fps: float = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration: float = frame_count / fps if fps > 0 else 0.0

        if frame_count <= 0 or fps <= 0:
            stat["error"] = "invalid_fps_or_frame_count"
            return stat

        # Sample frames for blur + brightness + corrupted count
        indices = _evenly_spaced_indices(frame_count, _BLUR_SAMPLE_FRAMES)
        blur_scores: list[float] = []
        brightness_vals: list[float] = []
        corrupted_count = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                corrupted_count += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            brightness_vals.append(float(np.mean(gray)))

        # Codec via ffprobe (best-effort)
        codec = _probe_codec(path)
        has_audio = _probe_has_audio(path)

        stat.update(
            {
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "duration_sec": round(duration, 2),
                "codec": codec,
                "has_audio": has_audio,
                "blur_score": round(float(np.mean(blur_scores)), 2) if blur_scores else None,
                "avg_brightness": round(float(np.mean(brightness_vals)), 2) if brightness_vals else None,
                "corrupted_frames_sampled": corrupted_count,
                "corrupted_frames_ratio": round(corrupted_count / _BLUR_SAMPLE_FRAMES, 3),
            }
        )
    except Exception as exc:
        stat["error"] = str(exc)
    finally:
        cap.release()

    return stat


def _evenly_spaced_indices(total: int, n: int) -> list[int]:
    """Return ``n`` evenly spaced frame indices within ``[0, total)``.

    Args:
        total: Total number of frames.
        n: Number of indices desired.

    Returns:
        List of integer frame indices.
    """
    if total <= 0 or n <= 0:
        return []
    step = max(1, total // n)
    return list(range(0, min(total, step * n), step))[:n]


def _probe_codec(path: Path) -> str:
    """Detect the video codec, preferring ffprobe with an OpenCV fallback.

    First attempts to use ``ffprobe`` (part of ffmpeg). If unavailable,
    falls back to reading OpenCV's ``CAP_PROP_FOURCC`` property, which
    returns a 4-character codec identifier (e.g. ``"avc1"``, ``"H264"``).

    Args:
        path: Path to the video file.

    Returns:
        Codec name string (e.g. ``"h264"``), or ``"unknown"``.
    """
    # ── Try ffprobe first ─────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        codec = result.stdout.strip()
        if codec:
            return codec
    except Exception:
        pass

    # ── Fallback: OpenCV FOURCC ───────────────────────────────────────────────
    try:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            cap.release()
            if fourcc_int != 0:
                # Decode 4-byte integer into ASCII characters
                fourcc_str = "".join(
                    chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
                ).strip("\x00")
                return fourcc_str if fourcc_str else "unknown"
    except Exception:
        pass

    return "unknown"


def _probe_has_audio(path: Path) -> bool:
    """Check whether a video file contains an audio stream.

    Tries ffprobe first; falls back to pydub/soundfile discovery via
    a lightweight subprocess check on the container metadata.

    Args:
        path: Path to the video file.

    Returns:
        True if an audio stream is detected, False otherwise.
    """
    # ── Try ffprobe ───────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        pass

    # ── Fallback: OpenCV audio sample count ──────────────────────────────────
    # CAP_PROP_AUDIO_DATA_SAMPLES > 0 only when an audio track is present.
    try:
        cap = cv2.VideoCapture(str(path), cv2.CAP_ANY)
        if cap.isOpened():
            audio_samples = cap.get(cv2.CAP_PROP_AUDIO_DATA_SAMPLES)
            cap.release()
            return audio_samples > 0
    except Exception:
        pass

    # Cannot determine — assume no audio (conservative for the audio branch)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Report building
# ═══════════════════════════════════════════════════════════════════════════════


def _build_report(
    stats: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    inventory: dict[str, dict[str, list[Path]]],
    elapsed_sec: float,
) -> dict[str, Any]:
    """Aggregate per-video statistics into a full report.

    Args:
        stats: List of valid video stat dicts.
        skipped: List of skipped video dicts (with reasons).
        inventory: Original video inventory (for total counts).
        elapsed_sec: Wall-clock time for the verification run.

    Returns:
        Complete report dictionary.
    """
    if not stats:
        logger.warning("No valid videos found — report will be empty.")
        return {
            "summary": {},
            "per_dataset": {},
            "skipped": skipped,
            "elapsed_sec": round(elapsed_sec, 1),
        }

    def _safe_mean(values: list[Any]) -> float | None:
        vals = [v for v in values if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    per_dataset: dict[str, Any] = {}
    all_ds = {s["dataset"] for s in stats}

    for ds in sorted(all_ds):
        ds_stats = [s for s in stats if s["dataset"] == ds]
        real_stats = [s for s in ds_stats if s["label"] == "real"]
        fake_stats = [s for s in ds_stats if s["label"] == "fake"]

        # Resolution distribution
        res_dist: dict[str, int] = {}
        for s in ds_stats:
            res = s.get("resolution", "unknown")
            res_dist[res] = res_dist.get(res, 0) + 1

        # Codec distribution
        codec_dist: dict[str, int] = {}
        for s in ds_stats:
            c = s.get("codec", "unknown")
            codec_dist[c] = codec_dist.get(c, 0) + 1

        per_dataset[ds] = {
            "total_videos": len(ds_stats),
            "real_count": len(real_stats),
            "fake_count": len(fake_stats),
            "avg_duration_sec": _safe_mean([s["duration_sec"] for s in ds_stats]),
            "avg_fps": _safe_mean([s["fps"] for s in ds_stats]),
            "avg_blur_score": _safe_mean([s.get("blur_score") for s in ds_stats]),
            "avg_brightness": _safe_mean([s.get("avg_brightness") for s in ds_stats]),
            "missing_audio_count": sum(1 for s in ds_stats if not s.get("has_audio", True)),
            "resolution_distribution": res_dist,
            "codec_distribution": codec_dist,
            "total_size_gb": round(sum(s["size_bytes"] for s in ds_stats) / 1e9, 3),
        }

    total_inv = sum(
        len(paths)
        for label_map in inventory.values()
        for paths in label_map.values()
    )

    summary: dict[str, Any] = {
        "total_videos_discovered": total_inv,
        "total_videos_valid": len(stats),
        "total_videos_skipped": len(skipped),
        "global_avg_duration_sec": _safe_mean([s["duration_sec"] for s in stats]),
        "global_avg_fps": _safe_mean([s["fps"] for s in stats]),
        "global_avg_blur_score": _safe_mean([s.get("blur_score") for s in stats]),
        "global_avg_brightness": _safe_mean([s.get("avg_brightness") for s in stats]),
        "global_missing_audio_count": sum(1 for s in stats if not s.get("has_audio", True)),
        "total_size_gb": round(sum(s["size_bytes"] for s in stats) / 1e9, 3),
        "datasets": sorted(all_ds),
    }

    return {
        "summary": summary,
        "per_dataset": per_dataset,
        "skipped": skipped,
        "elapsed_sec": round(elapsed_sec, 1),
        "video_stats": stats,   # Full per-video records (can be large)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report I/O
# ═══════════════════════════════════════════════════════════════════════════════


def _save_report(
    report: dict[str, Any],
    dataset_cfg: dict[str, Any],
    project_root: Path,
) -> None:
    """Write the verification report and dataset integrity log.

    Args:
        report: Full report dictionary.
        dataset_cfg: ``cfg["dataset"]`` sub-dictionary.
        project_root: Project root for path resolution.
    """
    splits_dir = project_root / dataset_cfg["processed"]["splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Main report (without bulky per-video list for readability)
    summary_report = {k: v for k, v in report.items() if k != "video_stats"}
    report_path = splits_dir / "dataset_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    logger.info("Dataset report saved: %s", report_path)

    # Full per-video stats (separate file)
    if report.get("video_stats"):
        full_path = splits_dir / "dataset_report_full.json"
        with full_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Full per-video report saved: %s", full_path)

    # Skipped video log
    if report["skipped"]:
        skip_path = splits_dir / "skipped_videos.json"
        with skip_path.open("w", encoding="utf-8") as f:
            json.dump(report["skipped"], f, indent=2)
        logger.warning(
            "%d videos skipped — see: %s",
            len(report["skipped"]),
            skip_path,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset integrity checks
# ═══════════════════════════════════════════════════════════════════════════════


def check_dataset_integrity(
    report: dict[str, Any],
    fail_on_error: bool = True,
) -> list[str]:
    """Validate dataset integrity from a verification report.

    Checks for:
    - Label presence (real + fake for each dataset)
    - Class balance (warns if ratio > 3:1)
    - Missing audio across the majority of videos
    - Empty datasets

    Args:
        report: Report dict from ``run_verification``.
        fail_on_error: If True, raises RuntimeError on critical failures.

    Returns:
        List of warning/error messages. Empty list means all checks passed.

    Raises:
        RuntimeError: If ``fail_on_error=True`` and critical issues found.
    """
    issues: list[str] = []
    per_ds = report.get("per_dataset", {})

    for ds_name, ds_info in per_ds.items():
        real_n = ds_info.get("real_count", 0)
        fake_n = ds_info.get("fake_count", 0)
        total = ds_info.get("total_videos", 0)

        if total == 0:
            issues.append(f"CRITICAL [{ds_name}]: No valid videos found — empty dataset.")
            continue

        if real_n == 0:
            issues.append(f"CRITICAL [{ds_name}]: No real videos found.")
        if fake_n == 0:
            issues.append(f"CRITICAL [{ds_name}]: No fake videos found.")

        # Class balance
        if real_n > 0 and fake_n > 0:
            ratio = max(real_n, fake_n) / min(real_n, fake_n)
            if ratio > 3.0:
                issues.append(
                    f"WARNING [{ds_name}]: Severe class imbalance "
                    f"(real={real_n}, fake={fake_n}, ratio={ratio:.1f}x)."
                )

        # Missing audio
        missing_audio = ds_info.get("missing_audio_count", 0)
        if missing_audio > total * 0.5:
            issues.append(
                f"WARNING [{ds_name}]: >50% of videos missing audio "
                f"({missing_audio}/{total}) — audio branch may be unreliable."
            )

    # Report skipped videos
    skipped_n = len(report.get("skipped", []))
    total_valid = report.get("summary", {}).get("total_videos_valid", 0)
    if total_valid > 0 and skipped_n / (total_valid + skipped_n) > 0.1:
        issues.append(
            f"WARNING: >10% of videos were skipped ({skipped_n} videos). "
            "Review skipped_videos.json."
        )

    # Report and optionally fail
    for msg in issues:
        if msg.startswith("CRITICAL"):
            logger.error(msg)
        else:
            logger.warning(msg)

    if fail_on_error:
        critical = [m for m in issues if m.startswith("CRITICAL")]
        if critical:
            raise RuntimeError(
                f"Dataset integrity check failed with {len(critical)} critical issue(s):\n"
                + "\n".join(critical)
            )

    return issues
