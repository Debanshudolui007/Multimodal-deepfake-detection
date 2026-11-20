"""
Experiment folder management and dataset versioning manifests.

Every training run auto-creates a timestamped experiment directory
under ``experiments/`` and writes a ``manifest.json`` capturing all
configuration and environment details for full reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_experiment(
    experiments_dir: str | Path,
    name: str | None = None,
) -> Path:
    """Create a timestamped experiment directory.

    Args:
        experiments_dir: Root experiments folder.
        name: Optional human-readable prefix (e.g. "spatial_run").
            If None, only the timestamp is used.

    Returns:
        Path to the newly created experiment directory.

    Example:
        >>> exp_dir = create_experiment("deepfake_detector/experiments", "spatial")
        >>> # → deepfake_detector/experiments/spatial_20260721_134530/
    """
    experiments_dir = Path(experiments_dir)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder_name = f"{name}_{timestamp}" if name else timestamp
    exp_dir = experiments_dir / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Sub-directories
    for sub in ("config", "metrics", "checkpoints", "tensorboard", "evaluation"):
        (exp_dir / sub).mkdir(exist_ok=True)

    logger.info("Experiment directory created: %s", exp_dir)
    return exp_dir


def write_manifest(
    output_dir: str | Path,
    dataset_name: str,
    dataset_version: str,
    cfg: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a dataset/preprocessing versioning manifest to disk.

    Args:
        output_dir: Directory where ``manifest.json`` is saved.
        dataset_name: Human-readable dataset name (e.g. "CelebDF").
        dataset_version: Version string (e.g. "v2").
        cfg: The merged config dict (used to extract key parameters).
        extra: Any additional key/value pairs to include.

    Returns:
        Path to the written ``manifest.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    freq_cfg = cfg.get("frequency", {})

    manifest: dict[str, Any] = {
        "dataset": dataset_name,
        "version": dataset_version,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        # Key preprocessing parameters
        "resolution": dataset_cfg.get("manifest", {}).get("resolution", 224),
        "face_detector": dataset_cfg.get("manifest", {}).get("face_detector", "RetinaFace"),
        "wavelet": freq_cfg.get("wavelet", {}).get("family", "db4"),
        "wavelet_level": freq_cfg.get("wavelet", {}).get("level", 1),
        "spatial_backbone": model_cfg.get("spatial", {}).get("backbone", "resnext50_32x4d"),
        "sample_fraction": dataset_cfg.get("sample_fraction", 1.0),
        # Reproducibility
        "random_seed": dataset_cfg.get("random_seed", 42),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_hash": _get_git_hash(),
        # Config fingerprint
        "config_hash": _hash_dict(cfg),
    }

    if extra:
        manifest.update(extra)

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written: %s", manifest_path)
    return manifest_path


def _get_git_hash() -> str:
    """Retrieve the current Git commit hash, or 'unknown'.

    Returns:
        Short commit hash string, or ``"unknown"`` if Git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _hash_dict(d: dict[str, Any]) -> str:
    """Compute a short SHA-256 fingerprint of a config dictionary.

    Args:
        d: Dictionary to hash.

    Returns:
        First 12 characters of the hex digest.
    """
    serialised = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]
