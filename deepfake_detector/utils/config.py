"""
YAML configuration loader with schema validation and path resolution.

Supports merging multiple YAML files, resolving all paths relative to the
project root, and validating required keys at load time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Required top-level keys per config file ───────────────────────────────────
_REQUIRED_KEYS: dict[str, list[str]] = {
    "dataset": ["raw", "processed", "split", "random_seed"],
    "train": ["random_seed", "spatial", "frequency", "audio", "fusion"],
    "model": ["spatial", "frequency", "audio", "fusion"],
    "frequency": ["wavelet", "fft", "channels", "output"],
    "audio": ["extraction", "mfcc", "mel", "output"],
    "fusion": ["attention", "classifier", "training", "ablation"],
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed YAML contents as a nested dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            contents: dict[str, Any] = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    logger.debug("Loaded config: %s", path)
    return contents


def load_all_configs(config_dir: str | Path) -> dict[str, Any]:
    """Load and merge all standard config files from a directory.

    Merges: dataset.yaml, train.yaml, model.yaml, frequency.yaml,
    audio.yaml, fusion.yaml into a single dict keyed by stem name.

    Args:
        config_dir: Directory containing the YAML config files.

    Returns:
        Dictionary with keys: "dataset", "train", "model",
        "frequency", "audio", "fusion".

    Raises:
        FileNotFoundError: If any expected config file is missing.
        ValueError: If required keys are absent.
    """
    config_dir = Path(config_dir)
    stems = ["dataset", "train", "model", "frequency", "audio", "fusion"]
    configs: dict[str, Any] = {}

    for stem in stems:
        cfg_path = config_dir / f"{stem}.yaml"
        cfg = load_yaml(cfg_path)
        _validate_required_keys(cfg, stem)
        configs[stem] = cfg

    logger.info("All configs loaded from: %s", config_dir)
    return configs


def _validate_required_keys(cfg: dict[str, Any], name: str) -> None:
    """Assert that a config dict contains all required top-level keys.

    Args:
        cfg: Loaded config dictionary.
        name: Config stem name (e.g. "dataset").

    Raises:
        ValueError: If any required key is missing.
    """
    required = _REQUIRED_KEYS.get(name, [])
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(
            f"Config '{name}.yaml' is missing required keys: {missing}"
        )


def resolve_paths(cfg: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Resolve all string values that look like relative paths.

    Walks the config dict recursively and converts any string value that
    does not start with '/' or a Windows drive letter into an absolute
    path anchored at ``project_root``.

    Args:
        cfg: Config dictionary (modified in-place).
        project_root: Project root directory.

    Returns:
        The same config dict with paths resolved.
    """
    root = Path(project_root)

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        if isinstance(obj, str) and _looks_like_path(obj):
            resolved = root / obj
            return str(resolved)
        return obj

    return _resolve(cfg)  # type: ignore[return-value]


def _looks_like_path(value: str) -> bool:
    """Heuristic: return True if string looks like a filesystem path.

    Args:
        value: String to inspect.

    Returns:
        True if the value appears to be a path.
    """
    if not value:
        return False
    # Starts with known dir prefix patterns used in our configs
    prefixes = (
        "dataset/", "deepfake_detector/", "experiments/",
        "processed/", "logs/", "results/", "research/",
        "checkpoints/",
    )
    return any(value.startswith(p) for p in prefixes)
