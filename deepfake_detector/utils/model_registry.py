"""
Model registry for tracking trained model metadata.

Maintains a JSON registry file that records every saved checkpoint
with its metrics, config hash, and timestamp. Enables structured
model management without relying on checkpoint filename conventions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "model_registry.json"


class ModelRegistry:
    """Persistent JSON registry for trained model checkpoints.

    Args:
        registry_dir: Directory where ``model_registry.json`` is stored.

    Example:
        >>> reg = ModelRegistry("deepfake_detector/checkpoints")
        >>> reg.register(
        ...     branch="spatial",
        ...     dataset="CelebDF",
        ...     checkpoint_path="checkpoints/spatial_best_roc.pth",
        ...     epoch=18,
        ...     metrics={"accuracy": 98.2, "f1": 98.0, "roc_auc": 99.3},
        ...     config_hash="abc123",
        ... )
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.registry_dir / _REGISTRY_FILENAME
        self._records: list[dict[str, Any]] = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def register(
        self,
        branch: str,
        dataset: str,
        checkpoint_path: str | Path,
        epoch: int,
        metrics: dict[str, float],
        config_hash: str = "",
        notes: str = "",
    ) -> None:
        """Add a checkpoint record to the registry.

        Args:
            branch: Model branch name (e.g. "spatial", "fusion").
            dataset: Dataset used for training/evaluation.
            checkpoint_path: Path to the ``.pth`` checkpoint file.
            epoch: Training epoch at which this checkpoint was saved.
            metrics: Performance metrics dictionary.
            config_hash: SHA-256 fingerprint of the config (12 chars).
            notes: Optional human-readable notes.
        """
        record: dict[str, Any] = {
            "branch": branch,
            "dataset": dataset,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "epoch": epoch,
            "metrics": metrics,
            "config_hash": config_hash,
            "notes": notes,
            "registered_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._records.append(record)
        self._save()
        logger.info(
            "Registry: registered %s/%s @ epoch %d  (ROC=%.4f)",
            branch,
            dataset,
            epoch,
            metrics.get("roc_auc", float("nan")),
        )

    def best(
        self,
        branch: str | None = None,
        metric: str = "roc_auc",
    ) -> dict[str, Any] | None:
        """Return the registry record with the highest value for a metric.

        Args:
            branch: Filter by branch name. If None, search all branches.
            metric: Metric key to maximise.

        Returns:
            The best record dict, or None if no records exist.
        """
        candidates = self._records
        if branch is not None:
            candidates = [r for r in candidates if r["branch"] == branch]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda r: r["metrics"].get(metric, float("-inf")),
        )

    def all_records(self, branch: str | None = None) -> list[dict[str, Any]]:
        """Return all registry records, optionally filtered by branch.

        Args:
            branch: Branch name filter. None returns all records.

        Returns:
            List of record dictionaries.
        """
        if branch is None:
            return list(self._records)
        return [r for r in self._records if r["branch"] == branch]

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> list[dict[str, Any]]:
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                data: Any = json.load(f)
                return data if isinstance(data, list) else []
        return []

    def _save(self) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)
