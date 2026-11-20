"""
Multi-format training logger.

Simultaneously writes metrics to:
  - Python logging (console + file)
  - TensorBoard SummaryWriter
  - CSV file (one row per epoch)
  - JSON summary file (updated after every epoch)
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# TensorBoard is optional — imported lazily so CPU-only runs still work.
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore

    _TB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TB_AVAILABLE = False


def setup_logging(
    log_dir: str | Path,
    run_name: str = "run",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure root logger to write to console and a rotating file.

    Args:
        log_dir: Directory where ``training.log`` will be saved.
        run_name: Prefix used in the log file name.
        level: Logging level (e.g. ``logging.INFO``).

    Returns:
        Configured root logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    log_file = log_dir / f"{run_name}.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger.info("Logging to: %s", log_file)
    return root


class MetricLogger:
    """Records per-epoch metrics to TensorBoard, CSV, and JSON.

    Args:
        log_dir: Root directory for all log outputs.
        run_name: Name of the current training run / experiment.
        use_tensorboard: Whether to write TensorBoard events.

    Example:
        >>> ml = MetricLogger("logs/", "spatial_run")
        >>> ml.log_epoch(1, {"train_loss": 0.42, "val_acc": 0.91})
        >>> ml.close()
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str = "run",
        use_tensorboard: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self._writer: Any = None
        if use_tensorboard and _TB_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard" / run_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info("TensorBoard writer: %s", tb_dir)
        elif use_tensorboard and not _TB_AVAILABLE:
            logger.warning(
                "TensorBoard not installed — skipping TB logging."
            )

        # CSV
        self._csv_path = self.log_dir / f"{run_name}_metrics.csv"
        self._csv_file = self._csv_path.open("a", newline="", encoding="utf-8")
        self._csv_writer: csv.DictWriter | None = None

        # JSON summary
        self._json_path = self.log_dir / f"{run_name}_summary.json"
        self._history: list[dict[str, Any]] = []

        # Timing
        self._epoch_start: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start_epoch(self) -> None:
        """Record the wall-clock start time of an epoch."""
        self._epoch_start = time.time()

    def log_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
        lr: float | None = None,
    ) -> None:
        """Log all metrics for a completed epoch.

        Args:
            epoch: Epoch index (1-indexed).
            metrics: Dictionary of metric names → values.
            lr: Current learning rate (optional).
        """
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        row: dict[str, Any] = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 2),
            **metrics,
        }
        if lr is not None:
            row["lr"] = lr

        # ── TensorBoard ──────────────────────────────────────────────────────
        if self._writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, global_step=epoch)
            if lr is not None:
                self._writer.add_scalar("learning_rate", lr, global_step=epoch)

        # ── CSV ──────────────────────────────────────────────────────────────
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # ── JSON ─────────────────────────────────────────────────────────────
        self._history.append(row)
        self._flush_json()

        # ── Console ──────────────────────────────────────────────────────────
        metric_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        logger.info("[Epoch %03d] %s  (%.1fs)", epoch, metric_str, elapsed)

    def close(self) -> None:
        """Flush and close all open file handles."""
        if self._writer is not None:
            self._writer.close()
        self._csv_file.close()
        logger.info("MetricLogger closed. CSV: %s  JSON: %s", self._csv_path, self._json_path)

    # ── Private ───────────────────────────────────────────────────────────────

    def _flush_json(self) -> None:
        """Write the full history list to the JSON summary file."""
        with self._json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"run_name": self.run_name, "history": self._history},
                f,
                indent=2,
            )

def write_live_status(phase: str, current: int, total: int, extra: dict[str, Any] | None = None) -> None:
    """Write live progress to live_batch.json for monitor_training.py to read.
    
    Args:
        phase: Name of the current phase (e.g. 'extract_frames', 'crop_faces').
        current: Current progress count.
        total: Total expected count.
        extra: Any extra dictionary fields to append.
    """
    try:
        live_json = Path("deepfake_detector/logs/live_batch.json")
        live_json.parent.mkdir(parents=True, exist_ok=True)
        
        # Preserve video progress fields if they exist
        video_current = None
        video_total = None
        if live_json.exists():
            try:
                with open(live_json, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    video_current = old_data.get("video_current")
                    video_total = old_data.get("video_total")
            except Exception:
                pass
        
        data = {
            "epoch": 1,
            "max_epoch": 1,
            "phase": phase,
            "batch": current,
            "n_batches": total,
            "loss": 0.0
        }
        
        if video_current is not None:
            data["video_current"] = video_current
        if video_total is not None:
            data["video_total"] = video_total
            
        if extra:
            data.update(extra)
            
        with open(live_json, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
