"""
System resource monitor for preprocessing and training runs.

Logs RAM, CPU, GPU VRAM (when available), and disk usage at configurable
intervals. Designed to run in a background thread so it doesn't block
the main training loop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# Lazy GPU monitoring — only imported if torch+CUDA is available
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def snapshot() -> dict[str, Any]:
    """Capture a single resource-usage snapshot.

    Returns:
        Dictionary with keys:
            - ``ram_used_gb``: RSS memory used by this process (GB).
            - ``ram_total_gb``: Total system RAM (GB).
            - ``ram_percent``: System-wide RAM usage (%).
            - ``cpu_percent``: CPU utilisation over the last second (%).
            - ``disk_free_gb``: Free disk space on the current partition (GB).
            - ``gpu_vram_used_mb``: GPU VRAM used (MB), or ``None``.
            - ``gpu_vram_total_mb``: Total GPU VRAM (MB), or ``None``.
    """
    proc = psutil.Process()
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(".")

    stats: dict[str, Any] = {
        "ram_used_gb": round(proc.memory_info().rss / 1e9, 3),
        "ram_total_gb": round(vm.total / 1e9, 3),
        "ram_percent": vm.percent,
        "cpu_percent": psutil.cpu_percent(interval=1.0),
        "disk_free_gb": round(disk.free / 1e9, 2),
        "gpu_vram_used_mb": None,
        "gpu_vram_total_mb": None,
    }

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        mem = torch.cuda.memory_stats()
        stats["gpu_vram_used_mb"] = round(
            mem.get("allocated_bytes.all.current", 0) / 1e6, 1
        )
        total = torch.cuda.get_device_properties(0).total_memory
        stats["gpu_vram_total_mb"] = round(total / 1e6, 1)

    return stats


def log_snapshot(tag: str = "") -> dict[str, Any]:
    """Capture and log a resource snapshot at INFO level.

    Args:
        tag: Optional label prepended to the log message.

    Returns:
        The snapshot dictionary.
    """
    s = snapshot()
    prefix = f"[{tag}] " if tag else ""
    logger.info(
        "%sRAM %.2fGB/%.2fGB (%.1f%%)  CPU %.1f%%  Disk free %.1fGB%s",
        prefix,
        s["ram_used_gb"],
        s["ram_total_gb"],
        s["ram_percent"],
        s["cpu_percent"],
        s["disk_free_gb"],
        f"  GPU {s['gpu_vram_used_mb']}MB/{s['gpu_vram_total_mb']}MB"
        if s["gpu_vram_used_mb"] is not None
        else "",
    )
    return s


class MemoryMonitor:
    """Background thread that logs resource snapshots at a fixed interval.

    Args:
        interval_sec: Seconds between snapshots. Default is 60.
        tag: Label for log messages.

    Example:
        >>> monitor = MemoryMonitor(interval_sec=30, tag="training")
        >>> monitor.start()
        >>> # ... training loop ...
        >>> monitor.stop()
    """

    def __init__(self, interval_sec: float = 60.0, tag: str = "") -> None:
        self.interval_sec = interval_sec
        self.tag = tag
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start the background monitoring thread."""
        logger.info(
            "MemoryMonitor started (interval=%.0fs, tag='%s')",
            self.interval_sec,
            self.tag,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the monitoring thread to stop and wait for it."""
        self._stop_event.set()
        self._thread.join(timeout=self.interval_sec + 5)
        logger.info("MemoryMonitor stopped.")

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self.interval_sec):
            try:
                log_snapshot(self.tag)
            except Exception as exc:
                logger.warning("MemoryMonitor error: %s", exc)
