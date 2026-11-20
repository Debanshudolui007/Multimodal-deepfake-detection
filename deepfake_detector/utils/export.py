"""
Model export utilities: PyTorch checkpoint, TorchScript, and ONNX.

Provides a single ``export_model`` entry point that handles all three
formats and logs a summary of exported file sizes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_model(
    model: nn.Module,
    export_dir: str | Path,
    stem: str,
    formats: list[str],
    example_input: torch.Tensor,
    checkpoint_state: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Export a trained model to one or more serialisation formats.

    Args:
        model: Trained PyTorch model in eval mode.
        export_dir: Directory where exported files are saved.
        stem: Base filename stem (e.g. ``"spatial_best"``).
        formats: List of formats to export. Supported: ``"pth"``,
            ``"torchscript"``, ``"onnx"``.
        example_input: A sample input tensor used for tracing/ONNX export.
            Shape must match the model's expected input.
        checkpoint_state: Optional extra data to embed in the ``.pth``
            file (e.g. ``{"epoch": 18, "metrics": {...}}``).

    Returns:
        Dictionary mapping format name → saved file path.

    Raises:
        ValueError: If an unsupported format is requested.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved: dict[str, Path] = {}

    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == "pth":
            saved["pth"] = _export_pth(model, export_dir, stem, checkpoint_state)
        elif fmt in {"torchscript", "pt"}:
            saved["torchscript"] = _export_torchscript(
                model, export_dir, stem, example_input
            )
        elif fmt == "onnx":
            saved["onnx"] = _export_onnx(model, export_dir, stem, example_input)
        else:
            raise ValueError(
                f"Unsupported export format '{fmt}'. "
                "Choose from: pth, torchscript, onnx."
            )

    # Log file sizes
    for fmt, path in saved.items():
        size_mb = path.stat().st_size / 1e6
        logger.info("Exported [%s] → %s (%.2f MB)", fmt, path, size_mb)

    return saved


# ── Private helpers ───────────────────────────────────────────────────────────


def _export_pth(
    model: nn.Module,
    out_dir: Path,
    stem: str,
    extra: dict[str, Any] | None,
) -> Path:
    path = out_dir / f"{stem}.pth"
    payload: dict[str, Any] = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def _export_torchscript(
    model: nn.Module,
    out_dir: Path,
    stem: str,
    example: torch.Tensor,
) -> Path:
    path = out_dir / f"{stem}.pt"
    try:
        scripted = torch.jit.trace(model, example)
        scripted.save(str(path))
    except Exception as exc:
        logger.warning(
            "TorchScript trace failed (%s) — attempting torch.jit.script.", exc
        )
        scripted = torch.jit.script(model)
        scripted.save(str(path))
    return path


def _export_onnx(
    model: nn.Module,
    out_dir: Path,
    stem: str,
    example: torch.Tensor,
) -> Path:
    path = out_dir / f"{stem}.onnx"
    torch.onnx.export(
        model,
        example,
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return path
