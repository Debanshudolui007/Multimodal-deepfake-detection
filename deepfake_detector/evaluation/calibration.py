"""Probability calibration utilities for binary deepfake classifiers.

Calibration is fitted only after model selection, using logits and labels from
a held-out calibration split. It must not be fitted from a decision threshold.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt_scaler(
    logits: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Fit ``sigmoid(a * logit + b)`` and return checkpoint-safe metadata."""
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        raise ValueError("logits and labels must be non-empty and equally sized")
    if not np.isfinite(x).all():
        raise ValueError("logits must be finite")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Platt scaling requires both real (0) and fake (1) labels")

    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(x.reshape(-1, 1), y)
    return {
        "method": "platt",
        "a": float(model.coef_[0, 0]),
        "b": float(model.intercept_[0]),
        "n_samples": int(x.size),
    }


def apply_platt_scaler(
    logits: Sequence[float] | np.ndarray, calibration: dict[str, Any]
) -> np.ndarray:
    """Return calibrated fake probabilities from serialized Platt metadata."""
    if calibration.get("method") != "platt":
        raise ValueError("calibration metadata must use method='platt'")
    x = np.asarray(logits, dtype=np.float64)
    z = float(calibration["a"]) * x + float(calibration["b"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
