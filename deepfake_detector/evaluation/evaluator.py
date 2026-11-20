"""
Milestone 8 — Evaluator.

Handles computing 15 evaluation metrics:
AUC, Accuracy, F1, Precision, Recall, Specificity, EER, Brier Score, LogLoss, 
MCC, AP (Average Precision), FNR, FPR, Cohen's Kappa, Balanced Accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)

class Evaluator:
    """Computes a comprehensive suite of metrics for binary classification."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
        """Compute all 15 metrics.
        
        Args:
            y_true: 1D array of binary ground truth (0 or 1).
            y_prob: 1D array of predicted probabilities [0, 1].
            
        Returns:
            Dictionary of computed metrics.
        """
        y_pred = (y_prob >= self.threshold).astype(int)
        
        metrics = {}
        
        # Core Metrics
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["ap"] = float(average_precision_score(y_true, y_prob))
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        
        # Matrix based
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        # Statistical / Calibrated
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred))
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
        
        # EER
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr_curve, tpr_curve)(x), 0.0, 1.0)
        metrics["eer"] = float(eer)
        
        return metrics
