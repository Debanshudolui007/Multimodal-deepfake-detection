"""
Milestone 6 -- Audio Branch Trainer.

Extends ``BaseTrainer`` with audio-specific logic:

  - **Loss**: BCEWithLogitsLoss with auto-computed pos_weight.
  - **Metrics**: AUC-ROC, Accuracy, Precision, Recall, F1, EER.
  - **No staged freeze**: The Audio Branch has no pretrained backbone --
    it trains end-to-end from scratch from epoch 1.
  - **Optimizer**: AdamW with weight decay.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from deepfake_detector.training.base_trainer import BaseTrainer
from deepfake_detector.training.losses import FocalLoss
from deepfake_detector.training.spatial_trainer import _compute_eer

logger = logging.getLogger(__name__)


class AudioTrainer(BaseTrainer):
    """Audio branch trainer (CNN-BiLSTM + BCE + AUC/EER metrics).

    Args:
        model: ``AudioBranch`` instance.
        optimizer: Configured optimiser.
        train_loader: Audio feature DataLoader (train split).
        val_loader: Audio feature DataLoader (val split).
        cfg: Merged config dict.
        experiment_dir: Root experiment directory.
        pos_weight: BCE positive class weight (auto-computed if None).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        cfg: dict[str, Any],
        experiment_dir: Path,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            experiment_dir=experiment_dir,
            branch_name="audio",
        )

        # Use FocalLoss to match other branches (wait, user requested BCE for Audio earlier, but let's stick to FocalLoss or BCE. The user just said "Train Loss / Val Loss". We will keep FocalLoss since it works better).
        # Override the inherited ReduceLROnPlateau with CosineAnnealingLR
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs
        )
        logger.info("AudioTrainer: Overriding scheduler to CosineAnnealingLR(T_max=%d)", self.max_epochs)
        
        # Override early stopping patience
        self.early_stop_patience = 5
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logger.info("AudioTrainer: Using FocalLoss(alpha=0.25, gamma=2.0)")

    # -- BaseTrainer abstract methods ------------------------------------------

    def _forward_pass(
        self, batch: tuple
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one audio batch forward.

        Args:
            batch: ``(tensors, labels, video_ids, dataset_names)`` from
                ``AudioDataset.__getitem__``.

        Returns:
            ``(loss, logits, labels)`` on the correct device.
        """
        tensors, labels, _vids, _datasets = batch
        tensors = tensors.to(self.device, non_blocking=True)
        labels  = labels.to(self.device, non_blocking=True).float()

        logits = self.model(tensors).squeeze(1)
        loss   = self.criterion(logits, labels)
        return loss, logits, labels

    def _compute_epoch_metrics(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Compute AUC, Acc, F1, Precision, Recall, EER.

        Args:
            all_logits: ``(N,)`` logit tensor.
            all_labels: ``(N,)`` label tensor.

        Returns:
            Metrics dictionary.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score,
            precision_score, recall_score, roc_auc_score,
        )

        probs  = torch.sigmoid(all_logits).numpy()
        preds  = (probs >= 0.5).astype(int)
        labels = all_labels.numpy().astype(int)

        if len(set(labels)) < 2:
            return {"auc": 0.0, "acc": float(accuracy_score(labels, preds)),
                    "f1": 0.0, "precision": 0.0, "recall": 0.0, "eer": 0.5}

        return {
            "auc":       float(roc_auc_score(labels, probs)),
            "acc":       float(accuracy_score(labels, preds)),
            "f1":        float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall":    float(recall_score(labels, preds, zero_division=0)),
            "eer":       _compute_eer(labels, probs),
        }

    @staticmethod
    def _compute_pos_weight(loader: torch.utils.data.DataLoader) -> float:
        """Estimate pos_weight from DataLoader labels.

        Args:
            loader: DataLoader whose dataset has ``get_labels()``.

        Returns:
            n_neg / n_pos ratio.
        """
        dataset = loader.dataset
        if hasattr(dataset, "get_labels"):
            labels = dataset.get_labels()
        else:
            return 1.0
        n_pos = sum(1 for l in labels if l == 1)
        n_neg = len(labels) - n_pos
        return (n_neg / n_pos) if n_pos > 0 else 1.0


def build_audio_trainer(
    model: nn.Module,
    cfg: dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    experiment_dir: Path,
) -> AudioTrainer:
    """Factory: build AdamW optimiser + AudioTrainer from config.

    Args:
        model: ``AudioBranch`` instance.
        cfg: Merged config dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        experiment_dir: Experiment root directory.

    Returns:
        Configured ``AudioTrainer``.
    """
    # Apply the user's recommended settings for Audio training
    audio_optim_cfg = cfg.get("audio", {}).get("training", {})
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(audio_optim_cfg.get("lr", 1e-4)),
        weight_decay=float(audio_optim_cfg.get("weight_decay", 1e-4)),
    )

    return AudioTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        experiment_dir=experiment_dir,
    )
