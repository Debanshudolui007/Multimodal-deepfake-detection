"""
Milestone 7 — Fusion Branch Trainer.

Extends the BaseTrainer for the AttentionFusion model.
Computes a combined loss:
    Loss = Loss_{fused} + 0.3 * (Loss_{spatial} + Loss_{frequency} + Loss_{audio})
Handles a two-stage training schedule:
    Stage 1: Backbones frozen, train only the fusion projection and attention heads.
    Stage 2: Backbones unfrozen, end-to-end fine-tuning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepfake_detector.training.base_trainer import BaseTrainer
from deepfake_detector.training.losses import FocalLoss

logger = logging.getLogger(__name__)


class FusionTrainer(BaseTrainer):
    """Trainer for the cross-attention Fusion Network."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict[str, Any],
        experiment_dir: Path,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            experiment_dir=experiment_dir,
            branch_name="fusion",
        )

        # Use standard BCE for the fusion branch to prevent mode collapse
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Read the staged schedule from fusion.yaml.
        train_cfg = cfg.get("fusion", {}).get("training", {})
        self.freeze_epochs = train_cfg.get("freeze_epochs", 10)
        # Do not stop the run while all backbones are still frozen.  Otherwise
        # a plateau in the fusion head prevents end-to-end fine-tuning.
        self.min_epochs_before_early_stop = self.freeze_epochs + 1
        
        # Auxiliary loss weight
        self.aux_weight = 0.3

    def _forward_pass(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack the multimodal batch and compute combined loss.

        Args:
            batch: (spatial_tensor, freq_tensor, audio_tensor, mask, label, video_id, dataset)

        Returns:
            Tuple of ``(loss, fused_logits, labels)``.
        """
        spatial_tensor = batch[0].to(self.device, non_blocking=True)
        freq_tensor = batch[1].to(self.device, non_blocking=True)
        audio_tensor = batch[2].to(self.device, non_blocking=True)
        mask = batch[3].to(self.device, non_blocking=True) # (B, 3)
        labels = batch[4].to(self.device, non_blocking=True).float().unsqueeze(1)

        # Handle ablation mode if specified in config (we default to None for standard training)
        ablation_mode = None 
        # (Ablation during evaluation can be passed differently, but training is all modalities)

        outputs = self.model(
            spatial_tensor, freq_tensor, audio_tensor,
            ablation_mode=ablation_mode, modality_mask=mask,
        )

        # Remove label smoothing to avoid focal loss interaction bugs
        smoothed_labels = labels

        # Compute losses using labels (fused loss is always computed)
        loss_fused = self.criterion(outputs["fused"], smoothed_labels)
        
        # Apply boolean mask to auxiliary losses so we don't penalize missing modalities
        valid_sp = mask[:, 0] > 0
        loss_sp = self.criterion(outputs["spatial"][valid_sp], smoothed_labels[valid_sp]) if valid_sp.any() else torch.tensor(0.0, device=self.device)
        
        valid_fr = mask[:, 1] > 0
        loss_fr = self.criterion(outputs["frequency"][valid_fr], smoothed_labels[valid_fr]) if valid_fr.any() else torch.tensor(0.0, device=self.device)
        
        valid_au = mask[:, 2] > 0
        loss_au = self.criterion(outputs["audio"][valid_au], smoothed_labels[valid_au]) if valid_au.any() else torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = loss_fused + self.aux_weight * (loss_sp + loss_fr + loss_au)

        return total_loss, outputs["fused"], labels.squeeze(1)

    def _compute_epoch_metrics(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Compute discrimination metrics that expose class collapse."""
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            roc_auc_score,
            roc_curve,
        )

        probs = torch.sigmoid(all_logits).numpy()
        preds = (probs >= 0.5).astype(int)
        labels = all_labels.numpy().astype(int)

        if len(set(labels)) < 2:
            return {
                "auc": 0.0, "acc": float(accuracy_score(labels, preds)),
                "balanced_acc": 0.0, "f1": 0.0, "f1_macro": 0.0,
                "threshold": 0.5, "eer": 0.5,
            }

        auc = float(roc_auc_score(labels, probs))
        acc = float(accuracy_score(labels, preds))
        f1 = float(f1_score(labels, preds, zero_division=0))
        balanced_acc = float(balanced_accuracy_score(labels, preds))
        f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))

        # Select an operating threshold by Youden's J, not fake-class F1.  It
        # is recorded for evaluation only; it is never probability calibration.
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        fpr, tpr, thresholds = roc_curve(labels, probs)
        best_thresh = float(thresholds[int((tpr - fpr).argmax())])
        if not 0.0 <= best_thresh <= 1.0:
            best_thresh = 0.5
        eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))

        return {
            "auc": auc,
            "acc": acc,
            "balanced_acc": balanced_acc,
            "f1": f1,
            "f1_macro": f1_macro,
            "threshold": best_thresh,
            "eer": eer,
        }

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Handle two-stage unfreezing schedule."""
        if epoch == self.freeze_epochs:
            logger.info("Epoch %d: Unfreezing backbones for end-to-end fine-tuning.", epoch)
            self.model.unfreeze_backbones()
            # The base trainer manages the LR scheduler (ReduceLROnPlateau), which will
            # naturally adjust the LR. We could re-init the optimizer here with `finetune_lr`,
            # but for simplicity we rely on the scheduler or a single LR.
            # If `finetune_lr` must be forced:
            train_cfg = self.cfg.get("fusion", {}).get("training", {})
            finetune_lr = float(train_cfg.get("finetune_lr", 1.0e-5))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = finetune_lr
            logger.info("Optimizer LR forced to %s for fine-tuning stage.", finetune_lr)
