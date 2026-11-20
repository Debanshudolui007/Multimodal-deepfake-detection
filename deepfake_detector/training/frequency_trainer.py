"""
Milestone 5 -- Frequency Branch Trainer.

Extends ``BaseTrainer`` with frequency-specific logic:

  - **Loss**: BCEWithLogitsLoss with auto-computed pos_weight.
  - **Metrics**: AUC-ROC, Accuracy, Precision, Recall, F1, EER.
  - **Staged fine-tuning**: backbone frozen for epochs 1-5,
    then unfrozen for end-to-end fine-tuning (identical protocol
    to the Spatial Branch).

The frequency branch intentionally trains with a lower learning rate
(5e-5) than the spatial branch because Swin-T's attention weights are
more sensitive to large gradient updates.
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

_UNFREEZE_EPOCH = 5


class FrequencyTrainer(BaseTrainer):
    """Frequency branch trainer (Swin-T + BCE + AUC/EER metrics).

    Args:
        model: ``FrequencyBranch`` instance.
        optimizer: Configured optimiser.
        train_loader: Frequency tensor DataLoader (train split).
        val_loader: Frequency tensor DataLoader (val split).
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
            branch_name="frequency",
        )

        # Use FocalLoss to match spatial branch and prevent massive overfitting on hard samples
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logger.info("FrequencyTrainer: Using FocalLoss(alpha=0.25, gamma=2.0)")

    # -- BaseTrainer abstract methods ------------------------------------------

    def _forward_pass(
        self, batch: tuple
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one frequency batch forward.

        Args:
            batch: ``(tensors, labels, video_ids, dataset_names)`` from
                ``FrequencyDataset.__getitem__``.

        Returns:
            ``(loss, logits, labels)`` on the correct device.
        """
        tensors, labels, _vids, _datasets = batch
        tensors = tensors.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True).float()

        logits = self.model(tensors).squeeze(1)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def _compute_epoch_metrics(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Compute AUC, Acc, F1, Precision, Recall, EER.

        Args:
            all_logits: Concatenated logits ``(N,)``.
            all_labels: Concatenated labels ``(N,)``.

        Returns:
            Metrics dictionary.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score,
            precision_score, recall_score, roc_auc_score,
        )

        probs = torch.sigmoid(all_logits).numpy()
        preds = (probs >= 0.5).astype(int)
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

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Unfreeze backbone at epoch 5.

        Args:
            epoch: Current epoch (1-based).
            metrics: Combined metrics dict.
        """
        if epoch == _UNFREEZE_EPOCH:
            if hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()
            
            # Scale down learning rate to protect Swin-T pretrained weights
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                
            logger.info("[frequency] Epoch %d: backbone unfrozen. Scaled learning rate by 0.1x.", epoch)

    # -- Override train() to inject on_epoch_end -------------------------------

    def train(self) -> dict[str, Any]:  # type: ignore[override]
        """Full staged training loop for the Frequency Branch."""
        if hasattr(self.model, "freeze_backbone"):
            if self.start_epoch <= _UNFREEZE_EPOCH:
                self.model.freeze_backbone()
                logger.info("[frequency] Backbone frozen for head-warmup (epochs %d-%d).", self.start_epoch, _UNFREEZE_EPOCH - 1)
            else:
                self.model.unfreeze_backbone()
                logger.info("[frequency] Resuming after unfreeze epoch. Backbone is UNFROZEN.")

        self.memory_monitor.start()
        t0 = time.time()

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.metric_logger.start_epoch()

            train_metrics = self._run_epoch(epoch, is_train=True)
            val_metrics   = self._run_epoch(epoch, is_train=False)

            val_auc = val_metrics.get("auc", 0.0)
            self.scheduler.step(val_auc)

            combined = {f"train_{k}": v for k, v in train_metrics.items()}
            combined.update({f"val_{k}": v for k, v in val_metrics.items()})
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metric_logger.log_epoch(epoch, combined, lr=current_lr)

            logger.info(
                "[frequency] Epoch %d/%d | train_loss=%.4f val_auc=%.4f lr=%.2e",
                epoch, self.max_epochs,
                train_metrics.get("loss", 0.0), val_auc, current_lr,
            )

            self.on_epoch_end(epoch, combined)

            is_best = val_auc > self._best_val_auc
            if is_best:
                self._best_val_auc = val_auc
                self._no_improve_count = 0
                self._save_checkpoint(epoch, combined, is_best=True)
            else:
                self._no_improve_count += 1
                self._save_checkpoint(epoch, combined, is_best=False)

            if self._no_improve_count >= self.early_stop_patience:
                logger.info("[frequency] Early stopping at epoch %d.", epoch)
                break

        elapsed = time.time() - t0
        self.metric_logger.close()
        self.memory_monitor.stop()

        summary = {
            "branch": "frequency",
            "best_val_auc": round(self._best_val_auc, 6),
            "total_epochs": epoch,
            "elapsed_sec": round(elapsed, 1),
        }
        logger.info("Frequency training complete: %s", summary)
        return summary

    @staticmethod
    def _compute_pos_weight(loader: torch.utils.data.DataLoader) -> float:
        """Estimate pos_weight from DataLoader labels.

        Args:
            loader: DataLoader whose dataset has a ``get_labels()`` method.

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


def build_frequency_trainer(
    model: nn.Module,
    cfg: dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    experiment_dir: Path,
) -> FrequencyTrainer:
    """Factory: build AdamW optimiser + FrequencyTrainer from config.

    Args:
        model: ``FrequencyBranch`` instance.
        cfg: Merged config dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        experiment_dir: Experiment root directory.

    Returns:
        Configured ``FrequencyTrainer``.
    """
    train_cfg = cfg.get("train", {})
    optim_cfg = train_cfg.get("optimizer", {})

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(optim_cfg.get("lr", 5e-5)),       # Lower LR for Swin-T
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )

    return FrequencyTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        experiment_dir=experiment_dir,
    )
