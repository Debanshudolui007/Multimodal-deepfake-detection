"""
Milestone 4 -- Spatial Branch Trainer.

Extends ``BaseTrainer`` with spatial-branch-specific logic:

  - **Loss**: ``BCEWithLogitsLoss`` (binary, num_classes=1) with
    ``pos_weight`` computed from the training label distribution.
  - **Metrics**: AUC-ROC, Accuracy, Precision, Recall, F1, EER.
  - **Weighted sampling**: passed-in DataLoader uses ``WeightedRandomSampler``
    to handle real/fake imbalance.
  - **Optimiser**: AdamW with weight decay from config.

Training protocol (from the architecture spec):
  1. Epoch 1-5: backbone frozen, only head trained.
  2. Epoch 6+: full model fine-tuned end-to-end (backbone unfrozen).

Usage:
    trainer = SpatialTrainer(model, optimizer, train_loader, val_loader,
                             cfg, experiment_dir)
    trainer.train()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from deepfake_detector.training.base_trainer import BaseTrainer
from deepfake_detector.training.losses import FocalLoss

logger = logging.getLogger(__name__)

# Epoch at which the backbone is unfrozen for end-to-end fine-tuning
_UNFREEZE_EPOCH = 5


class SpatialTrainer(BaseTrainer):
    """Spatial branch trainer with BCE loss, AUC metrics, and staged unfreezing.

    Args:
        model: ``SpatialBranch`` instance.
        optimizer: Configured optimiser (typically AdamW).
        train_loader: DataLoader for face crop training samples.
        val_loader: DataLoader for face crop validation samples.
        cfg: Merged config dict.
        experiment_dir: Root experiment directory.
        pos_weight: Optional BCEWithLogitsLoss positive class weight.
            If None, computed automatically from the training set.
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
            branch_name="spatial",
        )

        # We use FocalLoss instead of BCE to better handle the heavy class imbalance
        # and penalize confident misclassifications.
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logger.info("SpatialTrainer: Using FocalLoss(alpha=0.25, gamma=2.0)")

    # -- BaseTrainer abstract methods ------------------------------------------

    def _forward_pass(
        self, batch: tuple
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a spatial batch through the model.

        Args:
            batch: Tuple of ``(images, labels, video_ids, dataset_names)``
                as returned by ``SpatialFaceDataset.__getitem__``.

        Returns:
            ``(loss, logits, labels)`` tensors on the appropriate device.
        """
        images, labels, _video_ids, _datasets = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True).float()

        logits = self.model(images).squeeze(1)  # (B,) from (B, 1)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def _compute_epoch_metrics(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Compute AUC, accuracy, precision, recall, F1, EER, and optimal-threshold F1.

        The dataset is severely imbalanced (93% fake). At threshold=0.5 the model
        correctly ranks samples (AUC improves) but rarely predicts "real", so
        default F1 stays near zero. ``f1_opt`` uses the best threshold from the
        precision-recall curve so research quality is accurately tracked.

        Args:
            all_logits: ``(N,)`` float tensor of raw logits.
            all_labels: ``(N,)`` int tensor of ground-truth labels.

        Returns:
            Metrics dict with ``f1`` (thr=0.5) and ``f1_opt`` (best thr).
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_recall_curve,
            precision_score, recall_score, roc_auc_score,
        )

        probs  = torch.sigmoid(all_logits).numpy()
        preds  = (probs >= 0.5).astype(int)
        labels = all_labels.numpy().astype(int)

        if len(set(labels)) < 2:
            return {
                "auc": 0.0, "acc": float(accuracy_score(labels, preds)),
                "f1": 0.0,  "f1_opt": 0.0, "threshold": 0.5,
                "precision": 0.0, "recall": 0.0, "eer": 0.5,
            }

        auc       = float(roc_auc_score(labels, probs))
        acc       = float(accuracy_score(labels, preds))
        f1        = float(f1_score(labels, preds, zero_division=0))
        precision = float(precision_score(labels, preds, zero_division=0))
        recall    = float(recall_score(labels, preds, zero_division=0))
        eer       = _compute_eer(labels, probs)

        # Optimal threshold: maximise F1 over precision-recall curve
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(labels, probs)
        denom    = prec_arr + rec_arr
        f1_arr   = 2 * prec_arr * rec_arr / (denom + 1e-9)
        best_idx = int(f1_arr.argmax())
        f1_opt   = float(f1_arr[best_idx])
        best_thresh = float(thresh_arr[best_idx]) if best_idx < len(thresh_arr) else 0.5

        return {
            "auc":       auc,
            "acc":       acc,
            "f1":        f1,
            "f1_opt":    f1_opt,
            "threshold": best_thresh,
            "precision": precision,
            "recall":    recall,
            "eer":       eer,
        }

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Hook called after each epoch -- handles backbone unfreezing.

        Args:
            epoch: Current epoch (1-based).
            metrics: Combined train+val metrics for this epoch.
        """
        if epoch == _UNFREEZE_EPOCH:
            logger.info(
                "[spatial] Epoch %d: unfreezing backbone for end-to-end fine-tuning.",
                epoch,
            )
            if hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()
            
            # Drastically reduce learning rate to prevent destroying pretrained weights
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                
            logger.info("[spatial] Scaled learning rate by 0.1x for fine-tuning.")

    # -- Override train() to inject on_epoch_end ------------------------------

    def train(self) -> dict[str, Any]:  # type: ignore[override]
        """Run the full staged training loop with backbone unfreezing hook."""
        # Freeze backbone initially (head-only warmup)
        if hasattr(self.model, "freeze_backbone"):
            if self.start_epoch <= _UNFREEZE_EPOCH:
                self.model.freeze_backbone()
                logger.info("[spatial] Backbone frozen for head-warmup (epochs %d-%d).", self.start_epoch, _UNFREEZE_EPOCH - 1)
            else:
                self.model.unfreeze_backbone()
                logger.info("[spatial] Resuming after unfreeze epoch. Backbone is UNFROZEN.")

        self.memory_monitor.start()
        import time
        t0 = time.time()

        print(f"\n{'='*66}", flush=True)
        print(f"  SPATIAL BRANCH TRAINING  --  {len(self.train_loader.dataset):,} train samples", flush=True)
        print(f"  Device: {self.device}  |  Epochs: {self.max_epochs}  |  Batch: {self.train_loader.batch_size}", flush=True)
        print(f"{'='*66}\n", flush=True)

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.metric_logger.start_epoch()
            epoch_t0 = time.time()

            train_metrics = self._run_epoch(epoch, is_train=True)
            val_metrics   = self._run_epoch(epoch, is_train=False)

            val_auc = val_metrics.get("auc", 0.0)
            self.scheduler.step(val_auc)

            combined = {f"train_{k}": v for k, v in train_metrics.items()}
            combined.update({f"val_{k}": v for k, v in val_metrics.items()})
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metric_logger.log_epoch(epoch, combined, lr=current_lr)

            # On-epoch-end hook (unfreeze backbone at epoch 5)
            self.on_epoch_end(epoch, combined)

            is_best = val_auc > self._best_val_auc
            if is_best:
                self._best_val_auc = val_auc
                self._no_improve_count = 0
                self._save_checkpoint(epoch, combined, is_best=True)
            else:
                self._no_improve_count += 1
                self._save_checkpoint(epoch, combined, is_best=False)

            # -- Live epoch summary --------------------------------------------
            elapsed_epoch = time.time() - epoch_t0
            self._print_epoch_banner(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                elapsed_epoch=elapsed_epoch,
                current_lr=current_lr,
                is_best=is_best,
            )

            if self._no_improve_count >= self.early_stop_patience:
                print(f"\n  [!] Early stopping at epoch {epoch} (no improvement for {self.early_stop_patience} epochs)\n", flush=True)
                logger.info("[spatial] Early stopping at epoch %d.", epoch)
                break

        elapsed = time.time() - t0
        self.metric_logger.close()
        self.memory_monitor.stop()

        summary = {
            "branch": "spatial",
            "best_val_auc": round(self._best_val_auc, 6),
            "total_epochs": epoch,
            "elapsed_sec": round(elapsed, 1),
        }
        print(f"\n{'='*66}", flush=True)
        print(f"  SPATIAL TRAINING COMPLETE", flush=True)
        print(f"  Best Val AUC : {self._best_val_auc:.4f}", flush=True)
        print(f"  Total Epochs : {epoch}/{self.max_epochs}", flush=True)
        print(f"  Total Time   : {elapsed/60:.1f} min", flush=True)
        print(f"{'='*66}\n", flush=True)
        logger.info("Spatial training complete: %s", summary)
        return summary

    # -- Internal helpers ---------------------------------------------------------

    @staticmethod
    def _compute_pos_weight(loader: torch.utils.data.DataLoader) -> float:
        """Estimate BCEWithLogitsLoss pos_weight from DataLoader labels.

        Args:
            loader: DataLoader whose dataset has a ``get_labels()`` method.

        Returns:
            Ratio of negative to positive samples (n_neg / n_pos).
        """
        dataset = loader.dataset
        if hasattr(dataset, "get_labels"):
            labels = dataset.get_labels()
        else:
            return 1.0

        n_pos = sum(1 for l in labels if l == 1)
        n_neg = len(labels) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos


def _compute_eer(labels: Any, probs: Any) -> float:
    """Compute the Equal Error Rate (EER).

    EER is the point on the ROC curve where FAR = FRR.

    Args:
        labels: Binary ground-truth array.
        probs: Predicted probabilities.

    Returns:
        EER as a float in [0, 1].
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    # Find the crossing point
    eer_idx = int(abs(fnr - fpr).argmin())
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)


def build_spatial_trainer(
    model: nn.Module,
    cfg: dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    experiment_dir: Path,
) -> SpatialTrainer:
    """Convenience factory: build optimiser + trainer from config.

    Args:
        model: ``SpatialBranch`` instance.
        cfg: Merged configuration dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        experiment_dir: Experiment root directory.

    Returns:
        Configured ``SpatialTrainer`` ready to call ``.train()``.
    """
    train_cfg = cfg.get("train", {})
    optim_cfg = train_cfg.get("optimizer", {})

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(optim_cfg.get("lr", 1e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )

    return SpatialTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        experiment_dir=experiment_dir,
    )
