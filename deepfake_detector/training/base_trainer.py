"""
Milestone 4 -- Generic AMP Training Base Class.

``BaseTrainer`` encapsulates all branch-agnostic training machinery:

  - **AMP** (Automatic Mixed Precision) via ``torch.amp.GradScaler`` --
    GPU-ready, no-op on CPU.
  - **Gradient clipping** (configurable max norm).
  - **LR scheduling**: ReduceLROnPlateau (default) or any
    ``torch.optim.lr_scheduler`` instance.
  - **Checkpoint save / resume** with structured JSON metadata.
  - **Early stopping** with configurable patience.
  - **TensorBoard + CSV logging** via ``MetricLogger``.
  - **Memory monitoring** via ``MemoryMonitor``.

Subclasses implement:
  - ``_forward_pass(batch)``  -> ``(loss, logits, labels)``
  - ``_compute_epoch_metrics(all_logits, all_labels)``  -> ``dict``
  - (optional) ``on_epoch_end(epoch, metrics)``

The training loop guarantees:
  1. Deterministic resumption from the exact epoch + optimiser state.
  2. Best checkpoint is always the one with the highest val AUC.
  3. All hyperparameters are passed through config -- no hardcoding.
"""

from __future__ import annotations

import abc
import logging
import time
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from deepfake_detector.utils.logger import MetricLogger
from deepfake_detector.utils.memory_monitor import MemoryMonitor
from deepfake_detector.utils.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class BaseTrainer(abc.ABC):
    """Abstract base trainer providing the full AMP training loop.

    Args:
        model: PyTorch model to train.
        optimizer: Configured optimiser instance.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        cfg: Merged configuration dict (``cfg["train"]`` subdict used).
        experiment_dir: Directory for logs, checkpoints, and manifests.
        branch_name: Human-readable name (e.g. ``"spatial"``).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict[str, Any],
        experiment_dir: Path,
        branch_name: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir
        self.branch_name = branch_name

        # -- Training hyper-parameters from config ------------------------------
        train_cfg = cfg.get("train", cfg)   # allow passing cfg["train"] directly
        self.max_epochs: int = train_cfg.get("epochs", 30)
        self.grad_clip: float = train_cfg.get("grad_clip", 1.0)
        self.early_stop_patience: int = train_cfg.get("early_stopping", {}).get("patience", 7)
        self.use_amp: bool = train_cfg.get("amp", True) and torch.cuda.is_available()

        # -- Device ------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        logger.info("BaseTrainer [%s] -- device=%s AMP=%s", branch_name, self.device, self.use_amp)

        # -- AMP scaler (torch.cuda.amp works on 2.2+; torch.amp.GradScaler requires 2.3+)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # -- LR scheduler -----------------------------------------------------
        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",               # Maximise val AUC
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 3),
        )

        # -- Logging & registry -------------------------------------------------
        self.metric_logger = MetricLogger(
            experiment_dir / "logs", run_name=branch_name, use_tensorboard=True
        )
        self.registry = ModelRegistry(experiment_dir / "checkpoints")
        self.memory_monitor = MemoryMonitor(interval_sec=120, tag=branch_name)

        # -- State -------------------------------------------------------------
        self.start_epoch: int = 1
        self._best_val_auc: float = 0.0
        self._no_improve_count: int = 0
        self.min_epochs_before_early_stop: int = 1
        self._ckpt_dir = experiment_dir / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # -- Auto-resume from checkpoint ---------------------------------------
        import re
        checkpoints = list(self._ckpt_dir.glob("epoch_*.pth"))
        if checkpoints:
            def extract_epoch(p: Path) -> int:
                match = re.search(r"epoch_(\d+)\.pth", p.name)
                return int(match.group(1)) if match else -1
            latest_ckpt = max(checkpoints, key=extract_epoch)
            if extract_epoch(latest_ckpt) > 0:
                self.load_checkpoint(latest_ckpt)

    # -- Abstract interface ----------------------------------------------------

    @abc.abstractmethod
    def _forward_pass(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single batch forward pass.

        Args:
            batch: Raw batch from the DataLoader.

        Returns:
            Tuple of ``(loss, logits, labels)``:
              - ``loss``: Scalar tensor (unreduced).
              - ``logits``: Predicted logits ``(B, C)``.
              - ``labels``: Ground-truth integer labels ``(B,)``.
        """

    @abc.abstractmethod
    def _compute_epoch_metrics(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Compute epoch-level metrics from accumulated logits and labels.

        Args:
            all_logits: Concatenated logits from all batches ``(N, C)``.
            all_labels: Concatenated labels ``(N,)``.

        Returns:
            Metrics dict (e.g. ``{"auc": 0.93, "acc": 0.88, ...}``).
        """

    # -- Main training loop ----------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Summary dict: best epoch, best val AUC, total training time.
        """
        self.memory_monitor.start()
        t0 = time.time()
        logger.info(
            "Training [%s]: epochs=%d AMP=%s grad_clip=%.1f",
            self.branch_name, self.max_epochs, self.use_amp, self.grad_clip,
        )

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.metric_logger.start_epoch()

            # -- Train ---------------------------------------------------------
            train_metrics = self._run_epoch(epoch, is_train=True)
            val_metrics = self._run_epoch(epoch, is_train=False)

            # -- LR scheduling -------------------------------------------------
            val_auc = val_metrics.get("auc", 0.0)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_auc)
            else:
                self.scheduler.step()

            # -- Log -----------------------------------------------------------
            combined = {f"train_{k}": v for k, v in train_metrics.items()}
            combined.update({f"val_{k}": v for k, v in val_metrics.items()})
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metric_logger.log_epoch(epoch, combined, lr=current_lr)

            logger.info(
                "[%s] Epoch %d/%d | train_loss=%.4f val_auc=%.4f lr=%.2e",
                self.branch_name, epoch, self.max_epochs,
                train_metrics.get("loss", 0.0), val_auc, current_lr,
            )

            # -- Checkpoint ----------------------------------------------------
            is_best = val_auc > self._best_val_auc
            if is_best:
                self._best_val_auc = val_auc
                self._no_improve_count = 0
                self._save_checkpoint(epoch, combined, is_best=True)
            else:
                self._no_improve_count += 1
                self._save_checkpoint(epoch, combined, is_best=False)

            # Staged trainers may alter trainable layers or learning rates at
            # an epoch boundary.  The base loop owns this hook so subclasses
            # do not need to duplicate the training loop.
            on_epoch_end = getattr(self, "on_epoch_end", None)
            if callable(on_epoch_end):
                on_epoch_end(epoch, combined)

            # -- Early stopping -------------------------------------------------
            if (
                epoch >= self.min_epochs_before_early_stop
                and self._no_improve_count >= self.early_stop_patience
            ):
                logger.info(
                    "[%s] Early stopping triggered at epoch %d (patience=%d)",
                    self.branch_name, epoch, self.early_stop_patience,
                )
                break

        elapsed = time.time() - t0
        self.metric_logger.close()
        self.memory_monitor.stop()

        summary = {
            "branch": self.branch_name,
            "best_val_auc": round(self._best_val_auc, 6),
            "total_epochs": epoch,
            "elapsed_sec": round(elapsed, 1),
        }
        logger.info("Training complete: %s", summary)
        return summary

    def _run_epoch(self, epoch: int, is_train: bool) -> dict[str, float]:
        """Run one epoch (train or eval) with live tqdm progress display.

        Args:
            epoch: Current epoch number (1-based).
            is_train: If True, run in training mode with grad updates.

        Returns:
            Metrics dict for this epoch.
        """
        self.model.train() if is_train else self.model.eval()
        loader = self.train_loader if is_train else self.val_loader
        phase  = "Train" if is_train else "Val  "
        n_total = len(loader)

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        n_batches = 0

        # -- File for live monitoring ------------------------------------------
        live_status_file = self.experiment_dir / "logs" / "live_batch.json"
        
        # -- tqdm progress bar -------------------------------------------------
        bar_fmt = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        pbar = (
            _tqdm(
                loader,
                total=n_total,
                desc=f"  Epoch {epoch:>3}/{self.max_epochs} [{phase}]",
                unit="batch",
                bar_format=bar_fmt,
                dynamic_ncols=True,
                leave=False,
            )
            if _TQDM_AVAILABLE
            else loader
        )

        ctx = torch.enable_grad if is_train else torch.no_grad
        with ctx():
            for batch in pbar:
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, logits, labels = self._forward_pass(batch)

                if is_train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
                n_batches += 1
                
                preds = (torch.sigmoid(logits.detach()) > 0.5).float()
                total_correct += (preds.view_as(labels) == labels).sum().item()
                total_samples += labels.numel()

                # Update tqdm postfix every batch with running loss
                if _TQDM_AVAILABLE and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        loss=f"{total_loss / n_batches:.4f}",
                        refresh=False,
                    )
                    
                # Write live batch progress for monitor_training.py
                if n_batches % 10 == 0 or n_batches == n_total:
                    try:
                        live_status_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(live_status_file, "w") as f:
                            json.dump({
                                "epoch": epoch,
                                "max_epoch": self.max_epochs,
                                "phase": "train" if is_train else "val",
                                "batch": n_batches,
                                "n_batches": n_total,
                                "loss": total_loss / n_batches,
                                "acc": total_correct / max(total_samples, 1)
                            }, f)
                    except Exception:
                        pass

        if _TQDM_AVAILABLE and hasattr(pbar, "close"):
            pbar.close()

        all_logits_t = torch.cat(all_logits, dim=0)
        all_labels_t = torch.cat(all_labels, dim=0)
        metrics = self._compute_epoch_metrics(all_logits_t, all_labels_t)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

    def _print_epoch_banner(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        elapsed_epoch: float,
        current_lr: float,
        is_best: bool,
    ) -> None:
        """Print a formatted epoch summary to stdout and logger.

        Args:
            epoch: Current epoch.
            train_metrics: Training metrics dict.
            val_metrics: Validation metrics dict.
            elapsed_epoch: Time taken for this epoch in seconds.
            current_lr: Current learning rate.
            is_best: Whether this is the new best checkpoint.
        """
        best_tag = " * NEW BEST" if is_best else ""
        mins, secs = divmod(int(elapsed_epoch), 60)
        banner = (
            f"\n{'='*66}\n"
            f"  [{self.branch_name.upper()}] Epoch {epoch:>3}/{self.max_epochs}  "
            f"({mins}m {secs:02d}s){best_tag}\n"
            f"{'-'*66}\n"
            f"  {'Metric':<18} {'Train':>10} {'Val':>10}\n"
            f"{'-'*66}\n"
            f"  {'Loss':<18} {train_metrics.get('loss', 0):>10.4f} {val_metrics.get('loss', 0):>10.4f}\n"
            f"  {'AUC-ROC':<18} {train_metrics.get('auc', 0):>10.4f} {val_metrics.get('auc', 0):>10.4f}\n"
            f"  {'Accuracy':<18} {train_metrics.get('acc', 0):>10.4f} {val_metrics.get('acc', 0):>10.4f}\n"
            f"  {'F1':<18} {train_metrics.get('f1', 0):>10.4f} {val_metrics.get('f1', 0):>10.4f}\n"
            f"  {'EER':<18} {train_metrics.get('eer', 0):>10.4f} {val_metrics.get('eer', 0):>10.4f}\n"
            f"  {'LR':<18} {current_lr:>10.2e}\n"
            f"  {'Best Val AUC':<18} {self._best_val_auc:>10.4f}\n"
            f"{'='*66}"
        )
        print(banner, flush=True)
        logger.info(
            "[%s] Epoch %d/%d | train_loss=%.4f train_auc=%.4f "
            "val_loss=%.4f val_auc=%.4f val_f1=%.4f eer=%.4f lr=%.2e%s",
            self.branch_name, epoch, self.max_epochs,
            train_metrics.get("loss", 0), train_metrics.get("auc", 0),
            val_metrics.get("loss", 0), val_metrics.get("auc", 0),
            val_metrics.get("f1", 0), val_metrics.get("eer", 0),
            current_lr, " [BEST]" if is_best else "",
        )

    # -- Checkpoint utilities --------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool,
    ) -> None:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Dict of combined train/val metrics.
            is_best: If True, also copy to ``best.pth``.
        """
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "branch": self.branch_name,
        }
        epoch_path = self._ckpt_dir / f"epoch_{epoch:04d}.pth"
        torch.save(ckpt, epoch_path)

        if is_best:
            best_path = self._ckpt_dir / "best.pth"
            torch.save(ckpt, best_path)
            logger.info(
                "[%s] New best checkpoint at epoch %d (val_auc=%.4f)",
                self.branch_name, epoch, self._best_val_auc,
            )

        self.registry.register(
            branch=self.branch_name,
            dataset="all",
            checkpoint_path=epoch_path,
            epoch=epoch,
            metrics=metrics,
        )

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load a checkpoint to resume training or run evaluation.

        Args:
            checkpoint_path: Path to a ``.pth`` checkpoint file.

        Returns:
            The epoch number stored in the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self._best_val_auc = ckpt.get("metrics", {}).get("val_auc", 0.0)

        logger.info(
            "[%s] Resumed from epoch %d (best_val_auc=%.4f)",
            self.branch_name, ckpt["epoch"], self._best_val_auc,
        )
        return ckpt["epoch"]
