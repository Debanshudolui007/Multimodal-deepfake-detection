"""
Tests for Milestone 4 — Spatial Trainer.

Tests the concrete trainer components that don't need a full training run:
  - _forward_pass output shapes and types
  - _compute_epoch_metrics correctness
  - _compute_pos_weight calculation
  - EER computation
  - Checkpoint save / load round-trip
  - on_epoch_end backbone-unfreezing logic

All tests use tiny synthetic DataLoaders — no real data needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepfake_detector.models.spatial_branch import SpatialBranch
from deepfake_detector.training.spatial_trainer import (
    SpatialTrainer,
    _compute_eer,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _tiny_loader(n: int = 16, n_real: int = 8) -> DataLoader:
    """Build a minimal DataLoader returning (images, labels, vids, datasets)."""
    images = torch.randn(n, 3, 32, 32)   # Tiny images — model accepts any size
    labels = torch.zeros(n, dtype=torch.long)
    labels[:n - n_real] = 1              # First (n - n_real) are fake

    class _DS(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [(images[i], int(labels[i]), f"vid_{i}", "ds") for i in range(n)]
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]
        def get_labels(self): return [int(labels[i]) for i in range(n)]

    return DataLoader(_DS(), batch_size=4)


def _make_trainer(tmp_path: Path, n: int = 16) -> SpatialTrainer:
    """Build a fully configured SpatialTrainer for testing."""
    model = SpatialBranch(pretrained=False, num_classes=1, dropout_rate=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = _tiny_loader(n)
    cfg = {
        "train": {
            "epochs": 2,
            "grad_clip": 1.0,
            "amp": False,           # CPU — disable AMP
            "early_stopping": {"patience": 5},
            "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
            "scheduler": {"factor": 0.5, "patience": 3},
        }
    }
    return SpatialTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        val_loader=loader,
        cfg=cfg,
        experiment_dir=tmp_path,
        pos_weight=1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_eer
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeEer:
    def test_perfect_classifier_eer_near_zero(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.01, 0.01, 0.01, 0.99, 0.99, 0.99])
        eer = _compute_eer(labels, probs)
        assert eer < 0.05

    def test_random_classifier_eer_near_half(self) -> None:
        rng = np.random.RandomState(42)
        labels = (rng.rand(100) > 0.5).astype(int)
        probs = rng.rand(100)
        eer = _compute_eer(labels, probs)
        assert 0.3 < eer < 0.7   # Random — near 0.5

    def test_eer_in_zero_one_range(self) -> None:
        labels = np.array([0, 1, 0, 1, 1, 0])
        probs = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4])
        eer = _compute_eer(labels, probs)
        assert 0.0 <= eer <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_epoch_metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeEpochMetrics:
    @pytest.fixture()
    def trainer(self, tmp_path):
        return _make_trainer(tmp_path)

    def test_returns_required_keys(self, trainer) -> None:
        logits = torch.tensor([2.0, -2.0, 1.0, -1.0])
        labels = torch.tensor([1, 0, 1, 0])
        metrics = trainer._compute_epoch_metrics(logits, labels)
        for key in ("auc", "acc", "f1", "precision", "recall", "eer"):
            assert key in metrics

    def test_perfect_predictions(self, trainer) -> None:
        logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
        labels = torch.tensor([1, 1, 0, 0])
        metrics = trainer._compute_epoch_metrics(logits, labels)
        assert metrics["auc"] == pytest.approx(1.0, abs=0.01)
        assert metrics["acc"] == pytest.approx(1.0, abs=0.01)

    def test_single_class_labels_does_not_crash(self, trainer) -> None:
        logits = torch.tensor([1.0, 1.0, 1.0])
        labels = torch.tensor([1, 1, 1])
        metrics = trainer._compute_epoch_metrics(logits, labels)
        assert "auc" in metrics    # Should return 0.0, not raise


# ═══════════════════════════════════════════════════════════════════════════════
# _forward_pass
# ═══════════════════════════════════════════════════════════════════════════════


class TestForwardPass:
    def test_returns_three_tensors(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        batch = (
            torch.randn(4, 3, 32, 32),
            torch.tensor([0, 1, 1, 0]),
            ["v1", "v2", "v3", "v4"],
            ["ds", "ds", "ds", "ds"],
        )
        loss, logits, labels = trainer._forward_pass(batch)
        assert isinstance(loss, torch.Tensor)
        assert logits.shape == (4,)  # Squeezed from (4,1)
        assert labels.shape == (4,)

    def test_loss_is_scalar(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        batch = (
            torch.randn(4, 3, 32, 32),
            torch.tensor([0, 1, 0, 1]),
            ["v1", "v2", "v3", "v4"],
            ["ds", "ds", "ds", "ds"],
        )
        loss, _, _ = trainer._forward_pass(batch)
        assert loss.ndim == 0

    def test_loss_is_positive(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        batch = (
            torch.randn(8, 3, 32, 32),
            torch.tensor([0, 1, 0, 1, 1, 0, 0, 1]),
            [f"v{i}" for i in range(8)],
            ["ds"] * 8,
        )
        loss, _, _ = trainer._forward_pass(batch)
        assert loss.item() > 0


# ═══════════════════════════════════════════════════════════════════════════════
# on_epoch_end — backbone unfreezing
# ═══════════════════════════════════════════════════════════════════════════════


class TestOnEpochEnd:
    def test_backbone_unfrozen_at_epoch_5(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.model.freeze_backbone()
        # Verify frozen
        assert not next(trainer.model.backbone.parameters()).requires_grad
        # Trigger unfreeze
        trainer.on_epoch_end(5, {})
        # Now should be unfrozen
        assert next(trainer.model.backbone.parameters()).requires_grad

    def test_backbone_still_frozen_before_epoch_5(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.model.freeze_backbone()
        trainer.on_epoch_end(4, {})
        assert not next(trainer.model.backbone.parameters()).requires_grad


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint save / load round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointRoundTrip:
    def test_save_and_load_preserves_epoch(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        metrics = {"val_auc": 0.91, "train_loss": 0.3}
        trainer._save_checkpoint(7, metrics, is_best=True)

        best_path = tmp_path / "checkpoints" / "best.pth"
        assert best_path.exists()

        # Create a fresh trainer and load the checkpoint
        trainer2 = _make_trainer(tmp_path)
        epoch = trainer2.load_checkpoint(best_path)
        assert epoch == 7

    def test_load_checkpoint_missing_file_raises(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(tmp_path / "nonexistent.pth")

    def test_best_checkpoint_overwrites_on_improvement(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer._save_checkpoint(1, {"val_auc": 0.80}, is_best=False)
        trainer._save_checkpoint(2, {"val_auc": 0.90}, is_best=True)
        trainer._save_checkpoint(3, {"val_auc": 0.95}, is_best=True)

        best_path = tmp_path / "checkpoints" / "best.pth"
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        assert ckpt["epoch"] == 3
