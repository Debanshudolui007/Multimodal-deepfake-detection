"""
Tests for Milestone 4 — Spatial Branch Model (SpatialBranch).

CPU-only — no GPU required. Uses small tensors to keep runtime fast.
"""

from __future__ import annotations

import torch
import pytest

from deepfake_detector.models.spatial_branch import SpatialBranch, build_spatial_branch


# ── Fixture: lightweight model (no pretrained weights needed for shape tests) ──


@pytest.fixture(scope="module")
def model_no_pretrain() -> SpatialBranch:
    """ResNeXt50 without ImageNet weights — fast, shape-only tests."""
    return SpatialBranch(pretrained=False, num_classes=1, dropout_rate=0.0)


@pytest.fixture(scope="module")
def feature_model() -> SpatialBranch:
    """Feature-only mode (no classifier head)."""
    return SpatialBranch(pretrained=False, feature_only=True, dropout_rate=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpatialBranchArchitecture:
    def test_forward_output_shape_binary(self, model_no_pretrain) -> None:
        x = torch.randn(2, 3, 224, 224)
        out = model_no_pretrain(x)
        assert out.shape == (2, 1)

    def test_forward_output_shape_feature_mode(self, feature_model) -> None:
        x = torch.randn(2, 3, 224, 224)
        out = feature_model(x)
        assert out.shape == (2, 2048)

    def test_feature_dim_is_2048(self, model_no_pretrain) -> None:
        assert model_no_pretrain.get_feature_dim() == 2048

    def test_multiclass_head(self) -> None:
        model = SpatialBranch(pretrained=False, num_classes=3, dropout_rate=0.0)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_is_finite(self, model_no_pretrain) -> None:
        x = torch.randn(2, 3, 224, 224)
        out = model_no_pretrain(x)
        assert torch.isfinite(out).all()

    def test_gradients_flow_to_head(self, model_no_pretrain) -> None:
        model_no_pretrain.eval()
        x = torch.randn(1, 3, 224, 224)
        loss = model_no_pretrain(x).sum()
        loss.backward()
        head_grad = model_no_pretrain.head[1].weight.grad
        assert head_grad is not None
        assert not torch.allclose(head_grad, torch.zeros_like(head_grad))


# ═══════════════════════════════════════════════════════════════════════════════
# Freeze / Unfreeze
# ═══════════════════════════════════════════════════════════════════════════════


class TestFreezeUnfreeze:
    def test_freeze_disables_backbone_gradients(self) -> None:
        model = SpatialBranch(pretrained=False)
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_freeze_keeps_head_trainable(self) -> None:
        model = SpatialBranch(pretrained=False)
        model.freeze_backbone()
        for param in model.head.parameters():
            assert param.requires_grad

    def test_unfreeze_restores_backbone_gradients(self) -> None:
        model = SpatialBranch(pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_parameter_counts(self) -> None:
        model = SpatialBranch(pretrained=False)
        counts = model.count_parameters()
        assert counts["trainable"] > 0
        assert counts["total"] >= counts["trainable"]

    def test_frozen_reduces_trainable_params(self) -> None:
        model = SpatialBranch(pretrained=False)
        full_trainable = model.count_parameters()["trainable"]
        model.freeze_backbone()
        frozen_trainable = model.count_parameters()["trainable"]
        assert frozen_trainable < full_trainable


# ═══════════════════════════════════════════════════════════════════════════════
# build_spatial_branch factory
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildSpatialBranch:
    def test_builds_from_config(self) -> None:
        cfg = {
            "model": {
                "spatial": {
                    "backbone": "resnext50_32x4d",
                    "pretrained": False,
                    "num_classes": 1,
                    "dropout": 0.5,
                }
            }
        }
        model = build_spatial_branch(cfg)
        assert isinstance(model, SpatialBranch)

    def test_feature_only_flag_propagated(self) -> None:
        cfg = {"model": {"spatial": {"pretrained": False}}}
        model = build_spatial_branch(cfg, feature_only=True)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape[1] == 2048

    def test_missing_config_uses_defaults(self) -> None:
        model = build_spatial_branch({"model": {"spatial": {"pretrained": False}}})
        assert model.num_classes == 1
