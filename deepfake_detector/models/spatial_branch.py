"""
Milestone 4 — Spatial Branch Model.

Architecture:
  - **Backbone**: ResNeXt50_32×4d (``timm``, pretrained on ImageNet-1k).
  - **Feature extractor**: removes the classification head; outputs a
    ``(B, 2048)`` feature vector via global average pooling.
  - **Classification head** (optional): Dropout → Linear(2048, num_classes).

Two operational modes:
  1. ``feature_only=True`` — backbone only; returns ``(B, 2048)`` features.
     Used by the Fusion module (Milestone 7).
  2. ``feature_only=False`` (default) — backbone + head; returns logits
     ``(B, num_classes)``. Used for standalone training (Milestone 4).

Design principles:
  - Backbone weights are always loaded from the ``timm`` registry; no
    hardcoded URLs.
  - Supports ``freeze_backbone()`` / ``unfreeze_backbone()`` for staged
    fine-tuning as required by the training protocol.
  - ``get_feature_dim()`` lets downstream modules query the output dim.
"""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Default model identifier ──────────────────────────────────────────────────
_DEFAULT_BACKBONE = "resnext50_32x4d"
_FEATURE_DIM = 2048


class SpatialBranch(nn.Module):
    """ResNeXt50_32x4d-based spatial (RGB) deepfake detection branch.

    Args:
        backbone_name: timm model identifier. Default: ``"resnext50_32x4d"``.
        pretrained: Load ImageNet-1k pretrained weights. Default: ``True``.
        num_classes: Number of output classes. Default: 1 (binary BCE).
        dropout_rate: Dropout probability before the classifier head.
        feature_only: If True, skip the classifier head and return feature
            vectors. Default: ``False``.
    """

    def __init__(
        self,
        backbone_name: str = _DEFAULT_BACKBONE,
        pretrained: bool = True,
        num_classes: int = 1,
        dropout_rate: float = 0.5,
        feature_only: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_only = feature_only
        self.num_classes = num_classes

        # ── Load backbone from timm ────────────────────────────────────────────
        self.backbone: nn.Module = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # Remove timm's built-in head
            global_pool="avg",      # Global average pooling → (B, 2048)
        )
        feature_dim = self.backbone.num_features

        logger.info(
            "SpatialBranch: backbone=%s pretrained=%s feature_dim=%d",
            backbone_name, pretrained, feature_dim,
        )

        # ── Classification head ────────────────────────────────────────────────
        if not feature_only:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            - If ``feature_only=False``: logits ``(B, num_classes)``.
            - If ``feature_only=True``: feature vector ``(B, feature_dim)``.
        """
        features = self.backbone(x)         # (B, 2048)
        return self.head(features)          # (B, num_classes) or (B, 2048)

    def get_feature_dim(self) -> int:
        """Return the backbone output feature dimension.

        Returns:
            Feature vector length (typically 2048 for ResNeXt50).
        """
        return int(self.backbone.num_features)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters.

        Used in staged fine-tuning: train head-only first, then unfreeze.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("SpatialBranch backbone FROZEN.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("SpatialBranch backbone UNFROZEN.")

    def count_parameters(self) -> dict[str, int]:
        """Return trainable and total parameter counts.

        Returns:
            Dict with ``trainable`` and ``total`` keys.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable, "total": total}


def build_spatial_branch(cfg: dict[str, Any], feature_only: bool = False) -> SpatialBranch:
    """Instantiate a ``SpatialBranch`` from the model YAML config.

    Args:
        cfg: Merged configuration dict (must contain ``cfg["model"]["spatial"]``).
        feature_only: If True, return a feature extractor (no head).

    Returns:
        Configured and (optionally) pretrained ``SpatialBranch`` instance.
    """
    spatial_cfg = cfg.get("model", {}).get("spatial", {})
    return SpatialBranch(
        backbone_name=spatial_cfg.get("backbone", _DEFAULT_BACKBONE),
        pretrained=spatial_cfg.get("pretrained", True),
        num_classes=spatial_cfg.get("num_classes", 1),
        dropout_rate=spatial_cfg.get("dropout", 0.5),
        feature_only=feature_only,
    )
