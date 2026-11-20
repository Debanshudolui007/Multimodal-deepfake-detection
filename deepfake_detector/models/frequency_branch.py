"""
Milestone 5 — Frequency Branch Model.

Architecture:
  - **Backbone**: Swin Transformer Tiny (``swin_tiny_patch4_window7_224``)
    loaded from ``timm``, pretrained on ImageNet-1k.
  - **Input**: 3-channel frequency tensor ``(B, 3, 224, 224)`` where the
    three channels are:
      - CH0: FFT log-magnitude spectrum (global frequency distribution)
      - CH1: DWT LL subband (low-frequency approximation)
      - CH2: DWT energy map (high-frequency detail energy)
  - **Feature extractor**: Swin-T global average pool output → (B, 768).
  - **Classification head** (optional): Dropout → Linear(768, num_classes).

Swin Transformer is chosen over CNNs for this branch because its
window-based self-attention captures long-range frequency correlations
(e.g. GAN-specific grid artefacts spanning the full image) that local
convolutions miss.

Two modes:
  1. ``feature_only=False`` (default) — backbone + head → logits (B, C).
  2. ``feature_only=True``           — backbone only → features (B, 768).

Supports ``freeze_backbone()`` / ``unfreeze_backbone()`` for staged
fine-tuning identical to the Spatial Branch.
"""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_DEFAULT_BACKBONE = "swin_tiny_patch4_window7_224"
_FEATURE_DIM = 768


class FrequencyBranch(nn.Module):
    """Swin-T frequency artefact detection branch.

    Args:
        backbone_name: timm model identifier.
        pretrained: Load ImageNet-1k weights. Default: True.
        num_classes: Output classes. Default: 1 (binary BCE).
        dropout_rate: Dropout before the classification head.
        feature_only: If True, return feature vectors, not logits.
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

        self.backbone: nn.Module = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,        # Strip timm head
            global_pool="avg",
        )
        feature_dim: int = self.backbone.num_features

        logger.info(
            "FrequencyBranch: backbone=%s pretrained=%s feature_dim=%d",
            backbone_name, pretrained, feature_dim,
        )

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
            x: Frequency tensor ``(B, 3, H, W)`` — FFT + DWT LL + DWT energy.

        Returns:
            Logits ``(B, num_classes)`` or features ``(B, feature_dim)``.
        """
        features = self.backbone(x)
        return self.head(features)

    def get_feature_dim(self) -> int:
        """Return backbone output dimension (768 for Swin-T).

        Returns:
            Feature vector length.
        """
        return int(self.backbone.num_features)

    def freeze_backbone(self) -> None:
        """Freeze backbone for head-warmup training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("FrequencyBranch backbone FROZEN.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("FrequencyBranch backbone UNFROZEN.")

    def count_parameters(self) -> dict[str, int]:
        """Return trainable and total parameter counts.

        Returns:
            Dict with ``trainable`` and ``total`` keys.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable, "total": total}


def build_frequency_branch(
    cfg: dict[str, Any], feature_only: bool = False
) -> FrequencyBranch:
    """Instantiate a ``FrequencyBranch`` from config.

    Args:
        cfg: Merged config dict (reads ``cfg["model"]["frequency"]``).
        feature_only: If True, return a feature extractor (no head).

    Returns:
        Configured ``FrequencyBranch`` instance.
    """
    freq_cfg = cfg.get("model", {}).get("frequency", {})
    return FrequencyBranch(
        backbone_name=freq_cfg.get("backbone", _DEFAULT_BACKBONE),
        pretrained=freq_cfg.get("pretrained", True),
        num_classes=freq_cfg.get("num_classes", 1),
        dropout_rate=freq_cfg.get("dropout", 0.5),
        feature_only=feature_only,
    )
