"""
Milestone 7 — Fusion Network Model.

Architecture:
  - Loads all three pre-trained branches (Spatial, Frequency, Audio) in feature-only mode.
  - Projects each modality's features into a common dimension (512-D).
  - Uses Cross-Attention (via a TransformerEncoder layer) to allow modalities
    to attend to each other.
  - Passes the fused representation to an MLP classification head.
  - Computes auxiliary predictions for each modality individually, returning
    them in a dictionary for per-modality confidence output.
  - Supports modality dropout for robustness.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfake_detector.models.spatial_branch import build_spatial_branch
from deepfake_detector.models.frequency_branch import build_frequency_branch
from deepfake_detector.models.audio_branch import build_audio_branch

logger = logging.getLogger(__name__)


class AttentionFusion(nn.Module):
    """Cross-attention fusion network for multimodal deepfake detection."""

    def __init__(
        self,
        cfg: dict[str, Any],
        spatial_branch: nn.Module,
        frequency_branch: nn.Module,
        audio_branch: nn.Module,
    ) -> None:
        super().__init__()
        
        # Core models
        self.spatial_branch = spatial_branch
        self.frequency_branch = frequency_branch
        self.audio_branch = audio_branch
        
        # Ensure they are in feature extraction mode
        assert getattr(self.spatial_branch, "feature_only", False) is True
        assert getattr(self.frequency_branch, "feature_only", False) is True
        assert getattr(self.audio_branch, "feature_only", False) is True
        
        fusion_cfg = cfg.get("model", {}).get("fusion", {})
        proj_dim = fusion_cfg.get("projection_dim", 512)
        num_heads = fusion_cfg.get("num_heads", 8)
        attn_drop = fusion_cfg.get("attention_dropout", 0.1)
        hidden_dim = fusion_cfg.get("hidden_dim", 512)
        clf_drop = fusion_cfg.get("dropout", 0.4)
        self.num_classes = fusion_cfg.get("num_classes", 1)
        self.modality_dropout = fusion_cfg.get("modality_dropout_prob", 0.1)

        sp_dim = fusion_cfg.get("spatial_dim", 2048)
        fr_dim = fusion_cfg.get("frequency_dim", 768)
        au_dim = fusion_cfg.get("audio_dim", 512)

        # Modality Projections
        self.proj_spatial = nn.Sequential(
            nn.Linear(sp_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.proj_frequency = nn.Sequential(
            nn.Linear(fr_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.proj_audio = nn.Sequential(
            nn.Linear(au_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        # Cross-Attention
        # We treat the 3 modalities as a sequence of length 3.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=attn_drop, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Output Heads (Fused and individual for confidence/auxiliary loss)
        self.head_fused = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=clf_drop),
            nn.Linear(256, self.num_classes)
        )
        
        # Attention Pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(proj_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.head_spatial = nn.Linear(proj_dim, self.num_classes)
        self.head_frequency = nn.Linear(proj_dim, self.num_classes)
        self.head_audio = nn.Linear(proj_dim, self.num_classes)

        logger.info(f"AttentionFusion initialized with projection_dim={proj_dim}")

    def forward(
        self, 
        images: torch.Tensor, 
        freq_tensors: torch.Tensor, 
        audio_features: torch.Tensor,
        ablation_mode: list[str] | None = None,
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the fusion network.
        
        Args:
            images: Spatial RGB frames.
            freq_tensors: Frequency tensors.
            audio_features: Audio features (MFCC+Mel).
            ablation_mode: List of active modalities e.g. ["spatial", "audio"].
            If None, uses all 3.
            modality_mask: Per-sample availability mask of shape ``(B, 3)``.
                Unavailable modalities are excluded from attention.
                           
        Returns:
            Dict containing 'fused_logits', and optionally auxiliary logits.
        """
        B = images.size(0)
        device = images.device

        # Extract features
        sp_feat = self.spatial_branch(images)
        fr_feat = self.frequency_branch(freq_tensors)
        au_feat = self.audio_branch(audio_features)

        # Project features
        sp_proj = self.proj_spatial(sp_feat)    # (B, proj_dim)
        fr_proj = self.proj_frequency(fr_feat)
        au_proj = self.proj_audio(au_feat)

        # Handle Ablation / Modality Dropout
        active_mods = set(ablation_mode) if ablation_mode is not None else {"spatial", "frequency", "audio"}
        
        if self.training and ablation_mode is None:
            # Randomly drop modalities during training for robustness
            if torch.rand(1).item() < self.modality_dropout:
                drop_idx = torch.randint(0, 3, (1,)).item()
                if drop_idx == 0: active_mods.remove("spatial")
                elif drop_idx == 1: active_mods.remove("frequency")
                else: active_mods.remove("audio")
                
            # Must have at least one modality
            if not active_mods:
                active_mods = {"spatial", "frequency", "audio"}

        # Zero out inactive modalities
        if "spatial" not in active_mods:
            sp_proj = torch.zeros_like(sp_proj)
        if "frequency" not in active_mods:
            fr_proj = torch.zeros_like(fr_proj)
        if "audio" not in active_mods:
            au_proj = torch.zeros_like(au_proj)

        # Stack into sequence (B, 3, proj_dim)
        # Sequence: [Spatial, Frequency, Audio]
        seq = torch.stack([sp_proj, fr_proj, au_proj], dim=1)
        padding_mask = None
        if modality_mask is not None:
            if modality_mask.shape != (B, 3):
                raise ValueError("modality_mask must have shape (batch_size, 3)")
            available = modality_mask.to(device=device, dtype=torch.bool)
            if not available.any(dim=1).all():
                raise ValueError("Each sample must contain at least one modality")
            seq = seq * available.unsqueeze(-1)
            padding_mask = ~available
        
        # Self-attention over modalities
        fused_seq = self.transformer(seq, src_key_padding_mask=padding_mask)   # (B, 3, proj_dim)
        
        # Attention Pooling over the sequence
        attn_scores = self.attn_pool[0](fused_seq)
        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, 3, 1)
        fused_repr = (fused_seq * attn_weights).sum(dim=1)  # (B, proj_dim)
        
        # Predictions
        fused_logits = self.head_fused(fused_repr)
        sp_logits = self.head_spatial(sp_proj)
        fr_logits = self.head_frequency(fr_proj)
        au_logits = self.head_audio(au_proj)
        
        return {
            "fused": fused_logits,
            "spatial": sp_logits,
            "frequency": fr_logits,
            "audio": au_logits,
            "weights": attn_weights.squeeze(-1) # (B, 3)
        }

    def freeze_backbones(self) -> None:
        """Freeze all three backbones."""
        self.spatial_branch.freeze_backbone()
        self.frequency_branch.freeze_backbone()
        self.audio_branch.freeze_backbone()
        logger.info("All fusion backbones FROZEN.")

    def unfreeze_backbones(self) -> None:
        """Unfreeze all three backbones."""
        self.spatial_branch.unfreeze_backbone()
        self.frequency_branch.unfreeze_backbone()
        self.audio_branch.unfreeze_backbone()
        logger.info("All fusion backbones UNFROZEN.")


def build_fusion_branch(cfg: dict[str, Any]) -> AttentionFusion:
    """Builds the fusion network and its child branches."""
    logger.info("Building AttentionFusion branch...")
    
    # Load all branches in feature-only mode
    spatial_branch = build_spatial_branch(cfg, feature_only=True)
    frequency_branch = build_frequency_branch(cfg, feature_only=True)
    audio_branch = build_audio_branch(cfg, feature_only=True)
    
    # Load pre-trained weights from best.pth if they exist
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    exp_dir = project_root / "deepfake_detector" / "experiments"
    
    def load_ckpt(model: nn.Module, branch_name: str) -> None:
        ckpt_path = exp_dir / branch_name / "checkpoints" / "best.pth"
        if ckpt_path.exists():
            logger.info(f"Loading pre-trained weights for {branch_name} branch from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # Handle DataParallel/DistributedDataParallel prefix 'module.' if present
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                
            new_state = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "")
                new_state[new_k] = v
                
            model.load_state_dict(new_state, strict=False)
        else:
            logger.warning(f"No pre-trained weights found for {branch_name} branch at {ckpt_path}")

    load_ckpt(spatial_branch, "spatial")
    load_ckpt(frequency_branch, "frequency")
    load_ckpt(audio_branch, "audio")
    
    return AttentionFusion(cfg, spatial_branch, frequency_branch, audio_branch)
