"""
Milestone 6 — Audio Branch Model.

Architecture: **1D CNN → BiLSTM → Attention Pool → Classifier**

Input: concatenated MFCC and Mel-spectrogram features.
  - MFCC:  ``(batch, 120, 300)`` — 120 coefficients × 300 time-frames
  - Mel:   ``(batch, 128, 300)`` — 128 mel-bins  × 300 time-frames
  - Concatenated input: ``(batch, 248, 300)``

Pipeline:
  1. **3-layer 1D CNN** with increasing channel depth [64, 128, 256],
     batch normalisation, ReLU, and max pooling → temporal compression.
  2. **BiLSTM** (2 layers, 256 hidden, bidirectional) → sequential modelling.
  3. **Attention pooling** over time steps → fixed-size context vector.
  4. **MLP head**: Dropout → Linear(512, num_classes).

In ``feature_only=True`` mode, the head is removed and the 512-D
attention-pooled BiLSTM output is returned for fusion.

Why 1D CNN + BiLSTM over pure Transformer?
  - Audio sequences have strong local patterns (phoneme-level artefacts)
    that CNNs capture efficiently.
  - BiLSTM models longer-range dependencies across the utterance.
  - This combination is well-validated on speech deepfake detection
    (e.g. ASVspoof 2019 baselines).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_MFCC_DIM = 40
_MEL_DIM  = 128
_INPUT_DIM = _MFCC_DIM + _MEL_DIM   # 168
_CNN_CHANNELS = [64, 128, 256]
_LSTM_HIDDEN = 256
_LSTM_LAYERS = 2
_FEATURE_DIM = _LSTM_HIDDEN * 2      # 512  (bidirectional)


class AudioBranch(nn.Module):
    """1D CNN + BiLSTM audio deepfake detection branch.

    Args:
        input_dim: Number of input feature channels. Default: 248.
        cnn_channels: Channel progression for the 3 CNN layers.
        lstm_hidden: Hidden state size per direction.
        lstm_layers: Number of BiLSTM layers.
        lstm_dropout: Dropout between LSTM layers.
        num_classes: Output classes. Default: 1 (binary BCE).
        dropout_rate: Dropout before the MLP head.
        feature_only: If True, return 512-D features, not logits.
    """

    def __init__(
        self,
        input_dim: int = _INPUT_DIM,
        cnn_channels: list[int] | None = None,
        lstm_hidden: int = _LSTM_HIDDEN,
        lstm_layers: int = _LSTM_LAYERS,
        lstm_dropout: float = 0.3,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        feature_only: bool = False,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = _CNN_CHANNELS

        self.feature_only = feature_only
        self.num_classes = num_classes
        feature_dim = lstm_hidden * 2  # bidirectional

        # ── 1D CNN feature extractor ──────────────────────────────────────────
        cnn_layers: list[nn.Module] = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # ── BiLSTM ────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # ── Attention pooling over time ───────────────────────────────────────
        self.attention = nn.Linear(feature_dim, 1)

        # ── Classification head ────────────────────────────────────────────────
        if not feature_only:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            self.head = nn.Identity()

        logger.info(
            "AudioBranch: input_dim=%d cnn=%s lstm_hidden=%d feature_dim=%d",
            input_dim, cnn_channels, lstm_hidden, feature_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(B, input_dim, T)`` — concatenated MFCC + Mel.

        Returns:
            - ``feature_only=False``: logits ``(B, num_classes)``.
            - ``feature_only=True``: features ``(B, 512)``.
        """
        # CNN: (B, input_dim, T) → (B, 256, T//8)
        x = self.cnn(x)

        # LSTM: (B, T//8, 256) → (B, T//8, 512)
        x = x.permute(0, 2, 1)   # (B, T, C) for LSTM
        lstm_out, _ = self.lstm(x)

        # Attention pooling: (B, T, 512) → (B, 512)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)                 # (B, 512)

        return self.head(context)

    def get_feature_dim(self) -> int:
        """Return the feature vector dimension (512).

        Returns:
            BiLSTM output dimension (lstm_hidden × 2).
        """
        return _FEATURE_DIM

    def count_parameters(self) -> dict[str, int]:
        """Return trainable and total parameter counts.

        Returns:
            Dict with ``trainable`` and ``total`` keys.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable, "total": total}

    def freeze_backbone(self) -> None:
        """Freeze the CNN and LSTM backbone."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False
        logger.info("AudioBranch backbone FROZEN.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze the CNN and LSTM backbone."""
        for param in self.cnn.parameters():
            param.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = True
        logger.info("AudioBranch backbone UNFROZEN.")


def build_audio_branch(
    cfg: dict[str, Any], feature_only: bool = False
) -> AudioBranch:
    """Instantiate an ``AudioBranch`` from config.

    Args:
        cfg: Merged config dict (reads ``cfg["model"]["audio"]``).
        feature_only: If True, return a feature extractor (no head).

    Returns:
        Configured ``AudioBranch`` instance.
    """
    audio_cfg = cfg.get("model", {}).get("audio", {})
    feature_dim = cfg.get("audio", {}).get("feature_dim", _INPUT_DIM)
    
    return AudioBranch(
        input_dim=feature_dim,
        cnn_channels=audio_cfg.get("cnn_channels", _CNN_CHANNELS),
        lstm_hidden=audio_cfg.get("lstm_hidden", _LSTM_HIDDEN),
        lstm_layers=audio_cfg.get("lstm_layers", _LSTM_LAYERS),
        lstm_dropout=audio_cfg.get("lstm_dropout", 0.3),
        num_classes=audio_cfg.get("num_classes", 1),
        dropout_rate=audio_cfg.get("dropout", 0.3),
        feature_only=feature_only,
    )
