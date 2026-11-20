"""
Deepfake Detector — Production-Quality Multimodal Detection System.

Three-branch architecture:
  - Spatial Branch  (ResNeXt50 → 2048-D)
  - Frequency Branch (Swin-T   → 768-D)
  - Audio Branch    (CNN+BiLSTM → 512-D)

Fused via Cross-Attention → MLP → Softmax.
"""

__version__ = "1.0.0"
__author__ = "Deepfake Detection Research Team"
