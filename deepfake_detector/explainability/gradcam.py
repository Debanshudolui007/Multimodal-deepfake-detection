"""
Milestone 8 — Explainability (Grad-CAM).

Provides Grad-CAM (Gradient-weighted Class Activation Mapping) for the Spatial Branch.
Useful for visually confirming that the model focuses on the face/artefacts 
rather than background noise.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    """Computes Grad-CAM for a given CNN model and target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output

    def save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Generate heatmap.
        
        Args:
            x: Input tensor (1, C, H, W).
            target_class: Index of the target class. If None, uses the argmax prediction.
            
        Returns:
            Normalized heatmap array (H, W) in [0, 1].
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(x)
        
        if target_class is None:
            if logits.size(1) == 1:
                target_class = 0 # Binary BCE
            else:
                target_class = int(logits.argmax(dim=1).item())
        
        # Backward pass
        self.model.zero_grad()
        if logits.size(1) == 1:
            score = logits[0, 0]
        else:
            score = logits[0, target_class]
            
        score.backward()

        # Compute Grad-CAM
        assert self.gradients is not None and self.activations is not None
        
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize and resize to original image size
        cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays the heatmap onto the original RGB image."""
    # Ensure heatmap is uint8 [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply JET colormap
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, colormap, alpha, 0)
    return overlay
