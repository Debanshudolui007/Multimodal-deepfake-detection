import cv2
import numpy as np
import torch
import warnings
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Suppress annoying grad-cam warnings
warnings.filterwarnings('ignore', category=UserWarning)

class SpatialGradCAM:
    """Grad-CAM for the ResNeXt50 Spatial Branch."""
    def __init__(self, model, use_cuda=False):
        self.model = model
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Target the last bottleneck layer in ResNeXt50 (timm)
        try:
            # For resnext50_32x4d
            self.target_layers = [self.model.backbone.layer4[-1]]
        except Exception as e:
            print(f"Warning: Could not automatically detect layer4 for GradCAM. Defaulting to None. {e}")
            self.target_layers = []

        if self.target_layers:
            self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, input_tensor: torch.Tensor, original_image_rgb: np.ndarray, target_class: int = 1) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap overlay.
        Args:
            input_tensor: Shape (1, 3, 224, 224), normalized for model input
            original_image_rgb: Shape (224, 224, 3) float in range [0, 1] for overlay
            target_class: 1 for Deepfake, 0 for Real
        """
        if not hasattr(self, 'cam') or not self.target_layers:
            return original_image_rgb
            
        input_tensor = input_tensor.to(self.device)
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate heatmap (requires_grad is handled by GradCAM internally)
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay on original image
        visualization = show_cam_on_image(original_image_rgb, grayscale_cam, use_rgb=True)
        return visualization

class FrequencyEigenCAM:
    """Eigen-CAM for the Swin Transformer Frequency Branch.
    EigenCAM is often more stable for Transformers than standard Grad-CAM.
    """
    def __init__(self, model, use_cuda=False):
        self.model = model
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Target the last norm layer of Swin Transformer
        try:
            self.target_layers = [self.model.backbone.layers[-1].blocks[-1].norm2]
        except Exception:
            try:
                self.target_layers = [self.model.backbone.norm]
            except Exception as e:
                print(f"Warning: Could not automatically detect target layer for SwinCAM. {e}")
                self.target_layers = []

        if self.target_layers:
            self.cam = EigenCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, input_tensor: torch.Tensor, original_image_rgb: np.ndarray, target_class: int = 1) -> np.ndarray:
        if not hasattr(self, 'cam') or not self.target_layers:
            return original_image_rgb
            
        input_tensor = input_tensor.to(self.device)
        targets = [ClassifierOutputTarget(target_class)]
        
        # We might need reshape_transform for Swin, but EigenCAM often handles standard Swin block outputs natively 
        # depending on timm versions. If it crashes, we'll return original image.
        try:
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(original_image_rgb, grayscale_cam, use_rgb=True)
            return visualization
        except Exception as e:
            print(f"Frequency CAM Failed: {e}")
            return original_image_rgb
