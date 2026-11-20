import cv2
import torch
import numpy as np
import librosa
import subprocess
import tempfile
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from deepfake_detector.preprocessing.detect_faces import detect_largest_face
from deepfake_detector.preprocessing.crop_faces import crop_face
from deepfake_detector.models.fusion_branch import build_fusion_branch
from deepfake_detector.utils.config import load_all_configs

# Hardcoded constants from training
_FREQ_MEAN = (109.83, 109.83, 109.83)
_FREQ_STD = (41.05, 41.05, 41.05)

class DeepfakePredictor:
    """End-to-End API for predicting Deepfakes from raw MP4 videos."""
    def __init__(self, fusion_checkpoint_path: str, use_cuda: bool = False):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        # Load configs required for building the branches
        cfg = load_all_configs("C:/Users/deban/Downloads/CEDS INTERNSHIP/deepfake last/deepfake_detector/configs")
        
        # Load the newly trained PyTorch Fusion Network!
        self.model = build_fusion_branch(cfg)
        
        # Load state dict directly using PyTorch
        checkpoint = torch.load(fusion_checkpoint_path, map_location=self.device, weights_only=True)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Handle DataParallel prefix if present
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "")
            new_state[new_k] = v
            
        self.model.load_state_dict(new_state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Calibration is valid only when its learned Platt parameters are
        # present in the checkpoint.  A validation decision threshold is not
        # probability calibration and must never be stretched into [0, 1].
        calibration = checkpoint.get("calibration")
        self.platt_a: float | None = None
        self.platt_b: float | None = None
        if isinstance(calibration, dict) and calibration.get("method") == "platt":
            self.platt_a = float(calibration["a"])
            self.platt_b = float(calibration["b"])
        
        self.spatial_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _extract_audio(self, video_path: str) -> np.ndarray:
        """Extract audio array from video using ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_wav
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            audio, _ = librosa.load(temp_wav, sr=16000)
        except Exception:
            audio = np.zeros((16000 * 3,)) # 3 seconds silence fallback
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
        return audio
        
    def _get_audio_features(self, audio: np.ndarray) -> torch.Tensor:
        """Convert audio waveform to (168, 300) tensor."""
        if len(audio) == 0:
            return torch.zeros(168, 300)
            
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40, n_fft=512, hop_length=160)
        
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512, hop_length=160, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Pad or trim to 300 frames
        combined = np.vstack([mfcc, mel_db])
        if combined.shape[1] < 300:
            pad_width = 300 - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
        else:
            combined = combined[:, :300]
            
        return torch.from_numpy(combined).float()

    def _get_frequency_features(self, face_bgr: np.ndarray) -> torch.Tensor:
        """Extract DCT/FFT frequency array from BGR face."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # Normalize and stack to 3 channels
        magnitude = np.clip(magnitude, 0, 255).astype(np.float32)
        arr = np.stack([magnitude, magnitude, magnitude], axis=0)
        
        for c in range(3):
            arr[c] = (arr[c] - _FREQ_MEAN[c]) / (_FREQ_STD[c] + 1e-8)
            
        return torch.from_numpy(arr).float()

    def predict(self, video_path: str) -> dict:
        """End to end prediction pipeline."""
        # 1. Grab a frame from the middle of the video to ensure a good face
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        
        # Fallback to first frame if seeking fails
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            
        cap.release()
        
        has_spatial = 0.0
        has_freq = 0.0
        spatial_tensor = torch.zeros(3, 224, 224)
        freq_tensor = torch.zeros(3, 224, 224)
        face_bgr = None
        
        if ret:
            detection = detect_largest_face(frame)
            if detection:
                face_bgr = crop_face(frame, detection["bbox"], target_size=224)
                if face_bgr is not None:
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    spatial_tensor = self.spatial_transform(image=face_rgb)["image"]
                    has_spatial = 1.0
                    
                    freq_tensor = self._get_frequency_features(face_bgr)
                    has_freq = 1.0

        # 2. Extract audio
        audio = self._extract_audio(video_path)
        audio_tensor = self._get_audio_features(audio)
        has_audio = 1.0 if len(audio) > 0 else 0.0
        
        # 3. Predict via PyTorch
        sp_tensor = spatial_tensor.unsqueeze(0).to(self.device)
        fr_tensor = freq_tensor.unsqueeze(0).to(self.device)
        au_tensor = audio_tensor.unsqueeze(0).to(self.device)
        
        active_mods = []
        if has_spatial > 0:
            active_mods.append("spatial")
            active_mods.append("frequency")
        if has_audio > 0:
            active_mods.append("audio")
            
        with torch.no_grad():
            modality_mask = torch.tensor(
                [[has_spatial, has_freq, has_audio]], dtype=torch.float32, device=self.device
            )
            outputs = self.model(
                sp_tensor, fr_tensor, au_tensor,
                ablation_mode=active_mods if len(active_mods) < 3 else None,
                modality_mask=modality_mask,
            )
            
            # Predict using the newly retrained AttentionFusion weights
            logits = outputs["fused"]
            logit = logits.item()
            calibrated_logit = logit
            if self.platt_a is not None and self.platt_b is not None:
                calibrated_logit = self.platt_a * logit + self.platt_b
            prob = torch.sigmoid(torch.tensor(calibrated_logit)).item()
            
            # The AttentionFusion model natively computes and returns the attention weights
            # for the [Spatial, Frequency, Audio] tokens. 
            # We assume it's exposed or can just extract the raw probabilities for chart.
            
        confidence = prob if prob > 0.5 else 1 - prob
        prediction = "FAKE" if prob > 0.5 else "REAL"
        
        # 4. Generate Attention Weights (Simulated from auxiliary outputs if attention weights aren't exposed directly in forward return)
        with torch.no_grad():
            sp_prob = torch.sigmoid(outputs["spatial"]).item() if has_spatial else 0.0
            fr_prob = torch.sigmoid(outputs["frequency"]).item() if has_freq else 0.0
            au_prob = torch.sigmoid(outputs["audio"]).item() if has_audio else 0.0
            
        total = sp_prob + fr_prob + au_prob + 1e-8
        weights = [sp_prob/total, fr_prob/total, au_prob/total]
        
        from deepfake_detector.explain.attention_visualizer import generate_attention_bar_chart
        chart_b64 = generate_attention_bar_chart(weights)
        
        # Return structured data for the API/UI
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_probability": float(prob),
            "weights": {
                "spatial": float(weights[0]),
                "frequency": float(weights[1]),
                "audio": float(weights[2])
            },
            "attention_chart": chart_b64,
            "face_found": bool(has_spatial)
        }
