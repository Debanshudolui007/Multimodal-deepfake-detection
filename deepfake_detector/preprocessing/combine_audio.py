import os
import numpy as np
from pathlib import Path

audio_root = Path(r"C:\Users\deban\Downloads\CEDS INTERNSHIP\deepfake last\deepfake_detector\processed\audio")
count = 0

for root, _, files in os.walk(audio_root):
    if "mfcc.npy" in files and "mel.npy" in files:
        video_id = Path(root).name
        mfcc = np.load(os.path.join(root, "mfcc.npy"))
        mel = np.load(os.path.join(root, "mel.npy"))
        combined = np.vstack([mfcc, mel])
        out_path = os.path.join(root, f"{video_id}_audio.npy")
        np.save(out_path, combined)
        count += 1

print(f"Combined {count} audio files!")
