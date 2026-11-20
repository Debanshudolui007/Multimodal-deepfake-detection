"""
Milestone 3 — Audio Feature Extraction Module.

Extracts audio from video files and computes two spectral representations:

1. **MFCC** (Mel-Frequency Cepstral Coefficients):
   - 40 coefficients, delta and delta-delta (total 120 features)
   - Fixed temporal length: 300 frames (padded/trimmed)
   - Shape: (120, 300) float32 NumPy array

2. **Mel Spectrogram**:
   - 128 Mel filter banks, log-scaled
   - Fixed temporal length: 300 frames (padded/trimmed)
   - Shape: (128, 300) float32 NumPy array

Both are saved as ``.npy`` files. Additionally, the extracted WAV file is
retained for future analysis.

Audio extraction uses ``ffmpeg`` (subprocess) to demux the audio stream
directly from the source video, bypassing OpenCV's limited audio support.
Falls back gracefully if ``ffmpeg`` is unavailable or the video has no audio.

Output structure::

    processed/audio/
    └── <dataset>/
        └── <label>/
            └── <video_id>/
                ├── audio.wav
                ├── mfcc.npy          # (120, 300) float32
                ├── mel.npy           # (128, 300) float32
                └── audio_meta.json
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# ── Audio settings ────────────────────────────────────────────────────────────
_SAMPLE_RATE = 16_000          # Target sample rate (Hz)
_N_MFCC = 40                   # Number of MFCC coefficients
_N_MELS = 128                  # Number of Mel filterbanks
_N_FFT = 512                   # FFT window size
_HOP_LENGTH = 160              # STFT hop length
_FIXED_FRAMES = 300            # Temporal dimension (padded or trimmed)
_MONO = True                   # Downmix to mono

# ── File names ────────────────────────────────────────────────────────────────
_WAV_FILENAME = "audio.wav"
_MFCC_FILENAME = "mfcc.npy"
_MEL_FILENAME = "mel.npy"
_META_FILENAME = "audio_meta.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def extract_audio_features(
    video_path: Path,
    output_dir: Path,
    dataset: str,
    label: str,
    video_id: str,
    force: bool = False,
    audio_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract MFCC and Mel spectrogram features from a video's audio track.

    Args:
        video_path: Absolute path to the source video file.
        output_dir: Root audio output directory (``processed/audio/``).
        dataset: Dataset name (e.g. ``"celebdf"``).
        label: ``"real"`` or ``"fake"``.
        video_id: Video stem (used as directory name).
        force: If True, re-extract even if cached files exist.
        audio_cfg: Optional audio configuration dictionary.

    Returns:
        Summary dict:
          ``{status: "ok"/"cached"/"no_audio"/"failed", video_id, ...}``
    """
    audio_dir = output_dir / dataset / label / video_id
    meta_path = audio_dir / _META_FILENAME

    # ── Cache check ───────────────────────────────────────────────────────────
    if not force and _is_cached(audio_dir):
        return {"status": "cached", "video_id": video_id}

    audio_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract WAV ────────────────────────────────────────────────────
    wav_path = audio_dir / _WAV_FILENAME
    extract_ok = _extract_wav(video_path, wav_path)
    if not extract_ok:
        _write_meta(meta_path, video_id, dataset, label, status="no_audio")
        return {"status": "no_audio", "video_id": video_id}

    # ── Step 2: Compute features ───────────────────────────────────────────────
    try:
        y, sr = librosa.load(str(wav_path), sr=_SAMPLE_RATE, mono=_MONO)
    except Exception as exc:
        logger.warning("librosa failed to load WAV for %s: %s", video_id, exc)
        _write_meta(meta_path, video_id, dataset, label, status="failed", error=str(exc))
        return {"status": "failed", "video_id": video_id, "error": str(exc)}

    mfcc = _compute_mfcc(y, sr, audio_cfg)
    mel = _compute_mel(y, sr)

    # ── Step 3: Save features ──────────────────────────────────────────────────
    mfcc_path = audio_dir / _MFCC_FILENAME
    mel_path = audio_dir / _MEL_FILENAME
    np.save(str(mfcc_path), mfcc)
    np.save(str(mel_path), mel)

    meta = _write_meta(
        meta_path,
        video_id, dataset, label,
        status="ok",
        duration_sec=round(len(y) / _SAMPLE_RATE, 2),
        sample_rate=sr,
        mfcc_shape=list(mfcc.shape),
        mel_shape=list(mel.shape),
    )

    logger.debug(
        "[%s/%s/%s] MFCC=%s Mel=%s",
        dataset, label, video_id, mfcc.shape, mel.shape,
    )
    return {"status": "ok", "video_id": video_id, "meta": meta}


# ═══════════════════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_mfcc(y: np.ndarray, sr: int, audio_cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Compute MFCC features, optionally with deltas.

    Args:
        y: Audio waveform (mono float32).
        sr: Sample rate in Hz.
        audio_cfg: Config dict to check 'use_delta' and 'use_delta_delta'.

    Returns:
        Float32 array of shape ``(features, _FIXED_FRAMES)``.
    """
    if audio_cfg is None:
        audio_cfg = {}
        
    mfcc_cfg = audio_cfg.get("mfcc", {})
    use_delta = mfcc_cfg.get("use_delta", False)
    use_delta_delta = mfcc_cfg.get("use_delta_delta", False)
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=_N_MFCC,
        n_fft=_N_FFT, hop_length=_HOP_LENGTH,
    )
    
    feats = [mfcc]
    if use_delta or use_delta_delta:
        delta1 = librosa.feature.delta(mfcc, order=1)
        if use_delta:
            feats.append(delta1)
        if use_delta_delta:
            delta2 = librosa.feature.delta(mfcc, order=2)
            feats.append(delta2)
            
    stacked = np.vstack(feats)       # (N, T)
    return _pad_or_trim(stacked, _FIXED_FRAMES).astype(np.float32)


def _compute_mel(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute a log-scaled Mel spectrogram.

    Args:
        y: Audio waveform (mono float32).
        sr: Sample rate in Hz.

    Returns:
        Float32 array of shape ``(128, _FIXED_FRAMES)``.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=_N_MELS,
        n_fft=_N_FFT, hop_length=_HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)     # Log-scaled
    return _pad_or_trim(mel_db, _FIXED_FRAMES).astype(np.float32)


def _pad_or_trim(arr: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad or truncate the time axis of a 2D feature array.

    Args:
        arr: 2D array of shape ``(features, time_frames)``.
        target_frames: Desired number of time frames.

    Returns:
        Array of shape ``(features, target_frames)``.
    """
    _, t = arr.shape
    if t >= target_frames:
        return arr[:, :target_frames]
    # Zero-pad on the right
    pad_width = target_frames - t
    return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Audio extraction via ffmpeg
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_wav(video_path: Path, wav_path: Path) -> bool:
    """Use ffmpeg to extract the audio stream from a video file.

    Args:
        video_path: Source video file path.
        wav_path: Destination WAV file path.

    Returns:
        True if extraction succeeded (WAV file exists and is non-empty).
        False if the video has no audio stream or ffmpeg fails.
    """
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vn",                          # Disable video
        "-acodec", "pcm_s16le",         # Raw PCM output
        "-ar", str(_SAMPLE_RATE),       # Resample to 16 kHz
        "-ac", "1",                     # Mono
        str(wav_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            # Check stderr for "no audio" indication
            stderr = result.stderr.decode("utf-8", errors="replace")
            if "no audio" in stderr.lower() or "stream 0" not in stderr.lower():
                logger.debug("No audio stream in %s", video_path.name)
            else:
                logger.warning(
                    "ffmpeg failed for %s (rc=%d): %s",
                    video_path.name, result.returncode, stderr[:200],
                )
            return False

        return wav_path.exists() and wav_path.stat().st_size > 1000

    except FileNotFoundError:
        logger.debug("ffmpeg not found — audio extraction skipped for %s", video_path.name)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out for %s", video_path.name)
        return False
    except Exception as exc:
        logger.warning("Audio extraction error for %s: %s", video_path.name, exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _is_cached(audio_dir: Path) -> bool:
    """Check if audio features are fully cached for a video.

    Args:
        audio_dir: Directory for a specific video's audio outputs.

    Returns:
        True if mfcc.npy, mel.npy, and audio_meta.json all exist.
    """
    required = [_MFCC_FILENAME, _MEL_FILENAME, _META_FILENAME]
    return all((audio_dir / f).exists() for f in required)


def _write_meta(
    path: Path,
    video_id: str,
    dataset: str,
    label: str,
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    """Write audio extraction metadata to a JSON file.

    Args:
        path: Destination path for ``audio_meta.json``.
        video_id: Source video ID.
        dataset: Dataset name.
        label: Ground truth label.
        status: Outcome: ``"ok"``, ``"no_audio"``, ``"failed"``, ``"cached"``.
        **extra: Additional fields to include (e.g., shapes, duration).

    Returns:
        The metadata dictionary that was written.
    """
    meta: dict[str, Any] = {
        "video_id": video_id,
        "dataset": dataset,
        "label": label,
        "status": status,
        "sample_rate": _SAMPLE_RATE,
        "n_mfcc": _N_MFCC,
        "n_mels": _N_MELS,
        "fixed_frames": _FIXED_FRAMES,
        "extracted_at": datetime.now(tz=timezone.utc).isoformat(),
        **extra,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
