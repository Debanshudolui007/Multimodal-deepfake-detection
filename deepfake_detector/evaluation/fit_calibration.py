"""Attach held-out Platt-calibration parameters to a copied checkpoint.

Example:
    python -m deepfake_detector.evaluation.fit_calibration \
        --checkpoint experiments/fusion/checkpoints/best.pth \
        --logits calibration_logits.npy --labels calibration_labels.npy \
        --output-checkpoint experiments/fusion/checkpoints/best_calibrated.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from deepfake_detector.evaluation.calibration import fit_platt_scaler


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit Platt scaling on held-out logits and create a calibrated checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.output_checkpoint.exists():
        parser.error(f"refusing to overwrite existing file: {args.output_checkpoint}")

    logits = np.load(args.logits)
    labels = np.load(args.labels)
    calibration = fit_platt_scaler(logits, labels)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint["calibration"] = calibration
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_checkpoint)
    print(
        f"Wrote calibrated checkpoint to {args.output_checkpoint} "
        f"(a={calibration['a']:.6f}, b={calibration['b']:.6f}, "
        f"n={calibration['n_samples']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
