"""
Milestone 3 — Dataset Split Module.

Splits the processed face-crop dataset into train / val / test subsets at
the **video level**, preventing identity leakage between splits.

Key properties:
  - Split is performed by unique video IDs, not by individual face crops.
  - Class balance is preserved: real/fake ratios are consistent across splits.
  - Labels are allocated globally so every sufficiently large class has real
    and fake examples in train, validation, and test.
  - The split assignment is deterministic given the random seed.
  - Outputs a ``splits.json`` file listing video IDs per split per dataset.

Split ratios are read from ``cfg["dataset"]["split"]``:
  ``{ "train": 0.70, "val": 0.15, "test": 0.15 }``

The resulting ``splits.json`` is the reference used by all dataset loaders
(Milestone 4+) to construct their PyTorch Datasets.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Split names ───────────────────────────────────────────────────────────────
SPLITS = ("train", "val", "test")


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def build_splits(
    faces_root: Path,
    output_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Discover face-crop directories and assign each to a split.

    The function walks ``faces_root`` to enumerate all
    ``<dataset>/<label>/<video_id>`` directories that contain at least
    one ``.png`` file. It then assigns each video to train/val/test
    proportionally, stratified by dataset+label.

    Args:
        faces_root: Root directory of extracted face crops
            (``processed/faces/``).
        output_path: File path where ``splits.json`` will be written.
        train_ratio: Fraction of videos for training.
        val_ratio: Fraction of videos for validation.
        seed: Random seed for reproducible splitting.

    Returns:
        The splits dictionary written to disk:
        ``{ "train": [{video_id, dataset, label, face_dir}], ... }``

    Raises:
        ValueError: If the split ratios do not sum to ≤ 1.0 or no videos
            are discovered.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0 or abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0. "
            f"Got train={train_ratio}, val={val_ratio}, test={test_ratio:.3f}"
        )

    # ── Discover videos ────────────────────────────────────────────────────────
    videos = _discover_videos(faces_root)
    if not videos:
        raise ValueError(f"No face-crop directories found under: {faces_root}")

    logger.info(
        "Splitting %d videos (train=%.0f%% val=%.0f%% test=%.0f%%) seed=%d",
        len(videos), train_ratio * 100, val_ratio * 100, test_ratio * 100, seed,
    )

    # ── Stratified split ──────────────────────────────────────────────────────
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    # Stratify by label globally.  Per-dataset rounding previously allocated
    # small real-video groups only to train/validation, leaving the test set
    # with no real examples at all.  A globally stratified split preserves a
    # usable class distribution; dataset membership remains in every record
    # for later cross-dataset experiments.
    groups: dict[str, list[dict[str, Any]]] = {}
    for v in videos:
        groups.setdefault(v["label"], []).append(v)

    for lbl, group_videos in sorted(groups.items()):
        rng.shuffle(group_videos)
        n = len(group_videos)
        n_train, n_val, n_test = _allocate_split_counts(
            n, train_ratio, val_ratio, test_ratio
        )

        splits["train"].extend(group_videos[:n_train])
        splits["val"].extend(group_videos[n_train: n_train + n_val])
        splits["test"].extend(group_videos[n_train + n_val:])

        logger.info(
            "[%s] %d videos → train=%d val=%d test=%d",
            lbl, n, n_train, n_val, n_test,
        )

    # ── Stats ─────────────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "total_videos": len(videos),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(test_ratio, 4),
        },
        "seed": seed,
    }
    _check_no_leakage(splits)

    # ── Write to disk ──────────────────────────────────────────────────────────
    output: dict[str, Any] = {
        "summary": summary,
        "splits": splits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(
        "Splits written: %s | train=%d val=%d test=%d",
        output_path, len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return output


def load_splits(splits_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load a previously computed splits.json file.

    Args:
        splits_path: Path to ``splits.json``.

    Returns:
        Dictionary mapping split name → list of video metadata dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not splits_path.exists():
        raise FileNotFoundError(
            f"splits.json not found: {splits_path}. Run --preprocess first."
        )
    with splits_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["splits"]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _discover_videos(faces_root: Path) -> list[dict[str, Any]]:
    """Walk the faces directory to enumerate all video-level directories.

    Expected layout: ``<faces_root>/<dataset>/<label>/<video_id>/``.

    Args:
        faces_root: Root directory of face crops.

    Returns:
        List of dicts: ``{video_id, dataset, label, face_dir, face_count}``.
    """
    videos: list[dict[str, Any]] = []
    if not faces_root.exists():
        logger.warning("Faces root does not exist: %s", faces_root)
        return videos

    for dataset_dir in sorted(faces_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for label_dir in sorted(dataset_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            for video_dir in sorted(label_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                face_count = len(list(video_dir.glob("*.png")))
                if face_count == 0:
                    continue
                videos.append(
                    {
                        "video_id": video_dir.name,
                        "dataset": dataset_dir.name,
                        "label": label_dir.name,
                        "face_dir": str(video_dir),
                        "face_count": face_count,
                    }
                )
    return videos


def _allocate_split_counts(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Allocate a class group without silently dropping a split.

    For three or more examples each split receives at least one item.  This
    matters most for the minority class: an all-fake validation or test split
    makes ROC-AUC, calibration, and real-class recall meaningless.
    """
    if n < 1:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    # Reserve one member for each split, then distribute remaining examples by
    # the requested ratios using largest remainder rounding.
    remaining = n - 3
    ratios = (train_ratio, val_ratio, test_ratio)
    raw = [remaining * ratio for ratio in ratios]
    counts = [1 + int(value) for value in raw]
    leftovers = remaining - sum(int(value) for value in raw)
    order = sorted(range(3), key=lambda index: raw[index] - int(raw[index]), reverse=True)
    for index in order[:leftovers]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _check_no_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Assert that no video ID appears in more than one split.

    Args:
        splits: Dict mapping split name -> list of video dicts.

    Raises:
        AssertionError: If any video ID is duplicated across splits.
    """
    seen: dict[str, str] = {}
    for split_name, videos in splits.items():
        for v in videos:
            # Use dataset + label + video_id to avoid collisions between datasets and classes
            vid_key = f"{v['dataset']}/{v['label']}/{v['video_id']}"
            if vid_key in seen:
                raise AssertionError(
                    f"Identity leakage: video '{vid_key}' appears in both "
                    f"'{seen[vid_key]}' and '{split_name}'."
                )
            seen[vid_key] = split_name
    logger.debug("No identity leakage detected across %d videos.", len(seen))
