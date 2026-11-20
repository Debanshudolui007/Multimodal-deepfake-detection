"""
Deepfake Detector — Unified Command-Line Interface.

Provides a single entry point for all pipeline stages:
  --verify      Run dataset verification (Milestone 1)
  --preprocess  Run frame/face/frequency/audio preprocessing (Milestones 2-3)
  --train       Train a branch or fusion model (Milestones 4-7)
  --evaluate    Run evaluation and benchmarking (Milestone 8)
  --explain     Generate explainability visualisations (Milestone 8)
  --infer       Run inference on a single video (Milestone 9)

Every stage reads configuration exclusively from YAML files; no
hardcoded paths or parameters exist anywhere in this codebase.

Example:
    python deepfake_detector/main.py --verify
    python deepfake_detector/main.py --train spatial
    python deepfake_detector/main.py --infer --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Project root: the directory containing this file's parent package ─────────
_HERE = Path(__file__).resolve().parent          # deepfake_detector/
_PROJECT_ROOT = _HERE.parent                     # deepfake last/
_CONFIGS_DIR = _HERE / "configs"

# ── Ensure project root is on sys.path when run as a script ──────────────────
# This makes `python deepfake_detector/main.py` work identically to
# `python -m deepfake_detector.main`.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="deepfake_detector",
        description="Multimodal Deepfake Detection System — Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mutually exclusive stage flags ────────────────────────────────────────
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument(
        "--verify",
        action="store_true",
        help="Milestone 1: Verify all datasets and generate dataset_report.json.",
    )
    stages.add_argument(
        "--preprocess",
        action="store_true",
        help="Milestones 2-3: Extract frames, detect/align/crop faces, "
             "generate frequency tensors, and extract audio features.",
    )
    stages.add_argument(
        "--train",
        metavar="BRANCH",
        choices=["spatial", "frequency", "audio", "fusion"],
        help="Milestones 4-7: Train the specified branch or fusion model.",
    )
    stages.add_argument(
        "--evaluate",
        metavar="BRANCH",
        nargs="?",
        const="all",
        choices=["spatial", "frequency", "audio", "fusion", "all"],
        help="Milestone 8: Evaluate trained models and generate benchmark report.",
    )
    stages.add_argument(
        "--explain",
        action="store_true",
        help="Milestone 8: Generate Grad-CAM, attention rollout, and audio visualisations.",
    )
    stages.add_argument(
        "--infer",
        action="store_true",
        help="Milestone 9: Run end-to-end inference on a single video.",
    )

    # ── Optional arguments ────────────────────────────────────────────────────
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_CONFIGS_DIR,
        help="Path to the configs/ directory. Default: deepfake_detector/configs/",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Input video path (required for --infer).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="(--evaluate only) Run ablation study across all modality combinations.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for preprocessing. Default: 4.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console log verbosity. Default: INFO.",
    )
    parser.add_argument(
        "--no-integrity-check",
        action="store_true",
        help="Skip dataset integrity checks after verification.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the deepfake detection CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Bootstrap logging before any imports that use it ─────────────────────
    from deepfake_detector.utils.logger import setup_logging
    from deepfake_detector.utils.config import load_all_configs, resolve_paths
    from deepfake_detector.utils.seed import set_all_seeds

    # Determine logs dir from config (or fallback)
    logs_dir = _HERE / "logs"
    setup_logging(logs_dir, run_name="main", level=getattr(logging, args.log_level))

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Deepfake Detector — starting stage: %s", _active_stage(args))
    logger.info("Project root: %s", _PROJECT_ROOT)
    logger.info("=" * 60)

    # ── Load all configs ──────────────────────────────────────────────────────
    try:
        cfg = load_all_configs(args.config_dir)
        cfg = resolve_paths(cfg, _PROJECT_ROOT)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Config error: %s", exc)
        return 1

    # ── Seed ─────────────────────────────────────────────────────────────────
    seed: int = cfg["dataset"].get("random_seed", 42)
    set_all_seeds(seed)

    # ── Dispatch to the correct milestone handler ─────────────────────────────
    try:
        if args.verify:
            return _run_verify(args, cfg, logger)
        if args.preprocess:
            return _run_preprocess(args, cfg, logger)
        if args.train:
            return _run_train(args, cfg, logger)
        if args.evaluate:
            return _run_evaluate(args, cfg, logger)
        if args.explain:
            return _run_explain(args, cfg, logger)
        if args.infer:
            return _run_infer(args, cfg, logger)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Unhandled exception: %s", exc)
        return 1

    return 0


# ── Stage handlers ────────────────────────────────────────────────────────────


def _run_verify(args: argparse.Namespace, cfg: dict, logger: logging.Logger) -> int:
    """Milestone 1: Dataset verification."""
    from deepfake_detector.preprocessing.verify_dataset import (
        run_verification,
        check_dataset_integrity,
    )
    from deepfake_detector.utils.experiment import write_manifest
    from deepfake_detector.utils.memory_monitor import MemoryMonitor

    monitor = MemoryMonitor(interval_sec=60, tag="verify")
    monitor.start()

    try:
        report = run_verification(cfg, _PROJECT_ROOT, num_workers=args.workers)

        if not args.no_integrity_check:
            issues = check_dataset_integrity(report, fail_on_error=False)
            if issues:
                logger.warning(
                    "Integrity check raised %d issue(s). Review logs.", len(issues)
                )

        # Write manifest
        splits_dir = Path(cfg["dataset"]["processed"]["splits"])
        write_manifest(
            output_dir=splits_dir,
            dataset_name="ALL",
            dataset_version="v1",
            cfg=cfg,
            extra={
                "total_valid_videos": report["summary"].get("total_videos_valid", 0),
                "total_skipped_videos": report["summary"].get("total_videos_skipped", 0),
            },
        )

        logger.info(
            "Milestone 1 COMPLETE — %d valid videos, %d skipped.",
            report["summary"].get("total_videos_valid", 0),
            report["summary"].get("total_videos_skipped", 0),
        )
        return 0
    finally:
        monitor.stop()


def _run_preprocess(args: argparse.Namespace, cfg: dict, logger: logging.Logger) -> int:
    """Milestones 2–3: Full preprocessing pipeline."""
    from deepfake_detector.preprocessing.extract_frames import run_frame_extraction
    from deepfake_detector.preprocessing.run_face_pipeline import run_face_pipeline
    from deepfake_detector.utils.memory_monitor import MemoryMonitor

    monitor = MemoryMonitor(interval_sec=60, tag="preprocess")
    monitor.start()
    try:
        # ── Milestone 2: Frame Extraction ─────────────────────────────────────────────
        logger.info("=" * 50)
        logger.info("STAGE 1/2 — Frame Extraction (Milestone 2)")
        logger.info("=" * 50)
        frame_result = run_frame_extraction(cfg, _PROJECT_ROOT, num_workers=args.workers)
        logger.info(
            "Frame extraction complete — processed=%d cached=%d failed=%d total_frames=%d",
            frame_result["processed"],
            frame_result["cached"],
            frame_result["failed"],
            frame_result["total_frames"],
        )

        # ── Milestone 3: Face / Frequency / Audio ──────────────────────────────────
        logger.info("=" * 50)
        logger.info("STAGE 2/2 — Face / Frequency / Audio + Split (Milestone 3)")
        logger.info("=" * 50)
        face_result = run_face_pipeline(cfg, _PROJECT_ROOT, num_workers=1)  # 1 worker: avoid MTCNN init overhead
        logger.info(
            "Face pipeline complete — processed=%d cached=%d failed=%d "
            "total_faces=%d audio_ok=%d elapsed=%.0fs",
            face_result["processed"],
            face_result["cached"],
            face_result["failed"],
            face_result["total_faces"],
            face_result.get("audio_ok", 0),
            face_result["elapsed_sec"],
        )
        logger.info("Splits written to: %s", face_result.get("splits_path", "N/A"))
        return 0
    finally:
        monitor.stop()


def _run_train(args: argparse.Namespace, cfg: dict, logger: logging.Logger) -> int:
    """Milestones 4–7: Branch / fusion training."""
    from deepfake_detector.utils.memory_monitor import MemoryMonitor

    branch = args.train
    experiment_dir = _PROJECT_ROOT / "deepfake_detector" / "experiments" / branch
    experiment_dir.mkdir(parents=True, exist_ok=True)

    splits_path = _PROJECT_ROOT / "deepfake_detector" / "processed" / "splits" / "splits.json"
    faces_root  = _PROJECT_ROOT / "deepfake_detector" / "processed" / "faces"

    if not splits_path.exists():
        logger.error(
            "splits.json not found at %s.\n"
            "Run --preprocess first to generate face crops and splits.",
            splits_path,
        )
        return 1

    monitor = MemoryMonitor(interval_sec=120, tag=f"train_{branch}")
    monitor.start()
    try:
        if branch == "spatial":
            return _train_spatial(cfg, splits_path, faces_root, experiment_dir, logger)
        elif branch == "frequency":
            return _train_frequency(cfg, splits_path, faces_root, experiment_dir, logger)
        elif branch == "audio":
            return _train_audio(cfg, splits_path, experiment_dir, logger)
        elif branch == "fusion":
            return _train_fusion(cfg, splits_path, faces_root, experiment_dir, logger)
    finally:
        monitor.stop()
    return 0


def _train_spatial(
    cfg: dict,
    splits_path: Path,
    faces_root: Path,
    experiment_dir: Path,
    logger: logging.Logger,
) -> int:
    """Train the Spatial Branch (Milestone 4)."""
    import torch
    from torch.utils.data import DataLoader

    from deepfake_detector.datasets_loader.spatial_dataset import (
        SpatialFaceDataset, get_sampler,
    )
    from deepfake_detector.models.spatial_branch import build_spatial_branch
    from deepfake_detector.training.spatial_trainer import build_spatial_trainer

    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("dataloader_workers", 0)  # 0 = main process (safe on Windows)

    logger.info("Building Spatial datasets from: %s", splits_path)
    train_ds = SpatialFaceDataset(splits_path, "train", faces_root, augment=True)
    val_ds   = SpatialFaceDataset(splits_path, "val",   faces_root, augment=False)

    if len(train_ds) == 0:
        logger.error("Training dataset is empty — run --preprocess first.")
        return 1

    sampler = get_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    model = build_spatial_branch(cfg, feature_only=False)
    trainer = build_spatial_trainer(model, cfg, train_loader, val_loader, experiment_dir)

    logger.info(
        "Starting Spatial Branch training: %d train / %d val samples",
        len(train_ds), len(val_ds),
    )
    summary = trainer.train()
    logger.info("Spatial training complete: %s", summary)
    return 0


def _train_frequency(
    cfg: dict,
    splits_path: Path,
    faces_root: Path,
    experiment_dir: Path,
    logger: logging.Logger,
) -> int:
    """Train the Frequency Branch (Milestone 5)."""
    import torch
    from torch.utils.data import DataLoader

    from deepfake_detector.datasets_loader.frequency_dataset import (
        FrequencyDataset, get_frequency_sampler,
    )
    from deepfake_detector.models.frequency_branch import build_frequency_branch
    from deepfake_detector.training.frequency_trainer import build_frequency_trainer

    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("dataloader_workers", 0)

    logger.info("Building Frequency datasets from: %s", splits_path)
    train_ds = FrequencyDataset(splits_path, "train", faces_root, augment=True)
    val_ds   = FrequencyDataset(splits_path, "val",   faces_root, augment=False)

    if len(train_ds) == 0:
        logger.error("Training dataset is empty — run --preprocess first.")
        return 1

    sampler = get_frequency_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    model = build_frequency_branch(cfg, feature_only=False)
    trainer = build_frequency_trainer(model, cfg, train_loader, val_loader, experiment_dir)

    logger.info(
        "Starting Frequency Branch training: %d train / %d val samples",
        len(train_ds), len(val_ds),
    )
    summary = trainer.train()
    logger.info("Frequency training complete: %s", summary)
    return 0


def _train_audio(
    cfg: dict,
    splits_path: Path,
    experiment_dir: Path,
    logger: logging.Logger,
) -> int:
    """Train the Audio Branch (Milestone 6)."""
    import torch
    from torch.utils.data import DataLoader

    from deepfake_detector.datasets_loader.audio_dataset import (
        AudioDataset, get_audio_sampler,
    )
    from deepfake_detector.models.audio_branch import build_audio_branch
    from deepfake_detector.training.audio_trainer import build_audio_trainer

    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("dataloader_workers", 0)

    # Audio tensors are cached in processed/audio/
    audio_root = _PROJECT_ROOT / "deepfake_detector" / "processed" / "audio"

    # Check if a dedicated audio_splits.json exists (e.g. from FakeAVCeleb)
    audio_splits_path = splits_path.with_name("audio_splits.json")
    if audio_splits_path.exists():
        logger.info("Using dedicated audio splits: %s", audio_splits_path)
        splits_path = audio_splits_path

    logger.info("Building Audio datasets from: %s", splits_path)
    train_ds = AudioDataset(splits_path, "train", audio_root, dataset_filter="FakeAVCeleb", augment=True)
    val_ds   = AudioDataset(splits_path, "val",   audio_root, dataset_filter="FakeAVCeleb", augment=False)

    if len(train_ds) == 0:
        logger.error("Training dataset is empty — run --preprocess first.")
        return 1

    sampler = get_audio_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    model = build_audio_branch(cfg, feature_only=False)
    trainer = build_audio_trainer(model, cfg, train_loader, val_loader, experiment_dir)

    logger.info(
        "Starting Audio Branch training: %d train / %d val samples",
        len(train_ds), len(val_ds),
    )
    summary = trainer.train()
    logger.info("Audio training complete: %s", summary)
    return 0


def _train_fusion(
    cfg: dict,
    splits_path: Path,
    faces_root: Path,
    experiment_dir: Path,
    logger: logging.Logger,
) -> int:
    """Train the Fusion Network (Milestone 7)."""
    import torch
    from torch.utils.data import DataLoader

    from deepfake_detector.datasets_loader.multimodal_dataset import (
        MultimodalDataset, get_multimodal_sampler,
    )
    from deepfake_detector.models.fusion_branch import build_fusion_branch
    from deepfake_detector.training.fusion_trainer import FusionTrainer

    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("dataloader_workers", 0)

    # Audio tensors are cached in processed/audio/
    audio_root = faces_root.parent / "audio"
    
    # Fusion must use the same video-level split as the unimodal branches.
    # ``merged_splits.json`` was an auxiliary artifact with records that did
    # not necessarily have face/frequency/audio counterparts, creating a
    # different and invalid distribution for fusion training.
    logger.info("Building Multimodal datasets from canonical split: %s", splits_path)
    train_ds = MultimodalDataset(splits_path, "train", faces_root, audio_root, augment=True)
    val_ds   = MultimodalDataset(splits_path, "val",   faces_root, audio_root, augment=False)

    if len(train_ds) == 0:
        logger.error("Training dataset is empty — run --preprocess first.")
        return 1
    if not _has_both_classes(train_ds, "fusion train", logger):
        return 1
    if not _has_both_classes(val_ds, "fusion validation", logger):
        return 1

    sampler = get_multimodal_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    model = build_fusion_branch(cfg)
    
    # Build optimizer for fusion branch
    # Stage 1 runs with freeze_lr, Stage 2 finetunes with finetune_lr (managed inside FusionTrainer)
    fusion_cfg = cfg.get("fusion", {})
    fusion_training_cfg = fusion_cfg.get("training", {})
    freeze_lr = float(fusion_training_cfg.get("freeze_lr", 1.0e-3))
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=freeze_lr,
        weight_decay=float(fusion_training_cfg.get("weight_decay", 1e-4)),
    )
    
    trainer = FusionTrainer(model, optimizer, train_loader, val_loader, cfg, experiment_dir)

    logger.info(
        "Starting Fusion Network training: %d train / %d val samples",
        len(train_ds), len(val_ds),
    )
    # Start with frozen backbones according to the staging schedule
    model.freeze_backbones()
    
    summary = trainer.train()
    logger.info("Fusion training complete: %s", summary)
    return 0


def _has_both_classes(dataset: object, name: str, logger: logging.Logger) -> bool:
    """Reject a binary training split that lost either class after filtering."""
    get_labels = getattr(dataset, "get_labels", None)
    labels = set(get_labels()) if callable(get_labels) else set()
    if labels == {0, 1}:
        return True
    logger.error(
        "%s must contain both real (0) and fake (1) samples after modality filtering; got %s.",
        name, sorted(labels),
    )
    return False


def _run_evaluate(args: argparse.Namespace, cfg: dict, logger: logging.Logger) -> int:
    """Milestone 8: Evaluation."""
    from deepfake_detector.training.evaluator import run_evaluation
    
    branch = args.evaluate if isinstance(args.evaluate, str) else "all"
    
    splits_path = _PROJECT_ROOT / "deepfake_detector" / "processed" / "splits" / "splits.json"
    faces_root  = _PROJECT_ROOT / "deepfake_detector" / "processed" / "faces"
    
    if not splits_path.exists():
        logger.error("splits.json not found.")
        return 1
        
    return run_evaluation(branch, cfg, splits_path, faces_root, logger)


def _run_explain(_args: argparse.Namespace, _cfg: dict, logger: logging.Logger) -> int:
    """Milestone 8: Explainability."""
    logger.info("Explainability — not yet implemented.")
    return 0


def _run_infer(args: argparse.Namespace, _cfg: dict, logger: logging.Logger) -> int:
    """Milestone 9: Inference."""
    if args.video is None:
        logger.error("--infer requires --video <path>.")
        return 1
    if not args.video.exists():
        logger.error("Video file not found: %s", args.video)
        return 1
    logger.info("Inference — not yet implemented.")
    return 0


def _active_stage(args: argparse.Namespace) -> str:
    """Return a human-readable string of the active stage.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Stage name string.
    """
    if args.verify:
        return "verify"
    if args.preprocess:
        return "preprocess"
    if args.train:
        return f"train:{args.train}"
    if args.evaluate:
        return "evaluate"
    if args.explain:
        return "explain"
    if args.infer:
        return "infer"
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
