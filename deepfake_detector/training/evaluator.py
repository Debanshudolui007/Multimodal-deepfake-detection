import logging
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute AUC, accuracy, F1, and EER."""
    import numpy as np
    probs = torch.sigmoid(logits).squeeze(-1).numpy()
    labels = targets.float().numpy()

    if len(np.unique(labels)) < 2:
        return {"auc": 0.0, "acc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "eer": 0.0}

    auc = roc_auc_score(labels, probs)
    preds = (probs > 0.5).astype(np.float32)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))

    return {
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "eer": eer
    }

def run_evaluation(branch: str, cfg: dict, splits_path: Path, faces_root: Path, logger: logging.Logger) -> int:
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if branch == "all":
        branches = ["frequency", "spatial", "audio", "fusion"]
    else:
        branches = [branch]
        
    for b in branches:
        if b == "audio":
            _eval_audio(cfg, splits_path, device, logger)
        elif b == "frequency":
            _eval_frequency(cfg, splits_path, device, logger)
        elif b == "spatial":
            _eval_spatial(cfg, splits_path, device, logger)
        elif b == "fusion":
            _eval_fusion(cfg, splits_path, device, logger)
        else:
            logger.warning(f"Evaluation for {b} is not fully implemented in evaluator yet.")
            
    return 0

def _eval_audio(cfg: dict, splits_path: Path, device: torch.device, logger: logging.Logger):
    from deepfake_detector.datasets_loader.audio_dataset import AudioDataset
    from deepfake_detector.models.audio_branch import build_audio_branch
    
    logger.info("Evaluating AUDIO branch on TEST set...")
    
    # Audio tensors are cached in processed/audio/
    project_root = splits_path.parents[2]
    audio_root = project_root / "processed" / "audio"
    audio_splits_path = splits_path.with_name("audio_splits.json")
    if audio_splits_path.exists():
        splits_path = audio_splits_path
        
    test_ds = AudioDataset(splits_path, "test", audio_root, dataset_filter="FakeAVCeleb", augment=False)
    
    if len(test_ds) == 0:
        logger.error("Test dataset is empty.")
        return
        
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("train", {}).get("batch_size", 32),
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    model = build_audio_branch(cfg, feature_only=False)
    
    ckpt_path = project_root / "experiments" / "audio" / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()
    
    all_logits = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Audio Test Inference")):
            audio, labels, _, _ = batch
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(audio)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds.view_as(labels) == labels.float()).sum().item()
            total_samples += labels.numel()
            
            # Dump live progress for monitor_training.py
            if i % 10 == 0 or i == len(test_loader) - 1:
                try:
                    eval_log = project_root / "experiments" / "audio" / "logs" / "live_eval.json"
                    eval_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(eval_log, "w") as f:
                        json.dump({
                            "phase": "evaluate",
                            "batch": i + 1,
                            "n_batches": len(test_loader),
                            "acc": total_correct / max(total_samples, 1)
                        }, f)
                except Exception:
                    pass
            
    all_logits_t = torch.cat(all_logits, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    
    metrics = _compute_metrics(all_logits_t, all_labels_t)
    
    logger.info(
        "\n================ AUDIO TEST RESULTS ================\n"
        f"  AUC-ROC:   {metrics['auc']:.4f}\n"
        f"  Accuracy:  {metrics['acc']:.4f}\n"
        f"  F1 Score:  {metrics['f1']:.4f}\n"
        f"  Precision: {metrics['precision']:.4f}\n"
        f"  Recall:    {metrics['recall']:.4f}\n"
        f"  EER:       {metrics['eer']:.4f}\n"
        "===================================================="
    )
    
    # Save benchmark
    bench_path = project_root / "experiments" / "audio" / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved benchmark results to {bench_path}")

def _eval_frequency(cfg: dict, splits_path: Path, device: torch.device, logger: logging.Logger):
    from deepfake_detector.datasets_loader.frequency_dataset import FrequencyDataset
    from deepfake_detector.models.frequency_branch import build_frequency_branch
    
    logger.info("Evaluating FREQUENCY branch on TEST set...")
    
    project_root = splits_path.parents[2]
    faces_root = project_root / "processed" / "faces"
        
    test_ds = FrequencyDataset(splits_path, "test", faces_root, augment=False)
    
    if len(test_ds) == 0:
        logger.error("Test dataset is empty.")
        return
        
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("train", {}).get("batch_size", 32),
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    model = build_frequency_branch(cfg, feature_only=False)
    
    ckpt_path = project_root / "experiments" / "frequency" / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()
    
    all_logits = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Frequency Test Inference")):
            images, labels, _, _ = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds.view_as(labels) == labels.float()).sum().item()
            total_samples += labels.numel()
            
            # Dump live progress for monitor_training.py
            if i % 10 == 0 or i == len(test_loader) - 1:
                try:
                    eval_log = project_root / "experiments" / "frequency" / "logs" / "live_eval.json"
                    eval_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(eval_log, "w") as f:
                        json.dump({
                            "phase": "evaluate",
                            "batch": i + 1,
                            "n_batches": len(test_loader),
                            "acc": total_correct / max(total_samples, 1)
                        }, f)
                except Exception:
                    pass
            
    all_logits_t = torch.cat(all_logits, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    
    metrics = _compute_metrics(all_logits_t, all_labels_t)
    
    logger.info(
        "\n================ FREQUENCY TEST RESULTS ================\n"
        f"  AUC-ROC:   {metrics['auc']:.4f}\n"
        f"  Accuracy:  {metrics['acc']:.4f}\n"
        f"  F1 Score:  {metrics['f1']:.4f}\n"
        f"  Precision: {metrics['precision']:.4f}\n"
        f"  Recall:    {metrics['recall']:.4f}\n"
        f"  EER:       {metrics['eer']:.4f}\n"
        "========================================================"
    )
    
    bench_path = project_root / "experiments" / "frequency" / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved benchmark results to {bench_path}")

def _eval_spatial(cfg: dict, splits_path: Path, device: torch.device, logger: logging.Logger):
    from deepfake_detector.datasets_loader.spatial_dataset import SpatialFaceDataset
    from deepfake_detector.models.spatial_branch import build_spatial_branch
    
    logger.info("Evaluating SPATIAL branch on TEST set...")
    
    project_root = splits_path.parents[2]
    faces_root = project_root / "processed" / "faces"
        
    test_ds = SpatialFaceDataset(splits_path, "test", faces_root, augment=False)
    
    if len(test_ds) == 0:
        logger.error("Test dataset is empty.")
        return
        
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("train", {}).get("batch_size", 32),
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    model = build_spatial_branch(cfg, feature_only=False)
    
    ckpt_path = project_root / "experiments" / "spatial" / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()
    
    all_logits = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Spatial Test Inference")):
            images, labels, _, _ = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds.view_as(labels) == labels.float()).sum().item()
            total_samples += labels.numel()
            
            # Dump live progress for monitor_training.py
            if i % 10 == 0 or i == len(test_loader) - 1:
                try:
                    eval_log = project_root / "experiments" / "spatial" / "logs" / "live_eval.json"
                    eval_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(eval_log, "w") as f:
                        json.dump({
                            "phase": "evaluate",
                            "batch": i + 1,
                            "n_batches": len(test_loader),
                            "acc": total_correct / max(total_samples, 1)
                        }, f)
                except Exception:
                    pass
            
    all_logits_t = torch.cat(all_logits, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    
    metrics = _compute_metrics(all_logits_t, all_labels_t)
    
    logger.info(
        "\n================ SPATIAL TEST RESULTS ================\n"
        f"  AUC-ROC:   {metrics['auc']:.4f}\n"
        f"  Accuracy:  {metrics['acc']:.4f}\n"
        f"  F1 Score:  {metrics['f1']:.4f}\n"
        f"  Precision: {metrics['precision']:.4f}\n"
        f"  Recall:    {metrics['recall']:.4f}\n"
        f"  EER:       {metrics['eer']:.4f}\n"
        "========================================================"
    )
    
    bench_path = project_root / "experiments" / "spatial" / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved benchmark results to {bench_path}")

def _eval_fusion(cfg: dict, splits_path: Path, device: torch.device, logger: logging.Logger):
    from deepfake_detector.datasets_loader.multimodal_dataset import MultimodalDataset
    from deepfake_detector.models.fusion_branch import build_fusion_branch
    
    logger.info("Evaluating FUSION branch on TEST set...")
    
    project_root = splits_path.parents[2]
    faces_root = project_root / "processed" / "faces"
    audio_root = project_root / "processed" / "audio"
        
    test_ds = MultimodalDataset(splits_path, "test", faces_root, audio_root, augment=False)
    
    if len(test_ds) == 0:
        logger.error("Test dataset is empty.")
        return
        
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("train", {}).get("batch_size", 32),
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    model = build_fusion_branch(cfg)
    
    ckpt_path = project_root / "experiments" / "fusion" / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()
    
    all_logits = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Fusion Test Inference")):
            spatial_tensor, freq_tensor, audio_tensor, mask, labels, _, _ = batch
            spatial_tensor = spatial_tensor.to(device, non_blocking=True)
            freq_tensor = freq_tensor.to(device, non_blocking=True)
            audio_tensor = audio_tensor.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(spatial_tensor, freq_tensor, audio_tensor, modality_mask=mask)
            logits = outputs["fused"]
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds.view_as(labels) == labels.float()).sum().item()
            total_samples += labels.numel()
            
            # Dump live progress for monitor_training.py
            if i % 10 == 0 or i == len(test_loader) - 1:
                try:
                    eval_log = project_root / "experiments" / "fusion" / "logs" / "live_eval.json"
                    eval_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(eval_log, "w") as f:
                        json.dump({
                            "phase": "evaluate",
                            "batch": i + 1,
                            "n_batches": len(test_loader),
                            "acc": total_correct / max(total_samples, 1)
                        }, f)
                except Exception:
                    pass
            
    all_logits_t = torch.cat(all_logits, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    
    metrics = _compute_metrics(all_logits_t, all_labels_t)
    
    logger.info(
        "\n================ FUSION TEST RESULTS ================\n"
        f"  AUC-ROC:   {metrics['auc']:.4f}\n"
        f"  Accuracy:  {metrics['acc']:.4f}\n"
        f"  F1 Score:  {metrics['f1']:.4f}\n"
        f"  Precision: {metrics['precision']:.4f}\n"
        f"  Recall:    {metrics['recall']:.4f}\n"
        f"  EER:       {metrics['eer']:.4f}\n"
        "====================================================="
    )
    
    bench_path = project_root / "experiments" / "fusion" / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved benchmark results to {bench_path}")
