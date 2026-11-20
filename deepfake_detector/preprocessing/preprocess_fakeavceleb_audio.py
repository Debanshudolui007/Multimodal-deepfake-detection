"""
Extract Audio from FakeAVCeleb_v1.2

Reads meta_data.csv, determines Audio Fake/Real labels,
and extracts MFCC/Mel features using extract_audio.py
"""

import csv
import logging
from pathlib import Path
from deepfake_detector.preprocessing.extract_audio import extract_audio_features
from deepfake_detector.utils.config import load_all_configs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    dataset_dir = Path(r"D:\proj\deepfake last\DATASET\FAKEAVCELEB\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")
    meta_csv = dataset_dir / "meta_data.csv"
    
    cfg = load_all_configs(Path(r"d:\proj\deepfake last\deepfake_detector\configs"))
    
    output_dir = Path(r"d:\proj\deepfake last\deepfake_detector\processed\audio")
    
    if not meta_csv.exists():
        logger.error(f"Cannot find {meta_csv}")
        return

    # Fake Audio condition: type is FakeVideo-FakeAudio or RealVideo-FakeAudio
    tasks = []
    
    for mp4_path in dataset_dir.rglob("*.mp4"):
        # The path looks like: .../dataset/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/men/id00076/00109.mp4
        # We can find the type by checking the parts
        parts = mp4_path.parts
        video_type = None
        for pt in parts:
            if pt in ["FakeVideo-FakeAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "RealVideo-RealAudio"]:
                video_type = pt
                break
                
        if not video_type:
            continue
            
        is_audio_fake = video_type in ["FakeVideo-FakeAudio", "RealVideo-FakeAudio"]
        label_str = "fake" if is_audio_fake else "real"
        
        tasks.append({
            "video_path": mp4_path,
            "label": label_str,
            "video_id": mp4_path.stem
        })

    logger.info(f"Found {len(tasks)} videos to process in FakeAVCeleb.")
    
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    num_workers = min(multiprocessing.cpu_count(), 8)
    logger.info(f"Using {num_workers} workers...")
    
    processed = 0
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for task in tasks:
            futures.append(
                executor.submit(
                    extract_audio_features,
                    video_path=task["video_path"],
                    output_dir=output_dir,
                    dataset="FakeAVCeleb",
                    label=task["label"],
                    video_id=task["video_id"],
                    force=False,
                    audio_cfg=cfg.get("audio", {})
                )
            )
            
        iterator = as_completed(futures)
        if has_tqdm:
            iterator = tqdm(iterator, total=len(tasks), desc="Extracting Audio")
            
        for future in iterator:
            try:
                res = future.result()
                if res["status"] in ["ok", "cached"]:
                    processed += 1
            except Exception as e:
                logger.debug(f"Failed task: {e}")

    logger.info(f"Finished extracting {processed} / {len(tasks)} videos.")

    # Generate splits
    import random
    import json
    
    random.seed(42)
    random.shuffle(tasks)
    
    split_idx = int(0.8 * len(tasks))
    train_tasks = tasks[:split_idx]
    val_tasks = tasks[split_idx:]
    
    splits_data = {
        "splits": {
            "train": [{"dataset": "FakeAVCeleb", "label": t["label"], "video_id": t["video_id"]} for t in train_tasks],
            "val": [{"dataset": "FakeAVCeleb", "label": t["label"], "video_id": t["video_id"]} for t in val_tasks],
            "test": []
        },
        "summary": {
            "train": len(train_tasks),
            "val": len(val_tasks),
            "test": 0
        }
    }
    
    splits_path = Path(r"d:\proj\deepfake last\deepfake_detector\processed\splits\audio_splits.json")
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
        
    logger.info(f"Generated audio_splits.json with {len(train_tasks)} train and {len(val_tasks)} val samples.")

if __name__ == "__main__":
    main()
