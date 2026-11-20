import os
import argparse
from pathlib import Path

def shrink_dataset(faces_root: Path, dry_run: bool = True):
    total_files_deleted = 0
    total_bytes_saved = 0
    
    # Iterate through all datasets and labels
    for dataset_dir in faces_root.iterdir():
        if not dataset_dir.is_dir(): continue
        for label_dir in dataset_dir.iterdir():
            if not label_dir.is_dir(): continue
            for video_dir in label_dir.iterdir():
                if not video_dir.is_dir(): continue
                
                # Find all png frames and sort chronologically
                png_files = sorted(list(video_dir.glob("face_*.png")))
                
                # Delete every alternating frame (index 1, 3, 5, etc.)
                for i in range(1, len(png_files), 2):
                    png_path = png_files[i]
                    freq_name = png_path.stem + "_freq.npy"
                    freq_path = png_path.parent / freq_name
                    
                    # Track sizes
                    if png_path.exists():
                        total_bytes_saved += png_path.stat().st_size
                        if not dry_run:
                            png_path.unlink()
                        total_files_deleted += 1
                        
                    if freq_path.exists():
                        total_bytes_saved += freq_path.stat().st_size
                        if not dry_run:
                            freq_path.unlink()
                        total_files_deleted += 1

    gb_saved = total_bytes_saved / (1024**3)
    mode = "DRY RUN" if dry_run else "EXECUTION"
    print(f"--- {mode} RESULTS ---")
    print(f"Total files targeted for deletion: {total_files_deleted}")
    print(f"Total space to be saved: {gb_saved:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually delete the files")
    args = parser.parse_args()
    
    faces_dir = Path("deepfake_detector/processed/faces")
    if not faces_dir.exists():
        print(f"Directory not found: {faces_dir}")
    else:
        shrink_dataset(faces_dir, dry_run=not args.execute)
