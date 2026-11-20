import argparse
import json
import os
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

SPEED_STATES = {}

def _get_live_jsons() -> list[Path]:
    json_files = list(_ROOT.glob("deepfake_detector/**/live_batch.json")) + list(_ROOT.glob("deepfake_detector/**/live_eval.json"))
    now = time.time()
    # Only keep files modified in the last 15 minutes
    active_files = [p for p in json_files if (now - p.stat().st_mtime) < 900]
    # Sort by run name (which is the parent's parent directory name)
    return sorted(active_files, key=lambda p: p.parent.parent.name)

def _parse_states() -> list[dict]:
    states = []
    live_jsons = _get_live_jsons()
    for live_json in live_jsons:
        try:
            with open(live_json, "r", encoding="utf-8") as f:
                batch_data = json.load(f)
            states.append({"file": live_json, "batch": batch_data})
        except Exception:
            pass
    return states

def _bar(frac: float, width: int = 45) -> str:
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac*100:5.1f}%"

def _render(states: list[dict]):
    SEP = "=" * 68
    os.system("cls" if os.name == "nt" else "clear")

    print(SEP)
    print("  DEEPFAKE DETECTOR  --  LIVE PROGRESS MONITOR")
    print(SEP)
    print()

    if not states:
        print("  [waiting for updates...]")
        print()
    else:
        for state in states:
            live_json = state["file"]
            batch = state["batch"]
            run_name = live_json.parent.parent.name.upper()
            
            print(f"  --- RUN: {run_name} ---")
            
            if not batch:
                print("  [No data]")
                print()
                continue
                
            frac = batch.get("batch", 0) / max(batch.get("n_batches", 1), 1)
            current_b = batch.get("batch", 0)
            t = time.time()
            
            f_key = str(live_json)
            if f_key not in SPEED_STATES:
                SPEED_STATES[f_key] = {"last_batch": current_b, "last_time": t, "speed": 0.0}
            else:
                st = SPEED_STATES[f_key]
                if current_b < st["last_batch"]:
                    st["last_batch"] = current_b
                    st["last_time"] = t
                    st["speed"] = 0.0
                else:
                    dt = t - st["last_time"]
                    db = current_b - st["last_batch"]
                    if dt >= 5.0 and db > 0:
                        st["speed"] = db / dt
                        st["last_batch"] = current_b
                        st["last_time"] = t
                        
            speed = SPEED_STATES[f_key]["speed"]
            
            eta_str = "Calculating..."
            if speed > 0:
                rem_sec = (batch.get("n_batches", 0) - current_b) / speed
                eta_str = f"{int(rem_sec // 3600)}h {int((rem_sec % 3600) // 60)}m {int(rem_sec % 60)}s"
                
            print(f"  Stage: [{batch.get('phase', '?').upper()}]   |   Speed: {speed:.1f} items/sec   |   ETA: {eta_str}")
            print(f"  {_bar(frac)}")
            print(f"  Processed {current_b:>5} / {batch.get('n_batches', '?')} items")
            if "loss" in batch:
                print(f"  Live Loss: {batch['loss']:.4f}")
            if "acc" in batch:
                print(f"  Live Acc:  {batch['acc']*100:.2f}%")
            
            v_curr = batch.get("video_current")
            v_total = batch.get("video_total")
            if v_curr is not None and v_total is not None:
                print(f"  Total Videos: {v_curr:>5} / {v_total} videos  ({v_total - v_curr} left)")
                
            print()
            
            # History
            summary_files = list(live_json.parent.glob("*_summary.json"))
            if summary_files:
                try:
                    with open(summary_files[0], "r", encoding="utf-8") as f:
                        data = json.load(f)
                    history_list = data.get("history", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    if history_list:
                        print("  [Completed Epochs History]")
                        print("  Ep |   Time   | Tr.Loss | Tr.AUC | Val.Loss | Val.AUC")
                        print("  " + "-" * 60)
                        for ep in history_list:
                            e_id = ep.get("epoch", "?")
                            t_sec = ep.get("elapsed_sec", 0.0)
                            t_loss = ep.get("train_loss", 0.0)
                            t_auc = ep.get("train_auc", 0.0)
                            v_loss = ep.get("val_loss", 0.0)
                            v_auc = ep.get("val_auc", 0.0)
                            print(f"  {e_id:<2} | {t_sec:>7.1f}s | {t_loss:>7.4f} | {t_auc:>6.4f} | {v_loss:>8.4f} | {v_auc:>7.4f}")
                        print()
                except Exception:
                    pass
            print(SEP)
            print()

    print(f"  Refreshes every 5s. Press Ctrl+C to exit.")
    print(SEP)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", type=int, default=5)
    args = parser.parse_args()

    try:
        while True:
            states = _parse_states()
            _render(states)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")

if __name__ == "__main__":
    main()
