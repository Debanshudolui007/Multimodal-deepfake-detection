"""One-shot training status snapshot — run once to see current progress."""
import re
from pathlib import Path

log = Path("deepfake_detector") / "logs" / "main.log"
text = log.read_text(encoding="utf-8", errors="replace")

epoch_re = re.compile(
    r"\[(\w+)\] Epoch\s+(\d+)/(\d+) \| train_loss=([\d.]+) train_auc=([\d.]+) "
    r"\| val_loss=([\d.]+) val_auc=([\d.]+) val_f1=([\d.]+) eer=([\d.]+) lr=([\deE.+-]+)(.*)"
)
batch_re = re.compile(
    r"Epoch\s+(\d+)/(\d+) \[(\w+)\].*?(\d+)/(\d+).*?loss=([\d.]+)"
)

epochs  = [m.groups() for m in epoch_re.finditer(text)]
batches = list(batch_re.finditer(text))

SEP = "=" * 66
DIV = "-" * 66

print()
print(SEP)
print("  SPATIAL TRAINING -- LIVE SNAPSHOT")
print(SEP)

if batches:
    m  = batches[-1]
    ep, maxep = m.group(1), m.group(2)
    phase     = m.group(3)
    b, nb     = int(m.group(4)), int(m.group(5))
    loss      = m.group(6)
    pct       = b / max(nb, 1) * 100
    filled    = int(pct / 2.5)
    bar       = "#" * filled + "-" * (40 - filled)
    print(f"  Epoch {ep}/{maxep}  [{phase.upper():<5}]")
    print(f"  [{bar}] {pct:5.1f}%")
    print(f"  Batch {b:>5}/{nb}  |  Running loss: {loss}")
else:
    print("  Waiting for first batch...")

print()

if epochs:
    hdr = f"  {'Ep':>3}  {'T-Loss':>7}  {'T-AUC':>7}  {'V-Loss':>7}  {'V-AUC':>7}  {'F1':>6}  {'EER':>6}"
    print(hdr)
    print("  " + DIV)
    for e in epochs:
        tag = " <-- BEST" if "[BEST]" in str(e[-1]) else ""
        print(
            f"  {e[1]:>3}  {float(e[3]):>7.4f}  {float(e[4]):>7.4f}  "
            f"{float(e[5]):>7.4f}  {float(e[6]):>7.4f}  {float(e[7]):>6.4f}  "
            f"{float(e[8]):>6.4f}{tag}"
        )
    best = max(epochs, key=lambda x: float(x[6]))
    print()
    print(f"  Best: Epoch {best[1]}  |  Val AUC={float(best[6]):.4f}  F1={float(best[7]):.4f}")
    print(f"  Completed: {len(epochs)} / {epochs[-1][2]} epochs")
else:
    print("  No completed epochs yet (Epoch 1 in progress).")

print(SEP)
print()
