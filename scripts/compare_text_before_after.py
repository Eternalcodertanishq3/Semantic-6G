import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(path: Path) -> dict:
    snrs = []
    accs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snrs.append(float(row["snr_db"]))
            accs.append(float(row["semantic_token_acc"]))
    return {"snrs": snrs, "accs": accs}

def main():
    root = Path(__file__).parent.parent
    baseline_path = root / "results_archive" / "phase1_2_baseline" / "text_snr_sweep_metrics.csv"
    new_path = root / "outputs" / "text_snr_sweep_metrics.csv"
    
    baseline = load_metrics(baseline_path)
    new = load_metrics(new_path)
    
    plt.figure(figsize=(8, 5))
    plt.plot(baseline["snrs"], baseline["accs"], marker="o", label="Phase 1.2 (Baseline)", linestyle="--")
    plt.plot(new["snrs"], new["accs"], marker="s", label="Phase 1.3 (After Fix)")
    
    plt.xlabel("SNR (dB)")
    plt.ylabel("Token Accuracy")
    plt.title("Text Semantic Codec: Token Accuracy vs. SNR (Before/After)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    out_path = root / "outputs" / "text_token_acc_before_after.png"
    plt.savefig(out_path, dpi=160)
    print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    main()
