import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from pathlib import Path

def load_csv(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if v.strip():
                    data.setdefault(k, []).append(float(v))
    return data

def main():
    p13_dir = Path("results_archive/phase1_3_baseline")
    out_dir = Path("outputs")
    
    p13_img = load_csv(p13_dir / "snr_sweep_metrics.csv")
    p14_full = load_csv(out_dir / "snr_sweep_metrics.csv")
    
    # Hardcoded ablation data from logs
    snrs = [-5.0, -2.0, 0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]
    ablation_acc = [0.3984, 0.4775, 0.5195, 0.5419, 0.5849, 0.5859, 0.5957, 0.6054, 0.6093, 0.6113, 0.6083]
    
    plt.figure(figsize=(9, 6))
    
    if "semantic_meaning_acc" in p13_img and "snr_db" in p13_img:
        plt.plot(p13_img["snr_db"], p13_img["semantic_meaning_acc"], marker="o", linestyle="--", color="gray", label="Phase 1.3 Baseline (8 epochs, MSE only)")
        
    plt.plot(snrs, ablation_acc, marker="^", linestyle="-", color="orange", label="Ablation (8 epochs, Task-Aware Loss)")
    
    if "semantic_meaning_acc" in p14_full and "snr_db" in p14_full:
        plt.plot(p14_full["snr_db"], p14_full["semantic_meaning_acc"], marker="s", linestyle="-", color="blue", label="Phase 1.4 Full (30 epochs, Task-Aware Loss)")
        
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Meaning Accuracy", fontsize=12)
    plt.title("Ablation Study: Isolating the Task-Aware Loss", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    output_path = out_dir / "compare_image_meaning_acc_ablation.png"
    plt.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
