"""Generate comparison plots for Phase 1.4 vs Phase 1.3."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def plot_comparison(
    phase13_data: dict[str, list[float]],
    phase14_data: dict[str, list[float]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    p14_label: str = "Phase 1.4 (Task-Aware)",
):
    plt.figure(figsize=(8, 5))
    if metric_key in phase13_data and "snr_db" in phase13_data:
        plt.plot(phase13_data["snr_db"], phase13_data[metric_key], marker="o", linestyle="--", label="Phase 1.3 Baseline")
    if metric_key in phase14_data and "snr_db" in phase14_data:
        plt.plot(phase14_data["snr_db"], phase14_data[metric_key], marker="s", label=p14_label)
    
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    out_dir = Path("outputs")
    p13_dir = Path("results_archive/phase1_3_baseline")
    
    # Image comparisons
    p13_img = load_csv(p13_dir / "snr_sweep_metrics.csv")
    p14_img = load_csv(out_dir / "snr_sweep_metrics.csv")
    
    if p13_img and p14_img:
        plot_comparison(
            p13_img, p14_img, "semantic_meaning_acc",
            "Image Meaning Accuracy vs SNR (Phase 1.3 vs 1.4)",
            "Meaning Accuracy",
            out_dir / "compare_image_meaning_acc.png"
        )
        plot_comparison(
            p13_img, p14_img, "semantic_psnr",
            "Image PSNR vs SNR (Phase 1.3 vs 1.4)",
            "PSNR (dB)",
            out_dir / "compare_image_psnr.png"
        )

    # Text comparisons
    p13_txt = load_csv(p13_dir / "text_snr_sweep_metrics.csv")
    p14_txt = load_csv(out_dir / "text_snr_sweep_metrics.csv")
    
    if p13_txt and p14_txt:
        plot_comparison(
            p13_txt, p14_txt, "semantic_token_acc",
            "Text Token Accuracy vs SNR (Phase 1.3 vs 1.4)",
            "Token Accuracy",
            out_dir / "compare_text_token_acc.png",
            p14_label="Phase 1.4 (Extended Training)"
        )

if __name__ == "__main__":
    main()
