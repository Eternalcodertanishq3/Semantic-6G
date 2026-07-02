"""Zero-shot and robust channel robustness evaluation.

Evaluates the Semantic and Classical pipelines across AWGN, Rayleigh (block),
Rayleigh (fast), and CDL-approx channels without retraining the model.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from channel import make_channel
from models import ClassicalImagePipeline, SmallCifarClassifier
from train import build_model, select_device, seed_everything


CHANNEL_TYPES = ["awgn", "rayleigh_block", "rayleigh_fast", "cdl_approx"]
CHANNEL_LABELS = {
    "awgn": "AWGN",
    "rayleigh_block": "Rayleigh Block",
    "rayleigh_fast": "Rayleigh Fast",
    "cdl_approx": "CDL-Approx",
}
CHANNEL_COLORS = {
    "awgn": "tab:blue",
    "rayleigh_block": "tab:orange",
    "rayleigh_fast": "tab:red",
    "cdl_approx": "tab:purple",
}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_eval_loader(config: dict, fake_data: bool = False) -> DataLoader:
    transform = transforms.ToTensor()
    data_cfg = config["data"]
    if fake_data:
        dataset = datasets.FakeData(size=512, image_size=(3, 32, 32), num_classes=10, transform=transform)
    else:
        dataset = datasets.CIFAR10(root=data_cfg["root"], train=False, transform=transform, download=True)
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )


def tensor_to_images(batch: torch.Tensor) -> np.ndarray:
    return batch.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()


def image_metrics(reference: torch.Tensor, reconstruction: torch.Tensor) -> tuple[float, float]:
    ref_np = tensor_to_images(reference)
    rec_np = tensor_to_images(reconstruction)
    psnr_values = []
    ssim_values = []
    for ref, rec in zip(ref_np, rec_np):
        psnr_values.append(peak_signal_noise_ratio(ref, rec, data_range=1.0))
        ssim_values.append(structural_similarity(ref, rec, channel_axis=-1, data_range=1.0))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def batch_accuracy(classifier: torch.nn.Module | None, images: torch.Tensor, labels: torch.Tensor) -> float:
    if classifier is None:
        return float("nan")
    logits = classifier(images.clamp(0, 1))
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def load_classifier(config: dict, device: torch.device) -> SmallCifarClassifier | None:
    classifier_cfg = config.get("classifier", {})
    checkpoint_path = Path(classifier_cfg.get("checkpoint", "checkpoints/cifar_classifier.pt"))
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/cifar10_classifier.pt")
    if not checkpoint_path.exists():
        print(f"warning: no classifier checkpoint found; meaning accuracy will be NaN.")
        return None
    classifier = SmallCifarClassifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)
    print(f"loaded meaning classifier: {checkpoint_path}")
    return classifier


def evaluate_semantic_through_channel(
    model: torch.nn.Module,
    channel: torch.nn.Module,
    images: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Run semantic encoder -> external channel -> semantic decoder."""
    with torch.no_grad():
        symbols = model.encoder(images)
        snr_tensor = torch.full((images.shape[0],), snr_db, device=images.device)
        received = channel(symbols, snr_tensor)
        return model.decoder(received)


def evaluate_classical_through_channel(
    classical: ClassicalImagePipeline,
    channel: torch.nn.Module,
    images: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Run classical compress -> modulate -> external channel -> demodulate -> decode."""
    bits, codec_shape = classical.compress(images)
    coded = classical.encode_bits(bits)
    symbols = classical.modulate(coded)
    # Route through the specified channel instead of classical's own AWGN
    received = channel(symbols, snr_db)
    hard_bits = classical.demodulate(received)
    decoded = classical.decode_bits(hard_bits)
    return classical.decompress(decoded, codec_shape)


def run_sweep(
    model: torch.nn.Module,
    classical: ClassicalImagePipeline,
    loader: DataLoader,
    classifier: SmallCifarClassifier | None,
    config: dict,
    device: torch.device,
    max_batches: int,
    snrs: list[float],
    channel_types: list[str],
) -> dict[str, list[dict]]:
    """Run full SNR sweep for all channel types. Returns {channel_type: [row_per_snr]}."""
    symbol_power = float(config["channel"]["symbol_power"])
    results: dict[str, list[dict]] = {}

    for ch_type in channel_types:
        channel = make_channel(ch_type, symbol_power=symbol_power).to(device)
        rows = []
        print(f"\n=== Channel: {CHANNEL_LABELS[ch_type]} ===")

        for snr in snrs:
            sem_psnr, sem_ssim, sem_acc = [], [], []
            cls_psnr, cls_ssim, cls_acc = [], [], []
            orig_acc = []

            for batch_idx, (images, labels) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                images = images.to(device)
                labels = labels.to(device)

                # Semantic pipeline through external channel
                sem_recon = evaluate_semantic_through_channel(model, channel, images, snr)
                psnr, ssim = image_metrics(images, sem_recon)
                sem_psnr.append(psnr)
                sem_ssim.append(ssim)
                sem_acc.append(batch_accuracy(classifier, sem_recon, labels))

                # Classical pipeline through external channel
                cls_recon = evaluate_classical_through_channel(classical, channel, images, snr)
                psnr, ssim = image_metrics(images, cls_recon)
                cls_psnr.append(psnr)
                cls_ssim.append(ssim)
                cls_acc.append(batch_accuracy(classifier, cls_recon, labels))
                orig_acc.append(batch_accuracy(classifier, images, labels))

            row = {
                "channel": ch_type,
                "snr_db": snr,
                "semantic_psnr": float(np.nanmean(sem_psnr)),
                "semantic_ssim": float(np.nanmean(sem_ssim)),
                "semantic_meaning_acc": float(np.nanmean(sem_acc)),
                "classical_psnr": float(np.nanmean(cls_psnr)),
                "classical_ssim": float(np.nanmean(cls_ssim)),
                "classical_meaning_acc": float(np.nanmean(cls_acc)),
                "original_meaning_acc": float(np.nanmean(orig_acc)),
            }
            rows.append(row)
            print(f"  SNR={snr:+5.1f}dB  sem_psnr={row['semantic_psnr']:.2f}  "
                  f"sem_acc={row['semantic_meaning_acc']:.3f}  "
                  f"cls_psnr={row['classical_psnr']:.2f}  "
                  f"cls_acc={row['classical_meaning_acc']:.3f}")

        results[ch_type] = rows

    return results


def write_csv(path: Path, results: dict[str, list[dict]]) -> None:
    all_rows = []
    for rows in results.values():
        all_rows.extend(rows)
    if not all_rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)


def plot_metric_by_channel(
    results: dict[str, list[dict]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    pipeline: str = "semantic",
):
    """Plot a metric vs SNR with one curve per channel type."""
    plt.figure(figsize=(9, 6))
    for ch_type, rows in results.items():
        snrs = [r["snr_db"] for r in rows]
        values = [r[f"{pipeline}_{metric_key}"] for r in rows]
        plt.plot(snrs, values, marker="o", color=CHANNEL_COLORS[ch_type],
                 label=f"{CHANNEL_LABELS[ch_type]}")
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_semantic_vs_classical_by_channel(
    results: dict[str, list[dict]],
    output_path: Path,
):
    """Plot meaning accuracy: semantic (solid) vs classical (dashed) for each channel."""
    plt.figure(figsize=(10, 6))
    for ch_type, rows in results.items():
        snrs = [r["snr_db"] for r in rows]
        sem_acc = [r["semantic_meaning_acc"] for r in rows]
        cls_acc = [r["classical_meaning_acc"] for r in rows]
        color = CHANNEL_COLORS[ch_type]
        label = CHANNEL_LABELS[ch_type]
        plt.plot(snrs, sem_acc, marker="o", color=color, label=f"Semantic ({label})")
        plt.plot(snrs, cls_acc, marker="x", linestyle="--", color=color, alpha=0.6,
                 label=f"Classical ({label})")
    # Original ceiling
    if results:
        first_rows = list(results.values())[0]
        snrs = [r["snr_db"] for r in first_rows]
        orig = [r["original_meaning_acc"] for r in first_rows]
        if not all(np.isnan(orig)):
            plt.plot(snrs, orig, color="black", linestyle=":", label="Classifier ceiling")
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Meaning Accuracy", fontsize=12)
    plt.title("Meaning Accuracy: Semantic vs Classical across Channel Types", fontsize=13, fontweight="bold")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    loader = build_eval_loader(config, fake_data=args.fake_data)
    model = build_model(config, device)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(config["semantic"]["checkpoint"])
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"loaded semantic checkpoint: {checkpoint_path}")
    else:
        print(f"warning: no checkpoint at {checkpoint_path}; using random weights.")
    model.eval()

    classical_cfg = dict(config["classical"])
    classical_cfg["symbol_power"] = float(config["channel"]["symbol_power"])
    classical = ClassicalImagePipeline.from_dict(classical_cfg, device=device)
    classifier = load_classifier(config, device)

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_batches = args.max_batches if args.max_batches is not None else int(config["evaluation"]["max_batches"])
    snrs = [float(x) for x in config["channel"]["eval_snr_db"]]
    if args.snr_count is not None:
        snrs = snrs[:args.snr_count]

    channel_types = args.channels.split(",") if args.channels else CHANNEL_TYPES

    results = run_sweep(model, classical, loader, classifier, config, device, max_batches, snrs, channel_types)

    # Hard AWGN regression assertion
    if "awgn" in results and not args.fake_data:
        awgn_rows = results["awgn"]
        high_snr_rows = [r for r in awgn_rows if r["snr_db"] >= 20.0]
        if high_snr_rows:
            awgn_acc_20db = high_snr_rows[-1]["semantic_meaning_acc"]
            assert abs(awgn_acc_20db - 0.716) < 0.02, (
                f"AWGN regression: expected ~71.6%, got {awgn_acc_20db:.1%}. "
                f"Something changed in channel routing or normalization."
            )
            print(f"\n[OK] AWGN regression check passed: {awgn_acc_20db:.1%} (expected ~71.6%)")

    # Save CSV
    prefix = args.output_prefix
    write_csv(output_dir / f"channel_robustness_{prefix}.csv", results)

    # Plot: Semantic PSNR by channel
    plot_metric_by_channel(
        results, "psnr",
        f"Semantic PSNR vs SNR ({prefix})",
        "PSNR (dB)",
        output_dir / f"channel_robustness_{prefix}_psnr.png",
    )
    # Plot: Semantic Meaning Accuracy by channel
    plot_metric_by_channel(
        results, "meaning_acc",
        f"Semantic Meaning Accuracy vs SNR ({prefix})",
        "Meaning Accuracy",
        output_dir / f"channel_robustness_{prefix}_meaning_acc.png",
    )
    # Plot: Semantic vs Classical meaning accuracy across channels
    plot_semantic_vs_classical_by_channel(
        results,
        output_dir / f"channel_robustness_{prefix}_sem_vs_cls.png",
    )

    print(f"\nResults saved to {output_dir}/channel_robustness_{prefix}_*.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate channel robustness across AWGN, Rayleigh, and CDL-approx.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to semantic checkpoint (default: from config).")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--snr-count", type=int, default=None)
    parser.add_argument("--channels", default=None,
                        help="Comma-separated list of channel types (default: all four).")
    parser.add_argument("--output-prefix", default="zeroshot",
                        help="Prefix for output filenames (default: 'zeroshot').")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
