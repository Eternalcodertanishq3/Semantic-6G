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

from models import ClassicalImagePipeline, SmallCifarClassifier
from train import build_model, select_device, seed_everything


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


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return True


def load_classifier(config: dict, device: torch.device, allow_random: bool = False) -> SmallCifarClassifier | None:
    classifier_cfg = config.get("classifier", {})
    checkpoint_path = Path(classifier_cfg.get("checkpoint", "checkpoints/cifar_classifier.pt"))
    classifier = SmallCifarClassifier().to(device)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classifier.load_state_dict(checkpoint["model_state"])
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad_(False)
        print(f"loaded meaning classifier: {checkpoint_path}")
        return classifier
    if allow_random:
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad_(False)
        print("warning: using randomly initialized classifier; meaning accuracy is only for smoke testing.")
        return classifier
    print(f"warning: no classifier checkpoint at {checkpoint_path}; meaning accuracy columns will be NaN.")
    return None


def assert_link_budget(model: torch.nn.Module, classical: ClassicalImagePipeline, loader: DataLoader, device: torch.device, symbol_power: float) -> None:
    semantic_symbols = int(model.encoder.num_symbols)
    classical_symbols = int(classical.symbols_per_image)
    if semantic_symbols != classical_symbols:
        raise AssertionError(f"channel-use mismatch: semantic={semantic_symbols}, classical={classical_symbols}")

    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        semantic_power = float(model.encoder(images).pow(2).sum(dim=-1).mean().item())
    classical_power = classical.average_symbol_power()
    print(
        "link budget: "
        f"semantic_symbols={semantic_symbols}, classical_symbols={classical_symbols}, "
        f"semantic_power={semantic_power:.6f}, classical_power={classical_power:.6f}, "
        f"target_power={symbol_power:.6f}, classical_source_bits={classical.source_bit_budget}, "
        f"classical_coded_bits={classical.encoded_bit_count}/{classical.coded_bit_count}"
    )
    if not np.isclose(semantic_power, classical_power, rtol=1e-3, atol=1e-3):
        raise AssertionError(f"average-power mismatch: semantic={semantic_power}, classical={classical_power}")


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    loader = build_eval_loader(config, fake_data=args.fake_data)
    model = build_model(config, device)
    checkpoint_path = Path(config["semantic"]["checkpoint"])
    loaded = load_checkpoint_if_available(model, checkpoint_path, device)
    if not loaded:
        print(f"warning: no checkpoint found at {checkpoint_path}; semantic curve uses random weights.")
    model.eval()

    classical_cfg = dict(config["classical"])
    classical_cfg["symbol_power"] = float(config["channel"]["symbol_power"])
    classical = ClassicalImagePipeline.from_dict(classical_cfg, device=device)
    assert_link_budget(model, classical, loader, device, float(config["channel"]["symbol_power"]))
    classifier = load_classifier(config, device, allow_random=args.allow_random_classifier)

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_batches = args.max_batches if args.max_batches is not None else int(config["evaluation"]["max_batches"])
    snrs = [float(x) for x in config["channel"]["eval_snr_db"]]
    if args.snr_count is not None:
        snrs = snrs[: args.snr_count]
    rows = []

    with torch.no_grad():
        for snr in snrs:
            semantic_psnr = []
            semantic_ssim = []
            semantic_acc = []
            classical_psnr = []
            classical_ssim = []
            classical_acc = []
            original_acc = []
            for batch_index, (images, labels) in enumerate(loader):
                if batch_index >= max_batches:
                    break
                images = images.to(device)
                labels = labels.to(device)
                semantic_recon = model(images, torch.full((images.shape[0],), snr, device=device))
                classical_recon = classical.transmit(images, snr)

                psnr, ssim = image_metrics(images, semantic_recon)
                semantic_psnr.append(psnr)
                semantic_ssim.append(ssim)
                semantic_acc.append(batch_accuracy(classifier, semantic_recon, labels))

                psnr, ssim = image_metrics(images, classical_recon)
                classical_psnr.append(psnr)
                classical_ssim.append(ssim)
                classical_acc.append(batch_accuracy(classifier, classical_recon, labels))
                original_acc.append(batch_accuracy(classifier, images, labels))

            row = {
                "snr_db": snr,
                "semantic_psnr": float(np.nanmean(semantic_psnr)),
                "semantic_ssim": float(np.nanmean(semantic_ssim)),
                "semantic_meaning_acc": float(np.nanmean(semantic_acc)),
                "classical_psnr": float(np.nanmean(classical_psnr)),
                "classical_ssim": float(np.nanmean(classical_ssim)),
                "classical_meaning_acc": float(np.nanmean(classical_acc)),
                "original_meaning_acc": float(np.nanmean(original_acc)),
            }
            rows.append(row)
            print(row)

    write_csv(output_dir / "snr_sweep_metrics.csv", rows)
    plot_metric(rows, "psnr", output_dir / "psnr_vs_snr.png")
    plot_metric(rows, "ssim", output_dir / "ssim_vs_snr.png")
    plot_meaning_accuracy(rows, output_dir / "meaning_accuracy_vs_snr.png")


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict], metric: str, path: Path) -> None:
    snr = [r["snr_db"] for r in rows]
    semantic = [r[f"semantic_{metric}"] for r in rows]
    classical = [r[f"classical_{metric}"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(snr, classical, marker="o", label="Classical conv+QAM")
    plt.plot(snr, semantic, marker="s", label="Semantic JSCC")
    plt.xlabel("SNR (dB)")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs. SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    collapse = estimate_collapse_snr(snr, classical)
    if collapse is not None:
        plt.axvline(collapse, color="tab:red", linestyle="--", alpha=0.6)
        plt.annotate("classical cliff", xy=(collapse, min(classical)), xytext=(collapse + 1, min(classical) + 1))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_meaning_accuracy(rows: list[dict], path: Path) -> None:
    snr = [r["snr_db"] for r in rows]
    semantic = [r["semantic_meaning_acc"] for r in rows]
    classical = [r["classical_meaning_acc"] for r in rows]
    original = [r["original_meaning_acc"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(snr, classical, marker="o", label="Classical conv+QAM")
    plt.plot(snr, semantic, marker="s", label="Semantic JSCC")
    if not all(np.isnan(original)):
        plt.plot(snr, original, color="black", linestyle="--", label="Original classifier ceiling")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Classifier Accuracy")
    plt.ylim(0.0, 1.02)
    plt.title("Meaning Accuracy vs. SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def estimate_collapse_snr(snrs: list[float], values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    best = max(values)
    threshold = best - 4.0
    candidates = [snr for snr, value in zip(snrs, values) if value < threshold]
    return max(candidates) if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SNR sweep evaluation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--snr-count", type=int, default=None, help="Use only the first N configured SNR points for quick checks.")
    parser.add_argument("--allow-random-classifier", action="store_true", help="Smoke-test classifier metrics without a trained checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
