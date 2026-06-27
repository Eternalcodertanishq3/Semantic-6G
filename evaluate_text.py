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
from torch.utils.data import DataLoader

from models import CharVocabulary
from train_text import build_loaders, build_model, seed_everything, select_device


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return True


def token_accuracy(reference: torch.Tensor, logits: torch.Tensor, pad_id: int) -> float:
    predictions = logits.argmax(dim=-1)
    mask = reference != pad_id
    correct = (predictions[mask] == reference[mask]).float().sum()
    total = mask.float().sum()
    return float(correct / total.clamp_min(1.0))


def exact_match_accuracy(reference: torch.Tensor, logits: torch.Tensor, pad_id: int) -> float:
    predictions = logits.argmax(dim=-1)
    mask = reference != pad_id
    matches = []
    for ref_row, pred_row, mask_row in zip(reference, predictions, mask):
        r = ref_row[mask_row]
        p = pred_row[mask_row]
        matches.append(float(torch.equal(r, p)))
    return float(np.mean(matches))


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    _, loader = build_loaders(config, fake_data=args.fake_data)
    model = build_model(config, device)
    checkpoint_path = Path(config["text"]["checkpoint"])
    loaded = load_checkpoint_if_available(model, checkpoint_path, device)
    if not loaded:
        print(f"warning: no checkpoint found at {checkpoint_path}; semantic curve uses random weights.")
    model.eval()

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_batches = args.max_batches if args.max_batches is not None else int(config["evaluation"]["max_batches"])
    snrs = [float(x) for x in config["channel"]["eval_snr_db"]]
    if args.snr_count is not None:
        snrs = snrs[: args.snr_count]
    rows = []

    with torch.no_grad():
        for snr in snrs:
            semantic_token_acc = []
            semantic_exact_acc = []

            for batch_index, tokens in enumerate(loader):
                if batch_index >= max_batches:
                    break
                tokens = tokens.to(device)
                logits = model(tokens, snr)

                semantic_token_acc.append(token_accuracy(tokens, logits, CharVocabulary.pad_id))
                semantic_exact_acc.append(exact_match_accuracy(tokens, logits, CharVocabulary.pad_id))

            row = {
                "snr_db": snr,
                "semantic_token_acc": float(np.nanmean(semantic_token_acc)),
                "semantic_exact_acc": float(np.nanmean(semantic_exact_acc)),
            }
            rows.append(row)
            print(row)

    write_csv(output_dir / "text_snr_sweep_metrics.csv", rows)
    plot_metric(rows, "token_acc", output_dir / "text_token_acc_vs_snr.png")


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict], metric: str, path: Path) -> None:
    snr = [r["snr_db"] for r in rows]
    semantic = [r[f"semantic_{metric}"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(snr, semantic, marker="s", label="Semantic JSCC")
    plt.xlabel("SNR (dB)")
    plt.ylabel(metric.replace("_", " ").upper())
    plt.title(f"{metric.upper()} vs. SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text SNR sweep evaluation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--snr-count", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
