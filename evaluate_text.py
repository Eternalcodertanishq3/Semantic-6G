from __future__ import annotations

import argparse
import csv
import math
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
from channel.channel_models import snr_db_to_noise_variance


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


def levenshtein_distance(s1: list[int], s2: list[int]) -> int:
    """Compute Levenshtein edit distance between two integer sequences."""
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i in range(1, len(s1) + 1):
        curr = [i] + [0] * len(s2)
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(s2)]


def normalized_edit_distance(reference: torch.Tensor, logits: torch.Tensor, pad_id: int) -> float:
    """Compute mean normalized Levenshtein edit distance across the batch."""
    predictions = logits.argmax(dim=-1)
    distances = []
    for ref_row, pred_row in zip(reference, predictions):
        # Strip padding
        ref_tokens = [t.item() for t in ref_row if t.item() != pad_id]
        pred_tokens = [t.item() for t in pred_row if t.item() != pad_id]
        if len(ref_tokens) == 0:
            continue
        dist = levenshtein_distance(ref_tokens, pred_tokens)
        # Normalize by reference length
        distances.append(dist / len(ref_tokens))
    return float(np.mean(distances)) if distances else float("nan")


def char_bleu_score(reference: torch.Tensor, logits: torch.Tensor, pad_id: int, max_n: int = 4) -> float:
    """Compute character-level BLEU score across the batch (no external dependencies)."""
    predictions = logits.argmax(dim=-1)
    scores = []
    for ref_row, pred_row in zip(reference, predictions):
        ref_tokens = [t.item() for t in ref_row if t.item() != pad_id]
        pred_tokens = [t.item() for t in pred_row if t.item() != pad_id]
        if len(ref_tokens) == 0 or len(pred_tokens) == 0:
            scores.append(0.0)
            continue

        # Compute n-gram precisions
        log_precisions = []
        for n in range(1, min(max_n, len(pred_tokens)) + 1):
            ref_ngrams: dict[tuple, int] = {}
            for i in range(len(ref_tokens) - n + 1):
                ng = tuple(ref_tokens[i:i + n])
                ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

            pred_ngrams: dict[tuple, int] = {}
            for i in range(len(pred_tokens) - n + 1):
                ng = tuple(pred_tokens[i:i + n])
                pred_ngrams[ng] = pred_ngrams.get(ng, 0) + 1

            clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in pred_ngrams.items())
            total = max(1, sum(pred_ngrams.values()))
            precision = clipped / total
            if precision == 0:
                log_precisions.append(float("-inf"))
            else:
                log_precisions.append(math.log(precision))

        if not log_precisions or all(lp == float("-inf") for lp in log_precisions):
            scores.append(0.0)
            continue

        # Geometric mean of precisions
        avg_log_precision = sum(log_precisions) / len(log_precisions)

        # Brevity penalty
        bp = min(0.0, 1.0 - len(ref_tokens) / len(pred_tokens))
        bleu = math.exp(avg_log_precision + bp)
        scores.append(bleu)

    return float(np.mean(scores)) if scores else float("nan")


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
            edit_distances = []
            bleu_scores = []

            noise_var = snr_db_to_noise_variance(snr, float(config["channel"]["symbol_power"]))
            noise_std = math.sqrt(noise_var / 2.0)
            print(f"--- Evaluated SNR: {snr:5.1f} dB, applied noise std dev: {noise_std:.4f} ---")

            for batch_index, tokens in enumerate(loader):
                if batch_index >= max_batches:
                    break
                tokens = tokens.to(device)
                logits = model(tokens, snr)

                semantic_token_acc.append(token_accuracy(tokens, logits, CharVocabulary.pad_id))
                semantic_exact_acc.append(exact_match_accuracy(tokens, logits, CharVocabulary.pad_id))
                edit_distances.append(normalized_edit_distance(tokens, logits, CharVocabulary.pad_id))
                bleu_scores.append(char_bleu_score(tokens, logits, CharVocabulary.pad_id))

            row = {
                "snr_db": snr,
                "semantic_token_acc": float(np.nanmean(semantic_token_acc)),
                "semantic_exact_acc": float(np.nanmean(semantic_exact_acc)),
                "edit_distance": float(np.nanmean(edit_distances)),
                "bleu_score": float(np.nanmean(bleu_scores)),
            }
            rows.append(row)
            print(row)

    write_csv(output_dir / "text_snr_sweep_metrics.csv", rows)
    plot_metric(rows, "token_acc", output_dir / "text_token_acc_vs_snr.png")
    plot_metric(rows, "edit_distance", output_dir / "text_edit_distance_vs_snr.png", invert=True)
    plot_metric(rows, "bleu_score", output_dir / "text_bleu_vs_snr.png")


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict], metric: str, path: Path, invert: bool = False) -> None:
    snr = [r["snr_db"] for r in rows]
    semantic = [r[f"semantic_{metric}"] if f"semantic_{metric}" in r else r[metric] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(snr, semantic, marker="s", label="Semantic JSCC")
    plt.xlabel("SNR (dB)")
    label = metric.replace("_", " ").title()
    if invert:
        label += " (lower is better)"
    plt.ylabel(label)
    plt.title(f"{label} vs. SNR")
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
