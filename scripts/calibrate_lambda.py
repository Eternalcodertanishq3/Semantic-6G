"""Calibrate λ: log unweighted MSE and classifier CE for first 50 batches."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from train import build_model, build_loaders, random_snr, select_device, seed_everything
from evaluate import load_classifier


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--batches", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    train_loader, _ = build_loaders(config, fake_data=args.fake_data)
    model = build_model(config, device)

    # Load checkpoint if exists (we want to measure on the current trained model)
    ckpt_path = Path(config["semantic"]["checkpoint"])
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No checkpoint found, using random weights")

    classifier = load_classifier(config, device)
    if classifier is None:
        raise RuntimeError("Need a trained classifier for calibration")

    model.train()
    mse_vals = []
    ce_vals = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= args.batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        snr = random_snr(images.shape[0], config, device)

        with torch.no_grad():
            recon = model(images, snr)
            mse = F.mse_loss(recon, images).item()
            ce = F.cross_entropy(classifier(recon.clamp(0, 1)), labels).item()
            mse_vals.append(mse)
            ce_vals.append(ce)
            if batch_idx < 10 or batch_idx % 10 == 0:
                print(f"batch={batch_idx:3d}  mse={mse:.6f}  ce={ce:.4f}  ratio(ce/mse)={ce/mse:.1f}")

    mean_mse = np.mean(mse_vals)
    mean_ce = np.mean(ce_vals)
    ratio = mean_ce / mean_mse

    print(f"\n{'='*60}")
    print(f"Mean MSE:  {mean_mse:.6f}")
    print(f"Mean CE:   {mean_ce:.4f}")
    print(f"Ratio CE/MSE: {ratio:.1f}x")
    print(f"\nTo make weighted CE = 10% of MSE:  lambda = {0.10 / ratio:.5f}")
    print(f"To make weighted CE = 15% of MSE:  lambda = {0.15 / ratio:.5f}")
    print(f"To make weighted CE = 20% of MSE:  lambda = {0.20 / ratio:.5f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
