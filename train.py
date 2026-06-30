from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from channel import make_channel
from models import SemanticDecoder, SemanticEncoder, SmallCifarClassifier


class SemanticAutoencoder(nn.Module):
    def __init__(self, encoder: SemanticEncoder, channel: nn.Module, decoder: SemanticDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.channel = channel
        self.decoder = decoder

    def forward(self, images: torch.Tensor, snr_db: torch.Tensor | float) -> torch.Tensor:
        symbols = self.encoder(images)
        received = self.channel(symbols, snr_db)
        return self.decoder(received)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_loaders(config: dict, fake_data: bool = False) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    transform = transforms.ToTensor()
    if fake_data:
        dataset = datasets.FakeData(size=2048, image_size=(3, 32, 32), num_classes=10, transform=transform)
    else:
        dataset = datasets.CIFAR10(root=data_cfg["root"], train=True, transform=transform, download=True)

    val_size = int(data_cfg.get("val_size", 1000))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    loader_kwargs = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    )


def build_model(config: dict, device: torch.device) -> SemanticAutoencoder:
    channel_cfg = config["channel"]
    semantic_cfg = config["semantic"]
    encoder = SemanticEncoder(
        num_symbols=int(semantic_cfg["num_symbols"]),
        latent_channels=int(semantic_cfg["latent_channels"]),
        symbol_power=float(channel_cfg["symbol_power"]),
    )
    channel = make_channel(channel_cfg["type"], symbol_power=float(channel_cfg["symbol_power"]))
    decoder = SemanticDecoder(
        num_symbols=int(semantic_cfg["num_symbols"]),
        latent_channels=int(semantic_cfg["latent_channels"]),
    )
    return SemanticAutoencoder(encoder, channel, decoder).to(device)


def load_frozen_classifier(config: dict, device: torch.device) -> SmallCifarClassifier | None:
    """Load the frozen meaning classifier for task-aware loss."""
    classifier_cfg = config.get("classifier", {})
    checkpoint_path = Path(classifier_cfg.get("checkpoint", "checkpoints/cifar_classifier.pt"))
    if not checkpoint_path.exists():
        # Try alternate path
        checkpoint_path = Path("checkpoints/cifar10_classifier.pt")
    if not checkpoint_path.exists():
        print("WARNING: No classifier checkpoint found. Training without task-aware loss.")
        return None
    classifier = SmallCifarClassifier().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(ckpt["model_state"])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)
    print(f"Loaded frozen classifier for task-aware loss: {checkpoint_path}")
    return classifier


def random_snr(batch_size: int, config: dict, device: torch.device) -> torch.Tensor:
    channel_cfg = config["channel"]
    low = float(channel_cfg["train_snr_min_db"])
    high = float(channel_cfg["train_snr_max_db"])
    return torch.empty(batch_size, device=device).uniform_(low, high)


def evaluate_loss(model: SemanticAutoencoder, loader: DataLoader, config: dict, device: torch.device, max_batches: int = 8) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index, (images, _) in enumerate(loader):
            if index >= max_batches:
                break
            images = images.to(device)
            snr = random_snr(images.shape[0], config, device)
            recon = model(images, snr)
            losses.append(F.mse_loss(recon, images).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def compute_lambda_warmup(epoch: int, warmup_epochs: int, target_lambda: float) -> float:
    """Linear warmup: lambda=0 for first warmup_epochs, then ramp linearly."""
    if epoch <= warmup_epochs:
        return 0.0
    total_epochs_after_warmup = max(1, 1)  # Ramp over remaining epochs
    # Simple: full lambda after warmup
    return target_lambda


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    train_loader, val_loader = build_loaders(config, fake_data=args.fake_data)
    model = build_model(config, device)
    train_cfg = config["train"]
    total_epochs = int(train_cfg["epochs"])

    # Task-aware loss setup
    task_loss_weight = args.task_loss_weight
    warmup_epochs = args.warmup_epochs
    classifier = None
    if task_loss_weight > 0:
        classifier = load_frozen_classifier(config, device)
        if classifier is not None:
            print(f"Task-aware loss: target lambda={task_loss_weight}, warmup={warmup_epochs} epochs")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    checkpoint_path = Path(config["semantic"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_every = int(train_cfg.get("log_every", 25))
    global_step = 0

    for epoch in range(1, total_epochs + 1):
        model.train()

        # Compute current lambda with warmup
        if epoch <= warmup_epochs:
            current_lambda = 0.0
        else:
            # Linear ramp from 0 to target over (total_epochs - warmup_epochs)
            ramp_progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            current_lambda = task_loss_weight * min(1.0, ramp_progress)

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            if args.max_batches is not None and batch_index > args.max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            snr = random_snr(images.shape[0], config, device)
            recon = model(images, snr)

            mse_loss = F.mse_loss(recon, images)
            loss = mse_loss

            # Add task-aware loss if classifier available and lambda > 0
            if classifier is not None and current_lambda > 0:
                classifier_logits = classifier(recon.clamp(0, 1))
                ce_loss = F.cross_entropy(classifier_logits, labels)
                loss = mse_loss + current_lambda * ce_loss
            else:
                ce_loss = None

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % log_every == 0 or batch_index == 1:
                ce_str = f" ce={ce_loss.item():.4f}" if ce_loss is not None else ""
                lam_str = f" lam={current_lambda:.5f}" if classifier is not None else ""
                print(f"epoch={epoch} step={global_step} mse={mse_loss.item():.6f}{ce_str}{lam_str} total={loss.item():.6f}")

        val_mse = evaluate_loss(model, val_loader, config, device)
        print(f"epoch={epoch} val_mse={val_mse:.6f}")
        if epoch % int(train_cfg.get("save_every_epochs", 1)) == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                },
                checkpoint_path,
            )
            print(f"saved checkpoint: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train semantic JSCC autoencoder.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake-data", action="store_true", help="Use torchvision FakeData for fast smoke tests.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--task-loss-weight", type=float, default=0.0,
                        help="Weight lambda for classifier CE auxiliary loss. 0 = MSE only (default).")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of epochs to hold lambda at 0 before ramping (default: 5).")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
