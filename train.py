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
from models import SemanticDecoder, SemanticEncoder


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


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    train_loader, val_loader = build_loaders(config, fake_data=args.fake_data)
    model = build_model(config, device)
    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    checkpoint_path = Path(config["semantic"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_every = int(train_cfg.get("log_every", 25))
    global_step = 0

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        for batch_index, (images, _) in enumerate(train_loader, start=1):
            if args.max_batches is not None and batch_index > args.max_batches:
                break
            images = images.to(device)
            snr = random_snr(images.shape[0], config, device)
            recon = model(images, snr)
            loss = F.mse_loss(recon, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % log_every == 0 or batch_index == 1:
                print(f"epoch={epoch} step={global_step} train_mse={loss.item():.6f}")

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
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

