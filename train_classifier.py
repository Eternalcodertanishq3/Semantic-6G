from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.cifar_classifier import SmallCifarClassifier
from train import select_device


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(config: dict, fake_data: bool = False) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.ToTensor()
    if fake_data:
        train_dataset = datasets.FakeData(size=2048, image_size=(3, 32, 32), num_classes=10, transform=train_transform)
        val_dataset = datasets.FakeData(size=512, image_size=(3, 32, 32), num_classes=10, transform=eval_transform)
    else:
        full_train = datasets.CIFAR10(root=data_cfg["root"], train=True, transform=train_transform, download=True)
        val_size = int(data_cfg.get("val_size", 1000))
        train_size = len(full_train) - val_size
        generator = torch.Generator().manual_seed(int(config["seed"]))
        train_dataset, _ = random_split(full_train, [train_size, val_size], generator=generator)
        val_dataset = datasets.CIFAR10(root=data_cfg["root"], train=False, transform=eval_transform, download=True)
    kwargs = {
        "batch_size": int(config.get("classifier", {}).get("batch_size", data_cfg["batch_size"])),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    return DataLoader(train_dataset, shuffle=True, **kwargs), DataLoader(val_dataset, shuffle=False, **kwargs)


def evaluate_classifier(model: SmallCifarClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    model.train()
    return correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    classifier_cfg = config.get("classifier", {})
    if args.epochs is not None:
        classifier_cfg["epochs"] = args.epochs
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    train_loader, val_loader = build_loaders(config, fake_data=args.fake_data)
    model = SmallCifarClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(classifier_cfg.get("lr", 0.001)),
        weight_decay=float(classifier_cfg.get("weight_decay", 0.0005)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(classifier_cfg.get("epochs", 40))),
    )
    checkpoint_path = Path(classifier_cfg.get("checkpoint", "checkpoints/cifar_classifier.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(classifier_cfg.get("epochs", 40)) + 1):
        model.train()
        running = []
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            if args.max_batches is not None and batch_index > args.max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running.append(loss.item())
        scheduler.step()
        accuracy = evaluate_classifier(model, val_loader, device)
        print(f"epoch={epoch} train_ce={np.mean(running):.4f} val_acc={accuracy:.4f}")
        torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch, "val_acc": accuracy}, checkpoint_path)
        print(f"saved classifier checkpoint: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen CIFAR-10 meaning classifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
