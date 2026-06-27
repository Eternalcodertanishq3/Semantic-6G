from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ClassicalImagePipeline
from train import build_model, select_device, seed_everything


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_semantic_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Semantic checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])


def build_samples(config: dict, count: int, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    dataset = datasets.CIFAR10(root=config["data"]["root"], train=False, transform=transforms.ToTensor(), download=True)
    indices = [0, 7, 13, 21, 42, 88, 123, 256][:count]
    images = torch.stack([dataset[i][0] for i in indices], dim=0).to(device)
    labels = [int(dataset[i][1]) for i in indices]
    return images, labels


def to_image(tensor: torch.Tensor):
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def qualitative_check(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))

    model = build_model(config, device)
    load_semantic_checkpoint(model, Path(config["semantic"]["checkpoint"]), device)
    model.eval()

    classical_cfg = dict(config["classical"])
    classical_cfg["symbol_power"] = float(config["channel"]["symbol_power"])
    classical = ClassicalImagePipeline.from_dict(classical_cfg, device=device)

    images, labels = build_samples(config, args.count, device)
    snrs = [float(x) for x in args.snrs.split(",")]
    cols = len(snrs) * 3
    rows = images.shape[0]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.45))
    if rows == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row in range(rows):
            original = images[row : row + 1]
            for snr_index, snr in enumerate(snrs):
                semantic = model(original, torch.tensor([snr], device=device))
                classical_recon = classical.transmit(original, snr)
                triplet = [original[0], classical_recon[0], semantic[0]]
                titles = [f"orig\ny={labels[row]}", f"classical\n{snr:g} dB", f"semantic\n{snr:g} dB"]
                for item_index, (image, title) in enumerate(zip(triplet, titles)):
                    ax = axes[row, snr_index * 3 + item_index]
                    ax.imshow(to_image(image))
                    ax.set_title(title, fontsize=7)
                    ax.axis("off")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Qualitative reconstruction sanity check", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"saved qualitative grid: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render original/classical/semantic CIFAR-10 reconstruction grid.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="outputs/qualitative_grid.png")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--snrs", default="-5,0,10,20", help="Comma-separated SNR points in dB.")
    return parser.parse_args()


if __name__ == "__main__":
    qualitative_check(parse_args())
