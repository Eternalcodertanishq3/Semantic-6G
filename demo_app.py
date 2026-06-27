from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torchvision import datasets, transforms

from models import ClassicalImagePipeline
from train import build_model, select_device, seed_everything


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_sample(config: dict, index: int) -> torch.Tensor:
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=config["data"]["root"], train=False, transform=transform, download=True)
    image, _ = dataset[index % len(dataset)]
    return image.unsqueeze(0)


def main() -> None:
    import streamlit as st

    config = load_config()
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    st.set_page_config(page_title="Semantic 6G Demo", layout="wide")
    st.title("Semantic Communication over a Noisy 6G-Style Channel")

    snr = st.slider("SNR (dB)", min_value=-5.0, max_value=20.0, value=5.0, step=0.5)
    sample_index = st.number_input("CIFAR-10 sample", min_value=0, max_value=9999, value=0, step=1)

    image = load_sample(config, int(sample_index)).to(device)
    model = build_model(config, device)
    checkpoint_path = Path(config["semantic"]["checkpoint"])
    checkpoint_loaded = False
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        checkpoint_loaded = True
    model.eval()

    classical_cfg = dict(config["classical"])
    classical_cfg["symbol_power"] = float(config["channel"]["symbol_power"])
    classical = ClassicalImagePipeline.from_dict(classical_cfg, device=device)

    with torch.no_grad():
        semantic = model(image, torch.tensor([snr], device=device))
        classical_recon = classical.transmit(image, snr)

    if not checkpoint_loaded:
        st.warning("No trained checkpoint found yet. Train with `python train.py --config config.yaml` for a meaningful semantic reconstruction.")

    col1, col2, col3 = st.columns(3)
    col1.image(to_display(image), caption="Original", use_container_width=True)
    col2.image(to_display(classical_recon), caption="Classical", use_container_width=True)
    col3.image(to_display(semantic), caption="Semantic", use_container_width=True)


def to_display(tensor: torch.Tensor):
    return tensor.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


if __name__ == "__main__":
    main()

