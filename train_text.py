from __future__ import annotations

import argparse
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from channel import make_channel
from models import (
    CharVocabulary,
    TextSemanticAutoencoder,
    TextSemanticDecoder,
    TextSemanticEncoder,
)


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


class TextChunkDataset(Dataset):
    def __init__(self, text: str, vocab: CharVocabulary, max_len: int, num_samples: int):
        self.text = text
        self.vocab = vocab
        self.max_len = max_len
        self.num_samples = num_samples
        self.length = len(text)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = random.randint(0, max(0, self.length - self.max_len - 1))
        chunk = self.text[start : start + self.max_len]
        tokens = self.vocab.encode(chunk, self.max_len)
        return torch.tensor(tokens, dtype=torch.long)


def build_loaders(config: dict, fake_data: bool = False) -> tuple[DataLoader, DataLoader]:
    text_cfg = config["text"]
    vocab = CharVocabulary()
    max_len = int(text_cfg["max_len"])

    if fake_data:
        text = "Hello world! This is a fake dataset just for testing the text pipeline. " * 100
    else:
        data_path = Path("data/tinyshakespeare.txt")
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = TextChunkDataset(train_text, vocab, max_len, num_samples=10000 if not fake_data else 100)
    val_dataset = TextChunkDataset(val_text, vocab, max_len, num_samples=1000 if not fake_data else 20)

    loader_kwargs = {
        "batch_size": int(text_cfg["batch_size"]),
        "num_workers": int(config["data"].get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    )


def build_model(config: dict, device: torch.device) -> TextSemanticAutoencoder:
    channel_cfg = config["channel"]
    text_cfg = config["text"]

    encoder = TextSemanticEncoder(
        vocab_size=CharVocabulary.vocab_size,
        embed_dim=int(text_cfg["embed_dim"]),
        hidden_dim=int(text_cfg["hidden_dim"]),
        num_symbols=int(text_cfg["num_symbols"]),
        symbol_power=float(channel_cfg["symbol_power"]),
    )
    channel = make_channel(channel_cfg["type"], symbol_power=float(channel_cfg["symbol_power"]))
    decoder = TextSemanticDecoder(
        vocab_size=CharVocabulary.vocab_size,
        embed_dim=int(text_cfg["embed_dim"]),
        hidden_dim=int(text_cfg["hidden_dim"]),
        num_symbols=int(text_cfg["num_symbols"]),
        max_len=int(text_cfg["max_len"]),
    )
    return TextSemanticAutoencoder(encoder, channel, decoder).to(device)


def random_snr(batch_size: int, config: dict, device: torch.device) -> torch.Tensor:
    channel_cfg = config["channel"]
    low = float(channel_cfg["train_snr_min_db"])
    high = float(channel_cfg["train_snr_max_db"])
    return torch.empty(batch_size, device=device).uniform_(low, high)


def evaluate_loss(
    model: TextSemanticAutoencoder, loader: DataLoader, config: dict, device: torch.device, max_batches: int = 8
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index, tokens in enumerate(loader):
            if index >= max_batches:
                break
            tokens = tokens.to(device)
            snr = random_snr(tokens.shape[0], config, device)
            logits = model(tokens, snr)
            loss = F.cross_entropy(
                logits.view(-1, CharVocabulary.vocab_size), tokens.view(-1), ignore_index=CharVocabulary.pad_id
            )
            losses.append(loss.item())
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

    checkpoint_path = Path(config["text"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_every = int(train_cfg.get("log_every", 25))
    global_step = 0

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        for batch_index, tokens in enumerate(train_loader, start=1):
            if args.max_batches is not None and batch_index > args.max_batches:
                break
            tokens = tokens.to(device)
            snr = random_snr(tokens.shape[0], config, device)
            logits = model(tokens, snr)

            loss = F.cross_entropy(
                logits.view(-1, CharVocabulary.vocab_size), tokens.view(-1), ignore_index=CharVocabulary.pad_id
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % log_every == 0 or batch_index == 1:
                print(f"epoch={epoch} step={global_step} train_ce={loss.item():.6f}")

        val_ce = evaluate_loss(model, val_loader, config, device)
        print(f"epoch={epoch} val_ce={val_ce:.6f}")
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
    parser = argparse.ArgumentParser(description="Train text semantic JSCC autoencoder.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake-data", action="store_true", help="Use fake text for fast smoke tests.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
