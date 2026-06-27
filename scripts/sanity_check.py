from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel import AWGNChannel


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"commpy installed: {importlib.util.find_spec('commpy') is not None}")
    print(f"sionna installed: {importlib.util.find_spec('sionna') is not None}")
    channel = AWGNChannel(symbol_power=1.0)
    symbols = torch.zeros(2, 8, 2)
    received = channel(symbols, snr_db=10.0)
    print(f"awgn output shape: {tuple(received.shape)}")
    print(f"awgn output std: {received.std().item():.4f}")


if __name__ == "__main__":
    main()
