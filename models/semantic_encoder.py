from __future__ import annotations

import torch
from torch import Tensor, nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.prelu(out + residual)


class SemanticEncoder(nn.Module):
    """CNN image encoder that emits normalized IQ-style channel symbols."""

    def __init__(self, num_symbols: int = 384, latent_channels: int = 96, symbol_power: float = 1.0) -> None:
        super().__init__()
        self.num_symbols = int(num_symbols)
        self.symbol_power = float(symbol_power)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.PReLU(),
            ResBlock(latent_channels),
            ResBlock(latent_channels),
        )
        self.projection = nn.Linear(latent_channels * 4 * 4, self.num_symbols * 2)

    def forward(self, images: Tensor) -> Tensor:
        z = self.features(images)
        z = z.flatten(start_dim=1)
        symbols = self.projection(z).view(images.shape[0], self.num_symbols, 2)
        return self._normalize_power(symbols)

    def _normalize_power(self, symbols: Tensor) -> Tensor:
        power = symbols.pow(2).sum(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
        target = torch.as_tensor(self.symbol_power, device=symbols.device, dtype=symbols.dtype)
        return symbols * torch.sqrt(target / power.clamp_min(1e-8))

