from __future__ import annotations

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


class SemanticDecoder(nn.Module):
    """CNN decoder that reconstructs a 32x32 RGB image from received IQ symbols."""

    def __init__(self, num_symbols: int = 384, latent_channels: int = 96) -> None:
        super().__init__()
        self.num_symbols = int(num_symbols)
        self.latent_channels = int(latent_channels)
        self.input_projection = nn.Sequential(
            nn.Linear(self.num_symbols * 2, self.latent_channels * 4 * 4),
            nn.PReLU(),
        )
        self.reconstruction = nn.Sequential(
            ResBlock(latent_channels),
            ResBlock(latent_channels),
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(24, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, symbols: Tensor) -> Tensor:
        z = symbols.flatten(start_dim=1)
        z = self.input_projection(z)
        z = z.view(symbols.shape[0], self.latent_channels, 4, 4)
        return self.reconstruction(z)

