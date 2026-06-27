from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor, nn


SNRLike = Union[float, Tensor]


def snr_db_to_noise_variance(snr_db: SNRLike, symbol_power: float = 1.0) -> Tensor:
    """Return complex noise variance N0 for a target Es/N0 in dB."""
    snr = torch.as_tensor(snr_db, dtype=torch.float32)
    return torch.as_tensor(symbol_power, dtype=torch.float32) / torch.pow(10.0, snr / 10.0)


def _reshape_noise_variance(noise_variance: Tensor, symbols: Tensor) -> Tensor:
    if noise_variance.ndim == 0:
        return noise_variance.to(device=symbols.device, dtype=symbols.dtype)
    view_shape = [noise_variance.shape[0]] + [1] * (symbols.ndim - 1)
    return noise_variance.to(device=symbols.device, dtype=symbols.dtype).view(*view_shape)


class AWGNChannel(nn.Module):
    """Differentiable AWGN channel for IQ tensors shaped [..., 2].

    The last dimension stores in-phase and quadrature components. For complex
    noise variance N0, each real dimension gets variance N0 / 2. This keeps the
    channel physically aligned with complex baseband samples and lets gradients
    flow through y = x + n into the semantic encoder.
    """

    def __init__(self, symbol_power: float = 1.0) -> None:
        super().__init__()
        self.symbol_power = float(symbol_power)

    def forward(self, symbols: Tensor, snr_db: SNRLike) -> Tensor:
        if symbols.shape[-1] != 2:
            raise ValueError("AWGNChannel expects IQ tensors with last dimension size 2.")
        noise_variance = snr_db_to_noise_variance(snr_db, self.symbol_power)
        noise_variance = _reshape_noise_variance(noise_variance, symbols)
        noise_std = torch.sqrt(noise_variance / 2.0)
        return symbols + torch.randn_like(symbols) * noise_std


class RayleighBlockFadingChannel(nn.Module):
    """Simple differentiable flat Rayleigh fading channel with perfect equalization.

    This is a software-only robustness test, not a full 3GPP CDL implementation.
    The class keeps the same IQ interface as AWGN so a Sionna CDL wrapper can be
    added later without touching the semantic encoder/decoder contract.
    """

    def __init__(self, symbol_power: float = 1.0, equalize: bool = True) -> None:
        super().__init__()
        self.symbol_power = float(symbol_power)
        self.equalize = equalize
        self.awgn = AWGNChannel(symbol_power=symbol_power)

    def forward(self, symbols: Tensor, snr_db: SNRLike) -> Tensor:
        if symbols.shape[-1] != 2:
            raise ValueError("RayleighBlockFadingChannel expects IQ tensors with last dimension size 2.")

        batch = symbols.shape[0]
        h = torch.randn(batch, 1, 2, device=symbols.device, dtype=symbols.dtype) / math.sqrt(2.0)
        y = complex_mul(symbols, h)
        y = self.awgn(y, snr_db)
        if not self.equalize:
            return y
        h_conj = h.clone()
        h_conj[..., 1] *= -1
        h_power = h.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return complex_mul(y, h_conj) / h_power


def complex_mul(a: Tensor, b: Tensor) -> Tensor:
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack((real, imag), dim=-1)


def make_channel(channel_type: str, symbol_power: float = 1.0) -> nn.Module:
    name = channel_type.lower()
    if name == "awgn":
        return AWGNChannel(symbol_power=symbol_power)
    if name in {"rayleigh", "rayleigh_block"}:
        return RayleighBlockFadingChannel(symbol_power=symbol_power)
    raise ValueError(f"Unsupported channel type: {channel_type}")

