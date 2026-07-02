from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F
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


class RayleighChannel(nn.Module):
    """Differentiable Rayleigh fading channel with ZF equalization.

    Supports two fading modes:
    - 'block': one fading coefficient h per batch item (same fade across all
      symbols in one transmission). h shape: [batch, 1, 2].
    - 'fast': one fading coefficient h per symbol per batch item (independent
      fade on every symbol). h shape: [batch, num_symbols, 2].

    Equalization:
    - Block fading: ZF equalization divides all symbols by the single h.
    - Fast fading: per-symbol ZF — y_eq[i] = y[i] / h[i] for each symbol
      position i, where each received symbol is divided by its own fading
      coefficient.

    This is a software-only robustness test, not a full 3GPP CDL implementation.
    """

    def __init__(self, symbol_power: float = 1.0, fading_type: str = "block",
                 equalize: bool = True) -> None:
        super().__init__()
        self.symbol_power = float(symbol_power)
        self.fading_type = fading_type
        self.equalize = equalize
        self.awgn = AWGNChannel(symbol_power=symbol_power)

    def forward(self, symbols: Tensor, snr_db: SNRLike) -> Tensor:
        if symbols.shape[-1] != 2:
            raise ValueError("RayleighChannel expects IQ tensors with last dimension size 2.")

        batch = symbols.shape[0]
        if self.fading_type == "block":
            # One h per batch item, broadcast across all symbols
            h = torch.randn(batch, 1, 2, device=symbols.device, dtype=symbols.dtype) / math.sqrt(2.0)
        elif self.fading_type == "fast":
            # One h per symbol per batch item
            num_symbols = symbols.shape[1]
            h = torch.randn(batch, num_symbols, 2, device=symbols.device, dtype=symbols.dtype) / math.sqrt(2.0)
        else:
            raise ValueError(f"Unsupported fading_type: {self.fading_type}")

        # Apply fading: y = h * x (complex multiplication)
        y = complex_mul(symbols, h)
        # Add AWGN on top of faded signal
        y = self.awgn(y, snr_db)

        if not self.equalize:
            return y

        # ZF equalization: x_hat = y * conj(h) / |h|^2
        # For fast fading this is per-symbol: y_eq[i] = y[i] / h[i]
        h_conj = h.clone()
        h_conj[..., 1] *= -1
        h_power = h.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return complex_mul(y, h_conj) / h_power


# Backward-compatible alias
RayleighBlockFadingChannel = RayleighChannel


class CDLApproxChannel(nn.Module):
    """Differentiable frequency-selective fading channel (CDL approximation).

    Approximates 3GPP CDL-B/CDL-C level frequency selectivity using a tapped
    delay line (FIR filter with random complex taps) without requiring Sionna.

    Implementation:
    - Sample L complex taps from CN(0, 1), normalized so sum of power = 1
    - Apply as causal convolution over the symbol sequence
    - Add AWGN on top
    - Frequency-domain MMSE equalization (known channel) via FFT:
        Y = FFT(received_symbols)
        H = FFT(channel_taps, n=num_symbols)
        X_hat = Y * conj(H) / (|H|^2 + 1/SNR)
        output = IFFT(X_hat)

    NOTE: This is explicitly a practical approximation, NOT the real 3GPP CDL.
    The exact CDL comes in Phase 2B via Sionna on Docker.
    """

    def __init__(self, symbol_power: float = 1.0, num_taps: int = 6,
                 equalize: bool = True) -> None:
        super().__init__()
        self.symbol_power = float(symbol_power)
        self.num_taps = num_taps
        self.equalize = equalize

    def forward(self, symbols: Tensor, snr_db: SNRLike) -> Tensor:
        if symbols.shape[-1] != 2:
            raise ValueError("CDLApproxChannel expects IQ tensors with last dimension size 2.")

        batch, num_symbols, _ = symbols.shape
        device = symbols.device
        dtype = symbols.dtype

        # Sample L complex taps from CN(0, 1), then normalize so sum |h_l|^2 = 1
        h_taps = torch.randn(batch, self.num_taps, 2, device=device, dtype=dtype) / math.sqrt(2.0)
        tap_power = h_taps.pow(2).sum(dim=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-8)
        h_taps = h_taps / torch.sqrt(tap_power)

        # Convert IQ to complex for convolution
        x_complex = torch.complex(symbols[..., 0], symbols[..., 1])  # [batch, num_symbols]
        h_complex = torch.complex(h_taps[..., 0], h_taps[..., 1])    # [batch, num_taps]

        # Causal convolution via FFT (circular, zero-padded to avoid wrap-around)
        n_fft = num_symbols + self.num_taps - 1
        X = torch.fft.fft(x_complex, n=n_fft, dim=-1)   # [batch, n_fft]
        H = torch.fft.fft(h_complex, n=n_fft, dim=-1)    # [batch, n_fft]
        Y = X * H                                         # pointwise multiplication
        y_time = torch.fft.ifft(Y, dim=-1)                # [batch, n_fft]
        # Truncate to original length (causal: keep first num_symbols samples)
        y_time = y_time[..., :num_symbols]

        # Back to IQ format [batch, num_symbols, 2]
        y_iq = torch.stack((y_time.real, y_time.imag), dim=-1)

        # Add AWGN
        noise_variance = snr_db_to_noise_variance(snr_db, self.symbol_power)
        noise_variance = _reshape_noise_variance(noise_variance, y_iq)
        noise_std = torch.sqrt(noise_variance / 2.0)
        y_iq = y_iq + torch.randn_like(y_iq) * noise_std

        if not self.equalize:
            return y_iq

        # Frequency-domain MMSE equalization (known channel)
        # Y = FFT(received), H = FFT(taps, n=num_symbols)
        # X_hat = Y * conj(H) / (|H|^2 + 1/SNR)
        y_eq_complex = torch.complex(y_iq[..., 0], y_iq[..., 1])
        Y_freq = torch.fft.fft(y_eq_complex, n=num_symbols, dim=-1)
        H_freq = torch.fft.fft(h_complex, n=num_symbols, dim=-1)

        snr_linear = torch.pow(10.0, torch.as_tensor(snr_db, dtype=dtype, device=device) / 10.0)
        if snr_linear.ndim == 0:
            inv_snr = 1.0 / snr_linear
        else:
            inv_snr = (1.0 / snr_linear).view(-1, 1)

        H_conj = torch.conj(H_freq)
        H_power = (H_freq * H_conj).real
        X_hat_freq = Y_freq * H_conj / (H_power + inv_snr)
        x_hat = torch.fft.ifft(X_hat_freq, dim=-1)

        return torch.stack((x_hat.real, x_hat.imag), dim=-1)


def complex_mul(a: Tensor, b: Tensor) -> Tensor:
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack((real, imag), dim=-1)


def make_channel(channel_type: str, symbol_power: float = 1.0) -> nn.Module:
    name = channel_type.lower()
    if name == "awgn":
        return AWGNChannel(symbol_power=symbol_power)
    if name in {"rayleigh", "rayleigh_block"}:
        return RayleighChannel(symbol_power=symbol_power, fading_type="block")
    if name == "rayleigh_fast":
        return RayleighChannel(symbol_power=symbol_power, fading_type="fast")
    if name == "cdl_approx":
        return CDLApproxChannel(symbol_power=symbol_power)
    raise ValueError(f"Unsupported channel type: {channel_type}")


