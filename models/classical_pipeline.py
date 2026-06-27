from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from commpy.channelcoding import Trellis as CommPyTrellis
except Exception:  # pragma: no cover - optional runtime dependency guard
    CommPyTrellis = None


@dataclass(frozen=True)
class ClassicalConfig:
    num_symbols: int = 384
    qam_order: int = 16
    fec: str = "conv_viterbi"
    quant_bits: int = 4
    compressed_size: int = 7
    symbol_power: float = 1.0


class ClassicalImagePipeline:
    """Fixed-rate image codec + convolutional FEC + square QAM channel pipeline.

    The baseline is intentionally budget-locked against the semantic path: exactly
    ``num_symbols`` complex channel uses per image and the same average symbol
    power. A terminated rate-1/2 convolutional code protects the source bits;
    hard-decision Viterbi decoding recovers them after demodulation.
    """

    memory = 2
    generators = (0o7, 0o5)

    def __init__(self, config: ClassicalConfig | None = None, device: torch.device | str = "cpu") -> None:
        self.config = config or ClassicalConfig()
        self.device = torch.device(device)
        qam_side = int(math.sqrt(self.config.qam_order))
        if qam_side * qam_side != self.config.qam_order or qam_side % 2 != 0:
            raise ValueError("Only even square QAM orders such as 16 or 64 are supported.")
        if self.config.fec != "conv_viterbi":
            raise ValueError(f"Unsupported FEC for Windows baseline: {self.config.fec}")
        self.bits_per_symbol = int(math.log2(self.config.qam_order))
        self.target_coded_bits = self.config.num_symbols * self.bits_per_symbol
        self.source_bit_budget = self._source_budget_for_conv_code()
        self.constellation = self._make_constellation().to(self.device)
        self.bit_patterns = self._make_bit_patterns().to(self.device)
        self._trellis = self._build_trellis()
        self.commpy_trellis = self._build_commpy_trellis()

    @classmethod
    def from_dict(cls, cfg: dict, device: torch.device | str = "cpu") -> "ClassicalImagePipeline":
        return cls(
            ClassicalConfig(
                num_symbols=int(cfg.get("num_symbols", 384)),
                qam_order=int(cfg.get("qam_order", 16)),
                fec=str(cfg.get("fec", "conv_viterbi")),
                quant_bits=int(cfg.get("quant_bits", 4)),
                compressed_size=int(cfg.get("compressed_size", 7)),
                symbol_power=float(cfg.get("symbol_power", 1.0)),
            ),
            device=device,
        )

    @property
    def symbols_per_image(self) -> int:
        return self.config.num_symbols

    @property
    def coded_bit_count(self) -> int:
        return self.target_coded_bits

    @property
    def encoded_bit_count(self) -> int:
        return 2 * (self.source_bit_budget + self.memory)

    def average_symbol_power(self) -> float:
        return float(self.constellation.pow(2).sum(dim=-1).mean().item())

    def transmit(self, images: Tensor, snr_db: float) -> Tensor:
        bits, codec_shape = self.compress(images)
        coded = self.encode_bits(bits)
        symbols = self.modulate(coded)
        received = self.awgn(symbols, snr_db)
        hard_bits = self.demodulate(received)
        decoded = self.decode_bits(hard_bits)
        return self.decompress(decoded, codec_shape)

    def compress(self, images: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        images = images.to(self.device)
        size = self.config.compressed_size
        small = F.interpolate(images, size=(size, size), mode="area")
        levels = (1 << self.config.quant_bits) - 1
        quantized = torch.clamp(torch.round(small * levels), 0, levels).to(torch.long)
        bits = self._ints_to_bits(quantized.flatten(start_dim=1), self.config.quant_bits)
        bits = bits[:, : self.source_bit_budget]
        if bits.shape[1] < self.source_bit_budget:
            bits = F.pad(bits, (0, self.source_bit_budget - bits.shape[1]))
        return bits, (3, size, size)

    def decompress(self, bits: Tensor, codec_shape: tuple[int, int, int]) -> Tensor:
        batch = bits.shape[0]
        channels, height, width = codec_shape
        value_count = channels * height * width
        required_bits = value_count * self.config.quant_bits
        bits = bits[:, :required_bits]
        if bits.shape[1] < required_bits:
            bits = F.pad(bits, (0, required_bits - bits.shape[1]))
        values = self._bits_to_ints(bits, self.config.quant_bits)
        levels = (1 << self.config.quant_bits) - 1
        small = values[:, :value_count].float().view(batch, channels, height, width) / levels
        return F.interpolate(small, size=(32, 32), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    def encode_bits(self, bits: Tensor) -> Tensor:
        bits_np = bits.detach().cpu().numpy().astype(np.uint8)
        encoded = np.stack([self._conv_encode_one(row) for row in bits_np], axis=0)
        if encoded.shape[1] > self.target_coded_bits:
            raise ValueError("Convolutional encoder exceeded the configured channel-use budget.")
        if encoded.shape[1] < self.target_coded_bits:
            pad = self.target_coded_bits - encoded.shape[1]
            encoded = np.pad(encoded, ((0, 0), (0, pad)), constant_values=0)
        return torch.from_numpy(encoded.astype(np.int64)).to(self.device)

    def decode_bits(self, coded_bits: Tensor) -> Tensor:
        usable = self.encoded_bit_count
        coded_np = coded_bits[:, :usable].detach().cpu().numpy().astype(np.uint8)
        decoded = np.stack([self._viterbi_decode_one(row) for row in coded_np], axis=0)
        return torch.from_numpy(decoded.astype(np.int64)).to(self.device)

    def modulate(self, bits: Tensor) -> Tensor:
        bits = bits.to(self.device).to(torch.long)
        groups = bits.view(bits.shape[0], self.config.num_symbols, self.bits_per_symbol)
        powers = (2 ** torch.arange(self.bits_per_symbol - 1, -1, -1, device=self.device)).long()
        indexes = (groups * powers).sum(dim=-1)
        return self.constellation[indexes]

    def demodulate(self, symbols: Tensor) -> Tensor:
        distances = (symbols.unsqueeze(-2) - self.constellation.view(1, 1, -1, 2)).pow(2).sum(dim=-1)
        nearest = distances.argmin(dim=-1)
        return self.bit_patterns[nearest].reshape(symbols.shape[0], -1)

    def awgn(self, symbols: Tensor, snr_db: float) -> Tensor:
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        noise_variance = self.config.symbol_power / snr_linear
        noise_std = math.sqrt(noise_variance / 2.0)
        return symbols + torch.randn_like(symbols) * noise_std

    def _source_budget_for_conv_code(self) -> int:
        source = self.target_coded_bits // 2 - self.memory
        source -= source % self.config.quant_bits
        if source <= 0:
            raise ValueError("Channel-use budget is too small for the configured convolutional code.")
        return source

    def _build_trellis(self) -> dict[int, dict[int, tuple[int, np.ndarray]]]:
        trellis: dict[int, dict[int, tuple[int, np.ndarray]]] = {}
        for state in range(1 << self.memory):
            trellis[state] = {}
            prev_bits = self._state_to_bits(state)
            for input_bit in (0, 1):
                register = [input_bit] + prev_bits
                out = np.array([self._parity(register, generator) for generator in self.generators], dtype=np.uint8)
                next_state = self._bits_to_state(register[:-1])
                trellis[state][input_bit] = (next_state, out)
        return trellis

    def _build_commpy_trellis(self):
        if CommPyTrellis is None:
            return None
        return CommPyTrellis(np.array([self.memory]), np.array([list(self.generators)]))

    def _conv_encode_one(self, bits: np.ndarray) -> np.ndarray:
        state = 0
        out_bits: list[int] = []
        for bit in np.concatenate([bits[: self.source_bit_budget], np.zeros(self.memory, dtype=np.uint8)]):
            next_state, out = self._trellis[state][int(bit)]
            out_bits.extend(int(x) for x in out)
            state = next_state
        return np.asarray(out_bits, dtype=np.uint8)

    def _viterbi_decode_one(self, coded_bits: np.ndarray) -> np.ndarray:
        pairs = coded_bits.reshape(-1, 2)
        state_count = 1 << self.memory
        inf = 1_000_000
        metrics = np.full(state_count, inf, dtype=np.int32)
        metrics[0] = 0
        predecessors = np.zeros((len(pairs), state_count), dtype=np.uint8)
        decisions = np.zeros((len(pairs), state_count), dtype=np.uint8)

        for t, observed in enumerate(pairs):
            next_metrics = np.full(state_count, inf, dtype=np.int32)
            for state in range(state_count):
                if metrics[state] >= inf:
                    continue
                for input_bit in (0, 1):
                    next_state, expected = self._trellis[state][input_bit]
                    branch = int(np.count_nonzero(observed != expected))
                    candidate = metrics[state] + branch
                    if candidate < next_metrics[next_state]:
                        next_metrics[next_state] = candidate
                        predecessors[t, next_state] = state
                        decisions[t, next_state] = input_bit
            metrics = next_metrics

        state = 0 if metrics[0] < inf else int(metrics.argmin())
        decoded = np.zeros(len(pairs), dtype=np.uint8)
        for t in range(len(pairs) - 1, -1, -1):
            decoded[t] = decisions[t, state]
            state = int(predecessors[t, state])
        return decoded[: self.source_bit_budget]

    def _make_constellation(self) -> Tensor:
        side = int(math.sqrt(self.config.qam_order))
        levels = torch.arange(-(side - 1), side, 2, dtype=torch.float32)
        yy, xx = torch.meshgrid(levels.flip(0), levels, indexing="ij")
        points = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        avg_power = points.pow(2).sum(dim=-1).mean()
        points = points / torch.sqrt(avg_power)
        return points * math.sqrt(self.config.symbol_power)

    def _make_bit_patterns(self) -> Tensor:
        values = torch.arange(self.config.qam_order)
        bits = ((values[:, None] >> torch.arange(self.bits_per_symbol - 1, -1, -1)) & 1).long()
        return bits

    def _state_to_bits(self, state: int) -> list[int]:
        return [(state >> shift) & 1 for shift in range(self.memory - 1, -1, -1)]

    def _bits_to_state(self, bits: list[int]) -> int:
        state = 0
        for bit in bits:
            state = (state << 1) | int(bit)
        return state

    def _parity(self, register: list[int], generator: int) -> int:
        bits = [(generator >> shift) & 1 for shift in range(self.memory, -1, -1)]
        value = 0
        for reg_bit, gen_bit in zip(register, bits):
            if gen_bit:
                value ^= int(reg_bit)
        return value

    @staticmethod
    def _ints_to_bits(values: Tensor, width: int) -> Tensor:
        shifts = torch.arange(width - 1, -1, -1, device=values.device)
        return ((values.unsqueeze(-1) >> shifts) & 1).flatten(start_dim=1).long()

    @staticmethod
    def _bits_to_ints(bits: Tensor, width: int) -> Tensor:
        groups = bits.view(bits.shape[0], -1, width).long()
        powers = (2 ** torch.arange(width - 1, -1, -1, device=bits.device)).long()
        return (groups * powers).sum(dim=-1)
