from __future__ import annotations

import torch
from torch import Tensor, nn


class CharVocabulary:
    """Tiny byte-level vocabulary for text semantic communication experiments."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    offset = 3
    vocab_size = 259

    def encode(self, text: str, max_len: int) -> list[int]:
        payload = list(text.encode("utf-8", errors="replace"))[: max_len - 2]
        ids = [self.bos_id] + [byte + self.offset for byte in payload] + [self.eos_id]
        ids += [self.pad_id] * max(0, max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: list[int]) -> str:
        bytes_out: list[int] = []
        for token in ids:
            if token in {self.pad_id, self.bos_id}:
                continue
            if token == self.eos_id:
                break
            if token >= self.offset:
                bytes_out.append(token - self.offset)
        return bytes(bytes_out).decode("utf-8", errors="replace")


class TextSemanticEncoder(nn.Module):
    """GRU encoder that maps token sequences to normalized IQ symbols."""

    def __init__(self, vocab_size: int = 259, embed_dim: int = 128, hidden_dim: int = 192, num_symbols: int = 128, symbol_power: float = 1.0) -> None:
        super().__init__()
        self.num_symbols = int(num_symbols)
        self.symbol_power = float(symbol_power)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, self.num_symbols * 2)

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        _, hidden = self.encoder(embedded)
        state = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        symbols = self.projection(state).view(tokens.shape[0], self.num_symbols, 2)
        power = symbols.pow(2).sum(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
        target = torch.as_tensor(self.symbol_power, device=symbols.device, dtype=symbols.dtype)
        return symbols * torch.sqrt(target / power.clamp_min(1e-8))


class TextSemanticDecoder(nn.Module):
    """GRU decoder scaffold for reconstructing byte-level text tokens."""

    def __init__(self, vocab_size: int = 259, embed_dim: int = 128, hidden_dim: int = 192, num_symbols: int = 128, max_len: int = 96) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.context = nn.Sequential(
            nn.Linear(num_symbols * 2, hidden_dim),
            nn.Tanh(),
        )
        self.bos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, symbols: Tensor) -> Tensor:
        context = self.context(symbols.flatten(start_dim=1))
        repeated_context = context.unsqueeze(1).expand(-1, self.max_len, -1)
        bos = self.bos.expand(symbols.shape[0], self.max_len, -1)
        decoder_input = torch.cat([bos, repeated_context], dim=-1)
        decoded, _ = self.decoder(decoder_input)
        return self.output(decoded)


class TextSemanticAutoencoder(nn.Module):
    """End-to-end text JSCC scaffold that reuses the same differentiable channel API."""

    def __init__(self, encoder: TextSemanticEncoder, channel: nn.Module, decoder: TextSemanticDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.channel = channel
        self.decoder = decoder

    def forward(self, tokens: Tensor, snr_db: Tensor | float) -> Tensor:
        symbols = self.encoder(tokens)
        received = self.channel(symbols, snr_db)
        return self.decoder(received)
