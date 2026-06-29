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
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bos_token = CharVocabulary.bos_id
        
        self.decoder = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, symbols: Tensor, targets: Tensor | None = None) -> Tensor:
        batch_size = symbols.shape[0]
        context = self.context(symbols.flatten(start_dim=1))
        hidden = None
        
        if targets is not None:
            # Teacher forcing
            bos_tokens = torch.full((batch_size, 1), self.bos_token, device=symbols.device, dtype=torch.long)
            decoder_input_tokens = torch.cat([bos_tokens, targets[:, :-1]], dim=1)
            
            embedded = self.embedding(decoder_input_tokens)
            repeated_context = context.unsqueeze(1).expand(-1, self.max_len, -1)
            decoder_input = torch.cat([embedded, repeated_context], dim=-1)
            
            decoded, _ = self.decoder(decoder_input, hidden)
            return self.output(decoded)
            
        else:
            # Autoregressive decoding
            logits_list = []
            current_token = torch.full((batch_size, 1), self.bos_token, device=symbols.device, dtype=torch.long)
            
            for _ in range(self.max_len):
                embedded = self.embedding(current_token)
                repeated_context = context.unsqueeze(1)
                decoder_input = torch.cat([embedded, repeated_context], dim=-1)
                
                decoded, hidden = self.decoder(decoder_input, hidden)
                step_logits = self.output(decoded)
                logits_list.append(step_logits)
                
                current_token = step_logits.argmax(dim=-1)
                
            return torch.cat(logits_list, dim=1)


class TextSemanticAutoencoder(nn.Module):
    """End-to-end text JSCC scaffold that reuses the same differentiable channel API."""

    def __init__(self, encoder: TextSemanticEncoder, channel: nn.Module, decoder: TextSemanticDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.channel = channel
        self.decoder = decoder

    def forward(self, tokens: Tensor, snr_db: Tensor | float, targets: Tensor | None = None) -> Tensor:
        symbols = self.encoder(tokens)
        received = self.channel(symbols, snr_db)
        return self.decoder(received, targets)
