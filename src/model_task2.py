"""Encoder-decoder Transformer for melody -> chord harmonization.

The encoder reads the 16th-note grid melody sequence; the decoder emits one
chord token per bar (autoregressively, with BOS / EOS delimiters). Cross
attention is unrestricted -- since the melody and chord streams live on
different time grids we don't impose any alignment mask. The model learns to
attend over the melody bars that are relevant for each chord position.

This is intentionally a small model (~5-10M params) so a full training run
fits inside an hour on an 8 GB consumer GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import (
    CHORD_BOS_ID,
    CHORD_EOS_ID,
    CHORD_PAD_ID,
    MEL_PAD_ID,
    chord_vocab_size,
    melody_vocab_size,
)


@dataclass
class HarmonizerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_melody_len: int = 1024  # 16 bars * 16 steps = 256, plenty of headroom
    max_chord_len: int = 64


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class Harmonizer(nn.Module):
    def __init__(self, cfg: HarmonizerConfig | None = None):
        super().__init__()
        cfg = cfg or HarmonizerConfig()
        self.cfg = cfg
        self.mel_vocab = melody_vocab_size()
        self.chord_vocab = chord_vocab_size()

        self.mel_emb = nn.Embedding(self.mel_vocab, cfg.d_model, padding_idx=MEL_PAD_ID)
        self.chord_emb = nn.Embedding(self.chord_vocab, cfg.d_model, padding_idx=CHORD_PAD_ID)
        self.mel_pe = PositionalEncoding(cfg.d_model, cfg.max_melody_len)
        self.chord_pe = PositionalEncoding(cfg.d_model, cfg.max_chord_len)
        self.dropout = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_enc_layers, enable_nested_tensor=False
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_dec_layers)

        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, self.chord_vocab)
        nn.init.normal_(self.mel_emb.weight, mean=0.0, std=cfg.d_model ** -0.5)
        nn.init.normal_(self.chord_emb.weight, mean=0.0, std=cfg.d_model ** -0.5)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=cfg.d_model ** -0.5)
        nn.init.zeros_(self.out_proj.bias)

    def _encode(self, melody: torch.Tensor, mel_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.mel_emb(melody) * math.sqrt(self.cfg.d_model)
        x = self.mel_pe(x)
        x = self.dropout(x)
        key_padding = ~mel_mask if mel_mask is not None else None
        return self.encoder(x, src_key_padding_mask=key_padding)

    def _decode(
        self,
        chord_in: torch.Tensor,
        memory: torch.Tensor,
        chord_mask: torch.Tensor | None,
        mel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        y = self.chord_emb(chord_in) * math.sqrt(self.cfg.d_model)
        y = self.chord_pe(y)
        y = self.dropout(y)
        tgt_len = chord_in.shape[1]
        causal = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=chord_in.device),
            diagonal=1,
        )
        tgt_key_padding = ~chord_mask if chord_mask is not None else None
        mem_key_padding = ~mel_mask if mel_mask is not None else None
        out = self.decoder(
            y,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=mem_key_padding,
        )
        return self.out_proj(self.out_norm(out))

    def forward(
        self,
        melody: torch.Tensor,
        chord_in: torch.Tensor,
        *,
        mel_mask: torch.Tensor | None = None,
        chord_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self._encode(melody, mel_mask)
        return self._decode(chord_in, memory, chord_mask, mel_mask)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def harmonize(
        self,
        melody: torch.Tensor,
        n_bars: int,
        *,
        mel_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Autoregressive sampling. ``melody`` shape (B, L), returns (B, n_bars)."""
        self.eval()
        device = melody.device
        memory = self._encode(melody, mel_mask)
        B = melody.shape[0]
        seq = torch.full((B, 1), CHORD_BOS_ID, dtype=torch.long, device=device)
        for _ in range(n_bars):
            logits = self._decode(seq, memory, chord_mask=None, mel_mask=mel_mask)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if greedy:
                next_id = next_logits.argmax(-1, keepdim=True)
            else:
                next_id = _nucleus_sample(next_logits, top_p)
            seq = torch.cat([seq, next_id], dim=1)
        return seq[:, 1:]  # drop BOS

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _nucleus_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum > top_p
    mask[..., 0] = False  # keep at least the top one
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    idx_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
    chosen = sorted_idx.gather(-1, idx_in_sorted)
    return chosen
