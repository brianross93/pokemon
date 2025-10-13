"""
FBAM baseline with intra-frame attention and recurrent integration, but no symbolic memory.

This implementation mirrors the architecture described in "Attention is Not All You Need:
A Recurrence-Complete Alternative for Sequential Computation" (FBAM) by using
Transformer attention within each frame and an LSTM cell across frames. It omits the
symbolic ASSOC/FOLLOW operators used by SR-FBAM, providing a direct ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from src.models.sr_fbam import SimpleTokenizer


@dataclass
class FBAMBaselineConfig:
    token_vocab_size: int = 4096
    token_embedding_dim: int = 72
    frame_dim: int = 160
    hidden_dim: int = 192
    num_attention_heads: int = 4
    num_encoder_layers: int = 1
    ff_hidden_dim: int = 256
    dropout: float = 0.1
    node_vocab_size: int = 2048
    max_seq_len: int = 96
    integrator_steps: int = 10
    conditioning_scale: float = 0.1


class FBAMFrameHead(nn.Module):
    """FBAM frame encoder with intra-frame attention."""

    def __init__(self, config: FBAMBaselineConfig) -> None:
        super().__init__()
        self.config = config
        cfg = self.config

        self.tokenizer = SimpleTokenizer(cfg.token_vocab_size)
        self.token_embed = nn.Embedding(cfg.token_vocab_size, cfg.token_embedding_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.token_embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_embedding_dim,
            nhead=cfg.num_attention_heads,
            dim_feedforward=cfg.ff_hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)
        self.proj = nn.Linear(cfg.token_embedding_dim, cfg.frame_dim)
        self.state_proj = nn.Linear(cfg.hidden_dim, cfg.frame_dim)

    def forward(
        self,
        query_text: str,
        hidden_state: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(query_text)[: self.config.max_seq_len]
        if not tokens:
            tokens = ["<blank>"]

        indices = torch.tensor(
            [self.tokenizer.token_to_index(tok) for tok in tokens],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)  # [1, seq_len]

        pos = torch.arange(indices.size(1), device=device).unsqueeze(0)
        embeddings = self.token_embed(indices) + self.pos_embed(pos)
        attended = self.encoder(embeddings)
        pooled = attended.mean(dim=1)  # [1, d_model]
        frame = self.proj(pooled)  # [1, frame_dim]

        if hidden_state is not None:
            frame = frame + self.config.conditioning_scale * self.state_proj(hidden_state.unsqueeze(0))

        return frame.squeeze(0)  # [frame_dim]


class FBAMBaseline(nn.Module):
    """FBAM baseline without symbolic associative memory."""

    def __init__(self, config: Optional[FBAMBaselineConfig] = None) -> None:
        super().__init__()
        self.config = config or FBAMBaselineConfig()
        cfg = self.config

        self.frame_head = FBAMFrameHead(cfg)
        self.integrator = nn.LSTMCell(cfg.frame_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.node_vocab_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_single(self, query, device: torch.device) -> torch.Tensor:
        cfg = self.config
        h = torch.zeros(1, cfg.hidden_dim, device=device)
        c = torch.zeros(1, cfg.hidden_dim, device=device)

        for step in range(cfg.integrator_steps):
            prev_hidden = h.squeeze(0) if step > 0 else None
            frame = self.frame_head(query.natural_language, prev_hidden, device)
            frame = frame.unsqueeze(0)
            h, c = self.integrator(frame, (h, c))

        logits = self.classifier(self.dropout(h.squeeze(0)))
        return logits

    def forward_batch(self, queries: Sequence, device: torch.device) -> torch.Tensor:
        logits = [self.forward_single(query, device) for query in queries]
        return torch.stack(logits, dim=0)
