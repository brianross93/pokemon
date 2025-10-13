"""
Transformer baseline for multi-hop QA without explicit symbolic memory.

Matches SR-FBAM parameter budget (~1M params) to enable fair comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.models.sr_fbam import SimpleTokenizer


@dataclass
class TransformerBaselineConfig:
    token_vocab_size: int = 4096
    token_embedding_dim: int = 96
    num_layers: int = 4
    num_heads: int = 6
    ff_hidden_dim: int = 192
    dropout: float = 0.1
    node_vocab_size: int = 2048
    max_seq_len: int = 128


class TransformerBaseline(nn.Module):
    """Sequence-only transformer classifier baseline."""

    def __init__(self, config: Optional[TransformerBaselineConfig] = None) -> None:
        super().__init__()
        self.config = config or TransformerBaselineConfig()
        cfg = self.config

        self.tokenizer = SimpleTokenizer(cfg.token_vocab_size)
        self.token_embed = nn.Embedding(cfg.token_vocab_size, cfg.token_embedding_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.token_embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_embedding_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.classifier = nn.Linear(cfg.token_embedding_dim, cfg.node_vocab_size)

        nn.init.xavier_uniform_(self.token_embed.weight)
        nn.init.xavier_uniform_(self.pos_embed.weight)

    def encode_query(self, query_text: str, device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(query_text) or ["<blank>"]
        indices = torch.tensor(
            [self.tokenizer.token_to_index(tok) for tok in tokens[: self.config.max_seq_len]],
            dtype=torch.long,
            device=device,
        )
        return indices

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_indices: tensor [batch, seq_len]
        Returns:
            logits over node vocabulary: [batch, node_vocab_size]
        """
        seq_len = token_indices.size(1)
        positions = torch.arange(seq_len, device=token_indices.device).unsqueeze(0)
        embeddings = self.token_embed(token_indices) + self.pos_embed(positions)
        encoded = self.encoder(embeddings)
        pooled = encoded.mean(dim=1)  # average pooling
        logits = self.classifier(pooled)
        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
