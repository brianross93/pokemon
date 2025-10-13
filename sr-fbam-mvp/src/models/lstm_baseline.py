"""
Pure LSTM baseline for multi-hop QA without graph structure.

Matches the SR-FBAM parameter budget (~1M params) to enable fair comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from src.models.sr_fbam import SimpleTokenizer


@dataclass
class LSTMBaselineConfig:
    token_vocab_size: int = 4096
    token_embedding_dim: int = 96
    lstm_hidden_dim: int = 150
    lstm_num_layers: int = 2
    dropout: float = 0.1
    node_vocab_size: int = 2048


class LSTMBaseline(nn.Module):
    """Sequence-only baseline: embedding -> stacked LSTM -> classifier."""

    def __init__(self, config: Optional[LSTMBaselineConfig] = None) -> None:
        super().__init__()
        self.config = config or LSTMBaselineConfig()
        cfg = self.config

        self.tokenizer = SimpleTokenizer(cfg.token_vocab_size)
        self.token_embed = nn.Embedding(cfg.token_vocab_size, cfg.token_embedding_dim)

        self.lstm = nn.LSTM(
            input_size=cfg.token_embedding_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.lstm_hidden_dim, cfg.node_vocab_size),
        )

        nn.init.xavier_uniform_(self.token_embed.weight)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_indices: tensor shape [batch, seq_len]
        Returns:
            logits: tensor shape [batch, node_vocab_size]
        """
        embeddings = self.token_embed(token_indices)
        _, (h_n, _) = self.lstm(embeddings)
        final_hidden = h_n[-1]  # [batch, hidden_dim]
        logits = self.classifier(final_hidden)
        return logits

    def encode_query(self, query_text: str, device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(query_text) or ["<blank>"]
        indices = torch.tensor(
            [self.tokenizer.token_to_index(tok) for tok in tokens],
            dtype=torch.long,
            device=device,
        )
        return indices

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
