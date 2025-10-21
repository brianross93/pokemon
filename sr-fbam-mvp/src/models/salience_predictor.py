"""Salience predictor used to decide whether memory writes are worthwhile."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SaliencePredictor(nn.Module):
    """
    Small MLP that scores whether the current symbol summary is likely to be reused.

    The network consumes a concatenation of the symbol embedding (mean pooled) and a
    lightweight context feature vector (cursor position, step progress, etc.) and
    returns a scalar logit that can be turned into a probability with ``sigmoid``.
    """

    def __init__(self, memory_dim: int, context_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = int(memory_dim) + int(context_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, symbol_summary: Tensor, context_features: Tensor) -> Tensor:
        """
        Args:
            symbol_summary: Tensor with shape [B, memory_dim]
            context_features: Tensor with shape [B, context_dim]

        Returns:
            Tensor with shape [B, 1] containing the salience logit.
        """
        if symbol_summary.dim() != 2:
            raise ValueError("symbol_summary must be [B, memory_dim]")
        if context_features.dim() != 2:
            raise ValueError("context_features must be [B, context_dim]")
        if symbol_summary.size(0) != context_features.size(0):
            raise ValueError("Batch size mismatch between symbol_summary and context_features")
        features = torch.cat([symbol_summary, context_features], dim=-1)
        return self.net(features)
