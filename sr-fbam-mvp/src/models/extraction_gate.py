"""
Neural gate that decides when to run the expensive transformer-based
symbol extractor versus reusing cached summaries.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class ExtractionGate(nn.Module):
    """
    Learns to balance costly symbol extraction with inexpensive reuse.

    Inputs:
        - LSTM hidden state (controller context)
        - File-level memory summary (what we have seen for this file)
        - Binary flag indicating whether the raw frame changed

    Output:
        - Logit for the decision "extract fresh symbols"
    """

    def __init__(
        self,
        lstm_hidden_dim: int,
        memory_embedding_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        input_dim = lstm_hidden_dim + memory_embedding_dim + 1  # frame_changed flag
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        lstm_hidden: Tensor,
        file_memory_summary: Tensor,
        frame_changed: bool,
    ) -> Tensor:
        """
        Args:
            lstm_hidden: [1, lstm_hidden_dim]
            file_memory_summary: [1, memory_embedding_dim]
            frame_changed: bool

        Returns:
            gate_logit: [1, 1]
        """
        device: torch.device = lstm_hidden.device
        frame_changed_tensor = torch.tensor(
            [[1.0 if frame_changed else 0.0]], device=device, dtype=lstm_hidden.dtype
        )
        combined = torch.cat(
            [lstm_hidden, file_memory_summary, frame_changed_tensor],
            dim=1,
        )
        features = self.input_proj(combined)
        features = self.hidden(features)
        gate_logit = self.gate_head(features)
        return gate_logit
