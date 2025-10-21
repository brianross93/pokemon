"""
Cross-attention module that relates current frame symbols to episodic memory.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class MemoryCrossAttention(nn.Module):
    """
    Multi-head attention block with residual connection and layer normalization.

    The module attends current frame symbols (queries) to memory symbols
    (keys/values). When the memory is empty, it returns a normalized version of
    the current symbols to preserve scale without introducing noise.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        current_symbols: Tensor,
        memory_symbols: Tensor,
        return_weights: bool = False,
    ) -> Tensor | Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            current_symbols: [B, S, D] tensor of current frame symbol embeddings.
            memory_symbols: [B, M, D] tensor of memory embeddings.
            return_weights: Whether to return attention weights.

        Returns:
            output: [B, S, D] tensor after cross-attention and residual normalization.
            attn_weights: Optional attention weights [B, S, M].
        """
        if current_symbols.size(1) == 0:
            return (current_symbols, None) if return_weights else current_symbols

        if memory_symbols.size(1) == 0:
            normalized = self.norm(current_symbols)
            return (normalized, None) if return_weights else normalized

        attended, attn_weights = self.cross_attn(
            query=current_symbols,
            key=memory_symbols,
            value=memory_symbols,
            need_weights=return_weights,
        )
        output = self.norm(current_symbols + attended)
        return (output, attn_weights) if return_weights else output
