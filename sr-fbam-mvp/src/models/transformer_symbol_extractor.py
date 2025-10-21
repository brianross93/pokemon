"""
Transformer-based symbol extractor for SR-FBAM code agent.

Given a 40x120 ASCII grid representing the current frame, this module learns
to produce a compact set of salient symbol embeddings without relying on
hand-crafted regular expressions.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TransformerSymbolExtractor(nn.Module):
    """
    Learnable symbol extractor that ranks character positions by importance and
    projects the most informative ones into the memory embedding space.
    """

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        d_model: int,
        memory_dim: int,
        num_symbols: int,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        patch_rows: int = 4,
        patch_cols: int = 4,
        pad_value: int = 32,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.num_symbols = num_symbols
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.pad_value = pad_value

        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(sequence_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.importance_head = nn.Linear(d_model, 1)
        self.token_projector = nn.Linear(d_model, memory_dim)

        self.register_buffer(
            "_positions",
            torch.arange(sequence_length, dtype=torch.long),
            persistent=False,
        )

    def forward(self, frame_tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            frame_tensor: [H, W] uint8 tensor representing the current frame.

        Returns:
            symbol_embeddings: [K, memory_dim] embeddings of top-K positions.
            top_indices: [K] flattened indices (0..sequence_length-1).
            top_scores: [K] importance scores for each selected position.
        """
        if frame_tensor.dim() != 2:
            raise ValueError(f"Expected 2D frame tensor, got shape {frame_tensor.shape}")

        device = frame_tensor.device
        frame_long = frame_tensor.to(dtype=torch.long)

        patches = self._patchify(frame_long)
        patch_embeds = self.char_embedding(patches).mean(dim=1)

        seq_len = patch_embeds.size(0)
        pos_indices = self._positions[:seq_len].to(device)
        pos_embeds = self.pos_embedding(pos_indices)
        token_features = patch_embeds + pos_embeds

        encoded = self.transformer(token_features.unsqueeze(0)).squeeze(0)

        importance_scores = self.importance_head(encoded).squeeze(-1)

        k = min(self.num_symbols, importance_scores.shape[0])
        if k == 0:
            return (
                torch.zeros(0, self.token_projector.out_features, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, device=device),
            )

        top_scores, top_indices = torch.topk(importance_scores, k=k, largest=True)
        important_tokens = encoded.index_select(0, top_indices)
        symbol_embeddings = self.token_projector(important_tokens)

        return symbol_embeddings, top_indices, top_scores

    def _patchify(self, frame_long: Tensor) -> Tensor:
        h, w = frame_long.shape
        pr, pc = self.patch_rows, self.patch_cols
        pad_h = (pr - h % pr) % pr
        pad_w = (pc - w % pc) % pc
        if pad_h or pad_w:
            frame_long = F.pad(frame_long, (0, pad_w, 0, pad_h), value=self.pad_value)
        new_h, new_w = frame_long.shape
        num_h = new_h // pr
        num_w = new_w // pc
        patches = frame_long.view(num_h, pr, num_w, pc).permute(0, 2, 1, 3).reshape(num_h * num_w, pr * pc)
        return patches
