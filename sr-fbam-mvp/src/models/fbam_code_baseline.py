"""
Pure FBAM baseline tailored for code-editing frame/action episodes.

This adapts the architecture described in the FBAM paper:
- Intra-frame Transformer encoder over 40x120 character grids
- Inter-frame LSTMCell integrator
- Action classification head predicting discrete editor actions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
from torch import Tensor, nn


@dataclass
class PureFBAMCodeConfig:
    """Configuration for the pure FBAM baseline."""

    grid_height: int = 40
    grid_width: int = 120
    vocab_size: int = 256  # ASCII
    frame_d_model: int = 128
    frame_nhead: int = 4
    frame_num_layers: int = 2
    frame_ff_dim: int = 256
    frame_dropout: float = 0.1
    integrator_hidden_dim: int = 256
    integrator_dropout: float = 0.1
    action_embedding_dim: int = 32

    @classmethod
    def small(cls) -> "PureFBAMCodeConfig":
        """Baseline configuration (~0.75M parameters with 50 actions)."""
        return cls()

    @classmethod
    def large(cls) -> "PureFBAMCodeConfig":
        """Scaled-up configuration (~2.7M parameters with 50 actions)."""
        return cls(
            frame_d_model=192,
            frame_nhead=6,
            frame_num_layers=4,
            frame_ff_dim=448,
            integrator_hidden_dim=416,
            action_embedding_dim=56,
        )

    @classmethod
    def preset(cls, name: str) -> "PureFBAMCodeConfig":
        lookup = {
            "small": cls.small,
            "large": cls.large,
        }
        try:
            return lookup[name.lower()]()
        except KeyError as exc:  # pragma: no cover - simple guard
            raise ValueError(f"Unknown PureFBAMCodeConfig preset '{name}'") from exc


class FrameTransformer(nn.Module):
    """Encodes a 40x120 character grid into a dense embedding."""

    def __init__(self, config: PureFBAMCodeConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.frame_d_model)
        self.positional = nn.Embedding(config.grid_height, config.frame_d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.frame_d_model,
            nhead=config.frame_nhead,
            dim_feedforward=config.frame_ff_dim,
            dropout=config.frame_dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.frame_num_layers,
        )
        self.layer_norm = nn.LayerNorm(config.frame_d_model)

    def forward(self, grid: Tensor) -> Tensor:
        """
        Args:
            grid: uint8/long tensor of shape [H, W] or [B, H, W]
        Returns:
            embeddings: tensor [B, d_model]
        """
        cfg = self.config
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)
        batch_size, height, width = grid.shape
        assert height == cfg.grid_height and width == cfg.grid_width, (
            f"Expected grid {cfg.grid_height}x{cfg.grid_width}, got {height}x{width}"
        )
        tokens = grid.long()
        token_embeddings = self.embedding(tokens)  # [B, H, W, d_model]
        row_embeddings = token_embeddings.mean(dim=2)  # [B, H, d_model]
        positions = torch.arange(height, device=grid.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = row_embeddings + self.positional(positions)
        encoded = self.encoder(embeddings)
        pooled = encoded.mean(dim=1)
        return self.layer_norm(pooled)


class PureFBAMCodeAgent(nn.Module):
    """Pure FBAM model for code frame/action prediction."""

    def __init__(self, num_actions: int, config: Optional[PureFBAMCodeConfig] = None) -> None:
        super().__init__()
        self.config = config or PureFBAMCodeConfig()
        self.num_actions = num_actions

        self.frame_encoder = FrameTransformer(self.config)
        self.action_embed = nn.Embedding(num_actions + 1, self.config.action_embedding_dim)
        self.integrator = nn.LSTMCell(
            self.config.frame_d_model + self.config.action_embedding_dim,
            self.config.integrator_hidden_dim,
        )
        self.dropout = nn.Dropout(self.config.integrator_dropout)
        self.classifier = nn.Linear(self.config.integrator_hidden_dim, num_actions)
        self.start_action_index = num_actions

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Tensor:
        device = device or self.device
        h = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        c = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        return h, c

    def forward_episode(self, frames: Sequence[Tensor], actions: Tensor) -> Tensor:
        """
        Run the model on a single episode.

        Args:
            frames: list of tensors [H, W]
            actions: tensor of action indices [T] (teacher forcing)
        Returns:
            logits: tensor [T, num_actions]
        """
        device = self.device
        h, c = self.init_hidden(device=device)
        logits: List[Tensor] = []

        actions = actions.to(device)
        prev_actions = torch.cat(
            [
                torch.tensor([self.start_action_index], dtype=torch.long, device=device),
                actions[:-1],
            ]
        )

        for step, frame in enumerate(frames):
            frame_tensor = frame.to(device=device)
            embedding = self.frame_encoder(frame_tensor).view(1, -1)
            prev_idx = prev_actions[step].view(1)
            action_embedding = self.action_embed(prev_idx)
            lstm_input = torch.cat([embedding, action_embedding], dim=1)
            h, c = self.integrator(lstm_input, (h, c))
            step_logits = self.classifier(self.dropout(h))
            logits.append(step_logits)

        return torch.cat(logits, dim=0)

    def predict_actions(self, frames: Sequence[Tensor]) -> Tensor:
        """
        Autoregressive inference without teacher forcing.
        """
        device = self.device
        h, c = self.init_hidden(device=device)
        prev_idx = torch.tensor([self.start_action_index], dtype=torch.long, device=device)
        predictions: List[Tensor] = []

        for frame in frames:
            frame_tensor = frame.to(device=device)
            embedding = self.frame_encoder(frame_tensor).view(1, -1)
            action_embedding = self.action_embed(prev_idx)
            lstm_input = torch.cat([embedding, action_embedding], dim=1)
            h, c = self.integrator(lstm_input, (h, c))
            step_logits = self.classifier(self.dropout(h))
            pred = step_logits.argmax(dim=1)
            predictions.append(pred)
            prev_idx = pred

        return torch.cat(predictions, dim=0)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())
