"""
Mamba-based variant of the FBAM baseline for code-editing episodes.

This swaps the row-wise Transformer encoder with a stack of Mamba state-space
layers while keeping the downstream LSTM controller identical to the original
FBAM baseline for a fair comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "mamba-ssm must be installed to use the MambaFBAM model. "
        "Install with `pip install mamba-ssm causal-conv1d`."
    ) from exc


@dataclass
class MambaFBAMConfig:
    """Configuration for the Mamba-based FBAM variant."""

    grid_height: int = 40
    grid_width: int = 120
    vocab_size: int = 256  # ASCII
    frame_d_model: int = 128
    frame_num_layers: int = 2
    frame_state_size: int = 16
    frame_expand_factor: int = 2
    frame_conv_kernel: int = 4
    frame_dropout: float = 0.1
    integrator_hidden_dim: int = 256
    integrator_dropout: float = 0.1
    action_embedding_dim: int = 32


class MambaBlock(nn.Module):
    """Single pre-norm residual block with a Mamba layer."""

    def __init__(
        self,
        d_model: int,
        state_size: int,
        expand_factor: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=state_size,
            expand=expand_factor,
            d_conv=conv_kernel_size,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class MambaFrameEncoder(nn.Module):
    """Encodes a 40x120 character grid using stacked Mamba layers."""

    def __init__(self, config: MambaFBAMConfig) -> None:
        super().__init__()
        self.config = config
        seq_len = config.grid_height * config.grid_width
        self.embedding = nn.Embedding(config.vocab_size, config.frame_d_model)
        self.positional = nn.Embedding(seq_len, config.frame_d_model)
        self.dropout = nn.Dropout(config.frame_dropout)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=config.frame_d_model,
                    state_size=config.frame_state_size,
                    expand_factor=config.frame_expand_factor,
                    conv_kernel_size=config.frame_conv_kernel,
                    dropout=config.frame_dropout,
                )
                for _ in range(config.frame_num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.frame_d_model)

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

        tokens = grid.long().view(batch_size, -1)
        seq_len = tokens.size(1)

        positions = torch.arange(seq_len, device=grid.device).unsqueeze(0)
        embeddings = self.embedding(tokens) + self.positional(positions)
        x = self.dropout(embeddings)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        pooled = x.mean(dim=1)
        return pooled


class MambaFBAMAgent(nn.Module):
    """FBAM agent that relies on a Mamba frame encoder for dense processing."""

    def __init__(self, num_actions: int, config: Optional[MambaFBAMConfig] = None) -> None:
        super().__init__()
        self.config = config or MambaFBAMConfig()
        self.num_actions = num_actions

        self.frame_encoder = MambaFrameEncoder(self.config)
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

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        device = device or self.device
        h = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        c = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        return h, c

    def forward_episode(self, frames: Sequence[Tensor], actions: Tensor) -> Tensor:
        """
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

    @torch.no_grad()
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

