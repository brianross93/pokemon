"""
FBAM baseline augmented with a soft attention-based external memory.

Acts as an intermediary between pure FBAM (no external memory) and SR-FBAM
(discrete symbolic memory). Memory is represented as learnable slots updated
with differentiable write operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List

import torch
from torch import Tensor, nn

from .fbam_code_baseline import FrameTransformer, PureFBAMCodeConfig


@dataclass
class FBAMSoftMemoryConfig(PureFBAMCodeConfig):
    """Configuration for FBAM with soft external memory."""

    memory_slots: int = 128
    memory_dim: int = 128
    action_embedding_dim: int = 32


class FBAMSoftMemoryAgent(nn.Module):
    """FBAM variant with soft attention-based external memory."""

    def __init__(self, num_actions: int, config: Optional[FBAMSoftMemoryConfig] = None) -> None:
        super().__init__()
        self.config = config or FBAMSoftMemoryConfig()
        self.num_actions = num_actions

        self.frame_encoder = FrameTransformer(self.config)
        concat_dim = (
            self.config.frame_d_model
            + self.config.memory_dim
            + self.config.action_embedding_dim
        )

        self.action_embed = nn.Embedding(num_actions + 1, self.config.action_embedding_dim)
        self.integrator = nn.LSTMCell(concat_dim, self.config.integrator_hidden_dim)
        self.classifier = nn.Linear(self.config.integrator_hidden_dim, num_actions)
        self.dropout = nn.Dropout(self.config.integrator_dropout)

        # Memory parameters
        self.memory_init = nn.Parameter(
            torch.randn(self.config.memory_slots, self.config.memory_dim) * 0.02
        )
        self.query_proj = nn.Linear(self.config.integrator_hidden_dim, self.config.memory_dim)
        self.key_proj = nn.Linear(self.config.memory_dim, self.config.memory_dim)
        self.value_proj = nn.Linear(self.config.memory_dim, self.config.memory_dim)
        self.write_proj = nn.Linear(self.config.integrator_hidden_dim, self.config.memory_dim)
        self.write_gate = nn.Linear(self.config.integrator_hidden_dim, 1)

        self.start_action_index = num_actions

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> tuple[Tensor, Tensor]:
        device = device or self.device
        h = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        c = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        return h, c

    def init_memory(self, device: torch.device) -> dict[str, Tensor]:
        """Return fresh memory key/value buffers for an episode."""
        init = self.memory_init.to(device)
        keys = self.key_proj(init)
        values = self.value_proj(init)
        return {
            "keys": keys.clone(),
            "values": values.clone(),
        }

    def _read_memory(self, h: Tensor, memory: dict[str, Tensor]) -> Tensor:
        query = self.query_proj(h)  # [B, memory_dim]
        keys = memory["keys"]  # [S, memory_dim]
        values = memory["values"]  # [S, memory_dim]
        attn_logits = torch.matmul(query, keys.t()) / (self.config.memory_dim ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, S]
        retrieved = torch.matmul(attn_weights, values)
        memory["last_attention"] = attn_weights
        return retrieved

    def _write_memory(self, h: Tensor, memory: dict[str, Tensor]) -> None:
        attn = memory.get("last_attention")
        if attn is None:
            return
        gate = torch.sigmoid(self.write_gate(h))  # [B, 1]
        write_vec = self.write_proj(h)  # [B, memory_dim]
        # Single-episode processing -> batch size 1
        weight = (gate * attn).squeeze(0)  # [S]
        write = write_vec.squeeze(0)  # [D]
        weight = weight.unsqueeze(1)  # [S, 1]

        memory["values"] = memory["values"] * (1 - weight) + weight * write
        memory["keys"] = memory["keys"] * (1 - weight) + weight * write

    def forward_episode(self, frames: Sequence[Tensor], actions: Tensor) -> Tensor:
        device = self.device
        h, c = self.init_hidden(device=device)
        memory = self.init_memory(device)

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
            frame_embed = self.frame_encoder(frame_tensor).view(1, -1)
            if step == 0:
                retrieved = torch.zeros(1, self.config.memory_dim, device=device)
            else:
                retrieved = self._read_memory(h, memory)

            prev_idx = prev_actions[step].view(1)
            prev_action_embed = self.action_embed(prev_idx)
            lstm_input = torch.cat([frame_embed, retrieved, prev_action_embed], dim=1)
            h, c = self.integrator(lstm_input, (h, c))
            self._write_memory(h, memory)

            logit = self.classifier(self.dropout(h))
            logits.append(logit)

        return torch.cat(logits, dim=0)

    def predict_actions(self, frames: Sequence[Tensor]) -> Tensor:
        device = self.device
        h, c = self.init_hidden(device=device)
        memory = self.init_memory(device)
        prev_idx = torch.tensor([self.start_action_index], dtype=torch.long, device=device)
        preds: List[Tensor] = []

        for frame in frames:
            frame_tensor = frame.to(device=device)
            frame_embed = self.frame_encoder(frame_tensor).view(1, -1)
            if preds:
                retrieved = self._read_memory(h, memory)
            else:
                retrieved = torch.zeros(1, self.config.memory_dim, device=device)
            prev_action_embed = self.action_embed(prev_idx)
            lstm_input = torch.cat([frame_embed, retrieved, prev_action_embed], dim=1)
            h, c = self.integrator(lstm_input, (h, c))
            self._write_memory(h, memory)
            logit = self.classifier(self.dropout(h))
            pred = logit.argmax(dim=1)
            preds.append(pred)
            prev_idx = pred

        return torch.cat(preds, dim=0)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
