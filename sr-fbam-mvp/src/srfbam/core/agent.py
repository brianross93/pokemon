"""
Reusable SR-FBAM neural core shared by multiple task adapters.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from src.models.sr_fbam_code import (
    SRFBAMCodeConfig,
    TransformerSymbolExtractor,
    MemoryCrossAttention,
    SymbolicMemory,
)
from src.models.extraction_gate import ExtractionGate

from .types import EncodedFrame, SrfbamStepSummary, ensure_tensor_device


class SRFBAMCore(nn.Module):
    """
    Task-agnostic SR-FBAM module.

    This class bundles the Transformer symbol extractor, symbolic memory,
    learned gating logic, and recurrent integrator. Domain-specific agents
    can compose it with environment adapters and policy heads.
    """

    def __init__(self, config: Optional[SRFBAMCodeConfig] = None, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.config = config or SRFBAMCodeConfig.large()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = SymbolicMemory(self.config)

        patch_rows = self.config.extractor_patch_rows
        patch_cols = self.config.extractor_patch_cols
        sequence_length = (
            torch.div(self.config.frame_height + patch_rows - 1, patch_rows, rounding_mode="floor")
            * torch.div(self.config.frame_width + patch_cols - 1, patch_cols, rounding_mode="floor")
        )

        self.symbol_extractor = TransformerSymbolExtractor(
            vocab_size=self.config.extractor_vocab_size,
            sequence_length=int(sequence_length),
            d_model=self.config.extractor_d_model,
            memory_dim=self.config.memory_embedding_dim,
            num_symbols=self.config.extractor_num_symbols,
            num_layers=self.config.extractor_num_layers,
            num_heads=self.config.extractor_num_heads,
            feedforward_dim=self.config.extractor_feedforward_dim,
            dropout=self.config.dropout,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
        )

        self.memory_cross_attn = MemoryCrossAttention(
            embed_dim=self.config.memory_embedding_dim,
            num_heads=self.config.memory_attention_heads,
            dropout=self.config.dropout,
        )
        self.extraction_gate = ExtractionGate(
            lstm_hidden_dim=self.config.integrator_hidden_dim,
            memory_embedding_dim=self.config.memory_embedding_dim,
            hidden_dim=self.config.salience_hidden_dim,
        )

        self.gate_temperature = self.config.gate_temperature
        self.gate_threshold = self.config.gate_threshold

        self.file_embed = nn.Embedding(self.config.token_vocab_size, self.config.file_embedding_dim)
        self.action_embed = nn.Embedding(
            self.config.policy_action_vocab_size, self.config.policy_embedding_dim
        )
        self.numeric_proj = nn.Linear(8, self.config.numeric_hidden_dim)
        self.symbol_proj = nn.Linear(self.config.memory_embedding_dim, self.config.symbol_feature_dim)

        input_dim = (
            self.config.file_embedding_dim
            + self.config.symbol_feature_dim
            + self.config.policy_embedding_dim
            + self.config.numeric_hidden_dim
        )
        self.integrator = nn.LSTMCell(input_dim, self.config.integrator_hidden_dim)
        self.dropout = nn.Dropout(self.config.dropout)

        self.reset_state()
        self.to(self.device)

    def reset_state(self) -> None:
        """Reset hidden state and symbolic memory."""
        self.memory_state = self.memory.new_state()
        self.h = torch.zeros(1, self.config.integrator_hidden_dim, device=self.device)
        self.c = torch.zeros(1, self.config.integrator_hidden_dim, device=self.device)
        self.last_action_index = torch.tensor([0], device=self.device)

    def set_last_action_index(self, action_index: int) -> None:
        """Record the most recent discrete action index."""
        safe_index = max(0, min(int(action_index), self.action_embed.num_embeddings - 1))
        self.last_action_index = torch.tensor([safe_index], device=self.device)

    def _hash_context(self, context_key: str) -> int:
        return hash(context_key) % self.config.token_vocab_size

    def _numeric_embed(self, numeric_features: Tensor) -> Tensor:
        numeric_features = ensure_tensor_device(numeric_features, self.device)
        return self.numeric_proj(numeric_features)

    def encode_step(self, encoded: EncodedFrame) -> SrfbamStepSummary:
        """
        Run the SR-FBAM core on a single encoded frame.

        Returns a `SrfbamStepSummary` containing the embeddings and gate
        diagnostics required by downstream policies/telemetry.
        """

        frame_tensor = ensure_tensor_device(encoded.grid, self.device)
        numeric_features = ensure_tensor_device(encoded.features, self.device)
        context_key = encoded.context_key

        file_idx = torch.tensor([self._hash_context(context_key)], device=self.device)
        file_embed = self.file_embed(file_idx)
        action_embed = self.action_embed(self.last_action_index)
        numeric_embed = self._numeric_embed(numeric_features)

        frame_hash = hash(frame_tensor.detach().cpu().numpy().tobytes())
        summary_vector = self.memory_state.file_summary.get(context_key)
        cached_symbols = self.memory_state.frame_cache.get(frame_hash)
        frame_changed = frame_hash != self.memory_state.last_frame_hash

        if cached_symbols is not None:
            current_symbols = ensure_tensor_device(cached_symbols, self.device)
            self.memory_state.cache_hits += 1
            gate_decision = "CACHE_HIT"
        else:
            if summary_vector is not None:
                summary_vector = ensure_tensor_device(summary_vector, self.device).unsqueeze(0)
            else:
                summary_vector = torch.zeros(1, self.config.memory_embedding_dim, device=self.device)

            gate_logit = self.extraction_gate(self.h, summary_vector, frame_changed)
            gate_prob = torch.sigmoid(gate_logit)
            should_extract = gate_prob > self.gate_threshold
            if should_extract.item():
                extracted, _, _ = self.symbol_extractor(frame_tensor)
                if extracted.numel() == 0:
                    extracted = torch.zeros(1, self.config.memory_embedding_dim, device=self.device)
                current_symbols = extracted
                self.memory_state.extract_count += 1
                gate_decision = "EXTRACT"
                self.memory_state.frame_cache[frame_hash] = current_symbols.detach().cpu()
                while len(self.memory_state.frame_cache) > self.config.frame_cache_size:
                    self.memory_state.frame_cache.popitem(last=False)
            else:
                current_symbols = summary_vector
                self.memory_state.reuse_count += 1
                gate_decision = "REUSE"

        current_batch = current_symbols.unsqueeze(0)
        memory_embeddings = self.memory.retrieve(self.memory_state, context_key, self.device)
        memory_batch = memory_embeddings.unsqueeze(0)
        attended = self.memory_cross_attn(current_batch, memory_batch)
        pooled = attended.mean(dim=1)
        symbol_embed = self.symbol_proj(pooled)

        lstm_input = torch.cat([file_embed, symbol_embed, action_embed, numeric_embed], dim=1)
        self.h, self.c = self.integrator(lstm_input, (self.h, self.c))
        embedding = self.dropout(self.h)

        self.memory.update(self.memory_state, context_key, current_symbols, "MOVE", None)
        summary_update = current_symbols.mean(dim=0).detach().cpu()
        self.memory_state.file_summary[context_key] = summary_update
        self.memory_state.last_frame_hash = frame_hash
        self.last_action_index = torch.tensor([0], device=self.device)

        steps = max(self.memory_state.step_index, 1)
        gate_stats = {
            "decision": gate_decision,
            "cache_hits": self.memory_state.cache_hits,
            "reuse": self.memory_state.reuse_count,
            "extract": self.memory_state.extract_count,
            "cache_hit_rate": self.memory_state.cache_hits / steps,
            "reuse_rate": self.memory_state.reuse_count / steps,
            "extract_rate": self.memory_state.extract_count / steps,
        }

        return SrfbamStepSummary(
            embedding=embedding.squeeze(0).detach(),
            symbol_embedding=symbol_embed.squeeze(0).detach(),
            numeric_features=numeric_features.detach().cpu(),
            gate_stats=gate_stats,
            context_key=context_key,
        )
