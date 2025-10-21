"""
SR-FBAM prototype for code-editing episodes with a unified policy surface.

The decoder emits a single sequence of tokens that interleave memory operations
(`ENCODE`, `ASSOC`, `FOLLOW`, `WRITE`, `HALT`) with downstream environment
actions. This removes the implicit two-controller split and lets the policy
learn when memory lookups amortise the cost of the encoder.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import os

import torch
from torch import Tensor, nn

from src.data.frame_dataset import FrameActionEpisode, FrameActionStep
from src.models.memory_cross_attention import MemoryCrossAttention
from src.models.salience_predictor import SaliencePredictor
from src.models.transformer_symbol_extractor import TransformerSymbolExtractor


@dataclass
class SRFBAMCodeConfig:
    token_vocab_size: int = 8192
    file_embedding_dim: int = 86
    memory_embedding_dim: int = 144
    symbol_feature_dim: int = 216
    policy_embedding_dim: int = 34
    policy_action_vocab_size: int = 16
    numeric_hidden_dim: int = 20
    integrator_hidden_dim: int = 332
    dropout: float = 0.1
    max_tokens_per_file: int = 144
    max_global_tokens: int = 288
    frame_height: int = 40
    frame_width: int = 120
    extractor_vocab_size: int = 256
    extractor_d_model: int = 168
    extractor_num_symbols: int = 36
    extractor_num_layers: int = 2
    extractor_num_heads: int = 8
    extractor_feedforward_dim: int = 480
    memory_attention_heads: int = 8
    extractor_patch_rows: int = 4
    extractor_patch_cols: int = 4
    summary_decay: float = 0.1
    frame_cache_size: int = 512
    enable_gate: bool = True  # kept for CLI/backwards compatibility
    gate_mode: str = "unified"
    gate_temperature: float = 1.0
    gate_threshold: float = 0.5
    enable_memory: bool = True
    enable_salience: bool = True
    salience_hidden_dim: int = 64
    salience_context_dim: int = 8  # matches numeric features incl. gate flags
    salience_threshold: float = 0.4
    encode_latency_ms: float = 40.0
    assoc_latency_ms: float = 5.0
    follow_latency_ms: float = 6.5
    write_latency_ms: float = 2.0
    halt_latency_ms: float = 90.0
    skip_latency_ms: float = 1.5

    @classmethod
    def large(cls) -> "SRFBAMCodeConfig":
        return cls()

    @classmethod
    def small(cls) -> "SRFBAMCodeConfig":
        return cls(
            file_embedding_dim=44,
            memory_embedding_dim=76,
            symbol_feature_dim=104,
            policy_embedding_dim=20,
            numeric_hidden_dim=10,
            integrator_hidden_dim=164,
            max_tokens_per_file=104,
            max_global_tokens=208,
            extractor_d_model=64,
            extractor_num_symbols=14,
            extractor_num_layers=1,
            extractor_num_heads=4,
            extractor_feedforward_dim=128,
            extractor_patch_rows=5,
            extractor_patch_cols=6,
            memory_attention_heads=4,
        )

    @classmethod
    def preset(cls, name: str) -> "SRFBAMCodeConfig":
        lookup = {"small": cls.small, "large": cls.large}
        try:
            return lookup[name.lower()]()
        except KeyError as exc:  # pragma: no cover - simple guard
            raise ValueError(f"Unknown SRFBAMCodeConfig preset '{name}'") from exc


class MemoryOp(IntEnum):
    ENCODE = 0
    ASSOC = 1
    FOLLOW = 2
    WRITE = 3
    HALT = 4


@dataclass(frozen=True)
class PolicyVocabulary:
    memory_ops: Sequence[MemoryOp]
    num_actions: int

    @property
    def size(self) -> int:
        return len(self.memory_ops) + self.num_actions

    @property
    def action_offset(self) -> int:
        return len(self.memory_ops)

    def memory_index(self, op: MemoryOp) -> int:
        return int(op)

    def action_index(self, action_idx: int) -> int:
        if not (0 <= action_idx < self.num_actions):
            raise ValueError(f"action_idx {action_idx} outside 0..{self.num_actions-1}")
        return self.action_offset + action_idx

    def is_memory_index(self, token_idx: int) -> bool:
        return 0 <= token_idx < len(self.memory_ops)

    def decode(self, token_idx: int) -> tuple[str, int]:
        if self.is_memory_index(token_idx):
            op = self.memory_ops[token_idx]
            return ("memory", int(op))
        action_idx = token_idx - self.action_offset
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"token_idx {token_idx} outside vocabulary")
        return ("action", action_idx)


@dataclass
class PolicyTokenMeta:
    step_index: int
    phase: str  # "memory" or "action"
    memory_op: Optional[MemoryOp] = None
    action_index: Optional[int] = None


@dataclass
class EpisodeRollout:
    logits: Tensor  # [num_tokens, vocab_size]
    executed_tokens: List[int]
    memory_positions: List[int]
    action_positions: List[int]
    metadata: List[PolicyTokenMeta]
    vocabulary: PolicyVocabulary

    def memory_logits(self) -> Tensor:
        if not self.memory_positions:
            return torch.empty(0, self.logits.size(1), device=self.logits.device)
        return self.logits[self.memory_positions]

    def action_logits(self) -> Tensor:
        if not self.action_positions:
            return torch.empty(0, self.logits.size(1), device=self.logits.device)
        return self.logits[self.action_positions]

    def action_logits_env(self) -> Tensor:
        if not self.action_positions:
            return torch.empty(0, self.vocabulary.num_actions, device=self.logits.device)
        logits = self.logits[self.action_positions]
        return logits[:, self.vocabulary.action_offset :]


@dataclass
class SymbolicMemoryState:
    file_embeddings: Dict[str, Deque[Tensor]]
    global_embeddings: Deque[Tensor]
    frame_cache: "OrderedDict[int, Tensor]"
    file_summary: Dict[str, Tensor]
    memory_op_counts: Dict[MemoryOp, int]
    total_latency_ms: float = 0.0
    insertions: int = 0
    deletions: int = 0
    tests_runs: int = 0
    tests_passed: int = 0
    step_index: int = 0
    cache_hits: int = 0
    last_frame_hash: Optional[int] = None
    reuse_count: int = 0
    extract_count: int = 0
    halt_count: int = 0
    salience_attempts: int = 0
    salience_committed: int = 0
    salience_sum: float = 0.0


class SymbolicMemory(nn.Module):
    """Stores continuous symbol embeddings per file and provides lookups."""

    def __init__(self, config: SRFBAMCodeConfig) -> None:
        super().__init__()
        self.config = config
        self.enabled = bool(config.enable_memory)
        self.salience_threshold = float(config.salience_threshold)
        self.context_dim = int(config.salience_context_dim)

        if self.enabled and config.enable_salience:
            self.salience_predictor = SaliencePredictor(
                memory_dim=config.memory_embedding_dim,
                context_dim=config.salience_context_dim,
                hidden_dim=config.salience_hidden_dim,
            )
        else:
            self.salience_predictor = None

    def new_state(self) -> SymbolicMemoryState:
        return SymbolicMemoryState(
            file_embeddings=defaultdict(lambda: deque(maxlen=self.config.max_tokens_per_file)),
            global_embeddings=deque(maxlen=self.config.max_global_tokens),
            frame_cache=OrderedDict(),
            file_summary={},
            memory_op_counts=defaultdict(int),
        )

    def update(
        self,
        state: SymbolicMemoryState,
        file_name: str,
        symbol_embeddings: Tensor,
        action_type: str,
        terminal_output: Optional[str],
        salience_context: Optional[Tensor] = None,
        force_write: bool = False,
    ) -> Dict[str, Any]:
        """
        Update counters and optionally commit embeddings to memory.

        Returns a metrics dictionary with ``write_committed`` and ``salience_probability``.
        """
        metrics: Dict[str, Any] = {
            "write_committed": False,
            "salience_probability": None,
        }

        if action_type == "INSERT_LINE":
            state.insertions += 1
        elif action_type == "DELETE_LINE":
            state.deletions += 1
        elif action_type == "RUN_TESTS":
            state.tests_runs += 1
            if terminal_output and "PASSED" in terminal_output:
                state.tests_passed += 1

        if symbol_embeddings is None or symbol_embeddings.numel() == 0:
            state.step_index += 1
            return metrics

        if not self.enabled:
            state.step_index += 1
            return metrics

        if salience_context is None:
            salience_context = torch.zeros(
                (1, self.context_dim),
                dtype=symbol_embeddings.dtype,
                device=symbol_embeddings.device,
            )
        elif salience_context.dim() != 2 or salience_context.size(1) != self.context_dim:
            raise ValueError(
                f"salience_context expected shape [1,{self.context_dim}] "
                f"but received {tuple(salience_context.shape)}"
            )

        summary = symbol_embeddings.mean(dim=0, keepdim=True)
        should_write = True
        salience_prob_value: Optional[float] = None

        if self.salience_predictor is not None:
            salience_logit = self.salience_predictor(summary, salience_context)
            salience_prob = torch.sigmoid(salience_logit)
            salience_prob_value = float(salience_prob.item())
            should_write = bool(force_write or salience_prob_value >= self.salience_threshold)
            state.salience_sum += salience_prob_value
        else:
            should_write = True or force_write

        state.salience_attempts += 1
        metrics["salience_probability"] = salience_prob_value

        if should_write or force_write:
            metrics["write_committed"] = True
            state.salience_committed += 1
            file_deque = state.file_embeddings[file_name]
            for embed in symbol_embeddings:
                stored = embed.detach().cpu()
                file_deque.append(stored)
                state.global_embeddings.append(stored)

        state.step_index += 1
        return metrics

    def retrieve(self, state: SymbolicMemoryState, file_name: str, device: torch.device) -> Tensor:
        """Return stacked embeddings for the file or fall back to global memory."""
        if not self.enabled:
            return torch.zeros(0, self.config.memory_embedding_dim, device=device)

        file_deque = state.file_embeddings.get(file_name, None)
        if file_deque and len(file_deque):
            return self._stack_embeddings(file_deque, device)
        if len(state.global_embeddings):
            return self._stack_embeddings(state.global_embeddings, device)
        return torch.zeros(0, self.config.memory_embedding_dim, device=device)

    def _stack_embeddings(self, embeddings: Iterable[Tensor], device: torch.device) -> Tensor:
        if not embeddings:
            return torch.zeros(0, self.config.memory_embedding_dim, device=device)
        stacked = torch.stack([embed.to(device) for embed in embeddings], dim=0)
        return stacked


class SRFBAMCodeAgent(nn.Module):
    """Unified SR-FBAM controller for code-editing episodes."""

    def __init__(self, num_actions: int, config: Optional[SRFBAMCodeConfig] = None) -> None:
        super().__init__()
        self.config = config or SRFBAMCodeConfig()
        self.num_actions = num_actions
        self.vocab = PolicyVocabulary(tuple(MemoryOp), num_actions)
        self.memory = SymbolicMemory(self.config)

        patch_rows = self.config.extractor_patch_rows
        patch_cols = self.config.extractor_patch_cols
        sequence_length = math.ceil(self.config.frame_height / patch_rows) * math.ceil(
            self.config.frame_width / patch_cols
        )
        self.symbol_extractor = TransformerSymbolExtractor(
            vocab_size=self.config.extractor_vocab_size,
            sequence_length=sequence_length,
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

        self.file_embed = nn.Embedding(self.config.token_vocab_size, self.config.file_embedding_dim)
        self.policy_embed = nn.Embedding(self.vocab.size + 1, self.config.policy_embedding_dim)
        self.numeric_proj = nn.Linear(self.config.salience_context_dim, self.config.numeric_hidden_dim)
        self.summary_proj = nn.Linear(self.config.memory_embedding_dim, self.config.symbol_feature_dim)
        self.symbol_proj = nn.Linear(self.config.memory_embedding_dim, self.config.symbol_feature_dim)

        input_dim = (
            self.config.file_embedding_dim
            + self.config.symbol_feature_dim
            + self.config.numeric_hidden_dim
            + self.config.policy_embedding_dim
        )
        self.integrator = nn.LSTMCell(input_dim, self.config.integrator_hidden_dim)
        self.policy_head = nn.Linear(self.config.integrator_hidden_dim, self.vocab.size)
        self.dropout = nn.Dropout(self.config.dropout)

        self.start_token_index = self.vocab.size  # extra embedding slot

        self._last_gate_stats: Optional[Dict[str, float]] = None
        self._last_gate_trace: Optional[List[Dict[str, Any]]] = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
        device = device or self.device
        h = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        c = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        return h, c

    def _hash_index(self, token: str) -> int:
        if not token:
            return 0
        return hash(token.lower()) % self.config.token_vocab_size

    def _file_embedding(self, file_name: str, device: torch.device) -> Tensor:
        idx = self._hash_index(file_name or "")
        idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
        return self.file_embed(idx_tensor)

    def _token_embedding(self, token_index: int, device: torch.device) -> Tensor:
        idx_tensor = torch.tensor([token_index], dtype=torch.long, device=device)
        return self.policy_embed(idx_tensor)

    def _numeric_features_raw(
        self,
        state: SymbolicMemoryState,
        step: FrameActionStep,
        total_steps: int,
        frame_changed: bool,
        summary_available: bool,
        device: torch.device,
    ) -> Tensor:
        cursor = step.metadata.get("state", {}).get("cursor", {})
        line = float(cursor.get("line", 1))
        column = float(cursor.get("column", 1))
        cursor_line_norm = line / 400.0
        cursor_col_norm = column / 200.0
        step_fraction = (state.step_index + 1) / max(total_steps, 1)
        insertion_ratio = state.insertions / max(state.step_index + 1, 1)
        deletion_ratio = state.deletions / max(state.step_index + 1, 1)
        test_success_rate = state.tests_passed / state.tests_runs if state.tests_runs else 0.0
        features = torch.tensor(
            [
                [
                    cursor_line_norm,
                    cursor_col_norm,
                    step_fraction,
                    insertion_ratio,
                    deletion_ratio,
                    test_success_rate,
                    1.0 if frame_changed else 0.0,
                    1.0 if summary_available else 0.0,
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        return features

    def _summary_features(self, summary_vector: Tensor, device: torch.device) -> Tensor:
        if summary_vector.numel() == 0:
            return torch.zeros(1, self.config.symbol_feature_dim, device=device)
        return self.summary_proj(summary_vector)

    def _symbol_features(
        self,
        state: SymbolicMemoryState,
        file_name: str,
        current_symbols: Tensor,
        device: torch.device,
    ) -> Tensor:
        if current_symbols.numel() == 0:
            return torch.zeros(1, self.config.symbol_feature_dim, device=device)

        current_batch = current_symbols.unsqueeze(0)
        memory_embeddings = self.memory.retrieve(state, file_name, device)
        memory_batch = memory_embeddings.unsqueeze(0)

        attended = self.memory_cross_attn(current_batch, memory_batch)
        pooled = attended.mean(dim=1)
        return self.symbol_proj(pooled)

    def _run_extractor(self, frame_tensor: Tensor) -> Tensor:
        extracted, _, _ = self.symbol_extractor(frame_tensor)
        if extracted.numel() == 0:
            return torch.zeros(1, self.config.memory_embedding_dim, device=frame_tensor.device)
        return extracted

    def _execute_memory_op(
        self,
        memory_state: SymbolicMemoryState,
        memory_op: MemoryOp,
        frame_tensor: Tensor,
        frame_hash: int,
        summary_vector: Tensor,
        summary_available: bool,
        file_name: str,
        device: torch.device,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        info: Dict[str, Any] = {
            "decision": "",
            "cache_hit": False,
            "force_write": False,
            "latency_ms_increment": 0.0,
        }
        memory_state.memory_op_counts[memory_op] += 1

        if memory_op == MemoryOp.ENCODE:
            current_symbols = self._run_extractor(frame_tensor)
            info["decision"] = "ENCODE"
            info["latency_ms_increment"] = self.config.encode_latency_ms
            memory_state.extract_count += 1
            memory_state.frame_cache[frame_hash] = current_symbols.detach().cpu()
            while len(memory_state.frame_cache) > self.config.frame_cache_size:
                memory_state.frame_cache.popitem(last=False)
        elif memory_op == MemoryOp.ASSOC:
            current_symbols = None
            cached = memory_state.frame_cache.get(frame_hash)
            info["latency_ms_increment"] = self.config.assoc_latency_ms
            if cached is not None:
                current_symbols = cached.to(device=device)
                info["decision"] = "CACHE"
                info["cache_hit"] = True
                memory_state.cache_hits += 1
                memory_state.reuse_count += 1
                memory_state.frame_cache.move_to_end(frame_hash)
            elif summary_available:
                current_symbols = summary_vector
                info["decision"] = "SUMMARY"
                memory_state.reuse_count += 1
            else:
                current_symbols = self._run_extractor(frame_tensor)
                info["decision"] = "FALLBACK_ENCODE"
                info["latency_ms_increment"] = self.config.encode_latency_ms
                memory_state.extract_count += 1
                memory_state.frame_cache[frame_hash] = current_symbols.detach().cpu()
        elif memory_op == MemoryOp.FOLLOW:
            retrieved = self.memory.retrieve(memory_state, file_name, device)
            info["latency_ms_increment"] = self.config.follow_latency_ms
            if retrieved.numel():
                current_symbols = retrieved.mean(dim=0, keepdim=True)
                info["decision"] = "FOLLOW"
                memory_state.reuse_count += 1
            elif summary_available:
                current_symbols = summary_vector
                info["decision"] = "FOLLOW_SUMMARY"
                memory_state.reuse_count += 1
            else:
                current_symbols = self._run_extractor(frame_tensor)
                info["decision"] = "FOLLOW_FALLBACK_ENCODE"
                info["latency_ms_increment"] = self.config.encode_latency_ms
                memory_state.extract_count += 1
                memory_state.frame_cache[frame_hash] = current_symbols.detach().cpu()
        elif memory_op == MemoryOp.WRITE:
            info["latency_ms_increment"] = self.config.write_latency_ms
            info["decision"] = "WRITE"
            info["force_write"] = True
            current_symbols = summary_vector if summary_available else self._run_extractor(frame_tensor)
            if not summary_available:
                memory_state.extract_count += 1
        else:  # MemoryOp.HALT
            info["latency_ms_increment"] = self.config.halt_latency_ms
            info["decision"] = "HALT"
            current_symbols = torch.zeros(1, self.config.memory_embedding_dim, device=device)
            memory_state.halt_count += 1

        memory_state.total_latency_ms += float(info["latency_ms_increment"])
        return current_symbols, info

    def forward_episode(
        self,
        episode: FrameActionEpisode,
        memory_labels: Optional[Sequence[MemoryOp]] = None,
        teacher_force_actions: bool = True,
    ) -> EpisodeRollout:
        device = self.device
        memory_state = self.memory.new_state()
        h, c = self.init_hidden(device=device)
        prev_token_idx = self.start_token_index
        logits_seq: List[Tensor] = []
        executed_tokens: List[int] = []
        memory_positions: List[int] = []
        action_positions: List[int] = []
        metadata: List[PolicyTokenMeta] = []

        total_steps = len(episode.steps)
        if memory_labels is not None and len(memory_labels) != total_steps:
            raise ValueError("memory_labels length must match number of episode steps")

        self._last_gate_stats = None
        self._last_gate_trace = []

        for step_idx, step in enumerate(episode.steps):
            file_name = step.metadata.get("state", {}).get("active_file", "") or episode.task_id
            frame_tensor = step.frame.to(device=device)
            frame_hash = hash(frame_tensor.detach().cpu().numpy().tobytes())
            frame_changed = frame_hash != memory_state.last_frame_hash
            summary_available = self.config.enable_memory and file_name in memory_state.file_summary
            if summary_available:
                summary_vector = memory_state.file_summary[file_name].to(device=device).unsqueeze(0)
            else:
                summary_vector = torch.zeros(1, self.config.memory_embedding_dim, device=device)

            numeric_raw = self._numeric_features_raw(
                memory_state,
                step,
                total_steps,
                frame_changed=frame_changed,
                summary_available=summary_available,
                device=device,
            )
            numeric_embed = self.numeric_proj(numeric_raw)
            file_embed = self._file_embedding(file_name, device)
            summary_embed = self._summary_features(summary_vector, device)
            prev_token_embed = self._token_embedding(prev_token_idx, device)

            memory_input = torch.cat([file_embed, summary_embed, numeric_embed, prev_token_embed], dim=1)
            h, c = self.integrator(memory_input, (h, c))
            memory_logits = self.policy_head(self.dropout(h))
            logits_seq.append(memory_logits)
            memory_positions.append(len(logits_seq) - 1)

            if memory_labels is not None:
                memory_op = memory_labels[step_idx]
            else:
                memory_logits_slice = memory_logits[:, : self.vocab.action_offset]
                memory_op_idx = int(memory_logits_slice.argmax(dim=1).item())
                memory_op = MemoryOp(memory_op_idx)

            mem_token_idx = self.vocab.memory_index(memory_op)
            executed_tokens.append(mem_token_idx)
            metadata.append(PolicyTokenMeta(step_index=step_idx, phase="memory", memory_op=memory_op))
            prev_token_idx = mem_token_idx

            current_symbols, op_info = self._execute_memory_op(
                memory_state=memory_state,
                memory_op=memory_op,
                frame_tensor=frame_tensor,
                frame_hash=frame_hash,
                summary_vector=summary_vector,
                summary_available=summary_available,
                file_name=file_name,
                device=device,
            )

            numeric_embed_action = self.numeric_proj(numeric_raw)
            symbol_embed = self._symbol_features(memory_state, file_name, current_symbols, device)
            token_embed = self._token_embedding(prev_token_idx, device)
            action_input = torch.cat([file_embed, symbol_embed, numeric_embed_action, token_embed], dim=1)
            h, c = self.integrator(action_input, (h, c))
            action_logits = self.policy_head(self.dropout(h))
            logits_seq.append(action_logits)
            action_positions.append(len(logits_seq) - 1)

            if teacher_force_actions:
                env_action_idx = int(step.action_index)
            else:
                action_slice = action_logits[:, self.vocab.action_offset :]
                env_action_idx = int(action_slice.argmax(dim=1).item())

            action_token_idx = self.vocab.action_index(env_action_idx)
            executed_tokens.append(action_token_idx)
            metadata.append(PolicyTokenMeta(step_index=step_idx, phase="action", action_index=env_action_idx))
            prev_token_idx = action_token_idx

            terminal_output = step.metadata.get("terminal_output")
            salience_metrics = self.memory.update(
                memory_state,
                file_name,
                current_symbols,
                step.action_type,
                terminal_output,
                salience_context=numeric_raw,
                force_write=bool(op_info.get("force_write", False)),
            )
            if self.config.enable_memory and current_symbols.numel():
                summary_update = current_symbols.mean(dim=0).detach().cpu()
                if file_name not in memory_state.file_summary:
                    memory_state.file_summary[file_name] = summary_update
                else:
                    old_summary = memory_state.file_summary[file_name]
                    memory_state.file_summary[file_name] = (
                        self.config.summary_decay * summary_update
                        + (1.0 - self.config.summary_decay) * old_summary
                    )

            memory_state.last_frame_hash = frame_hash

            self._last_gate_trace.append(
                {
                    "episode_id": episode.episode_id,
                    "step_index": step_idx,
                    "memory_op": memory_op.name,
                    "memory_decision": op_info.get("decision"),
                    "action_type": step.action_type,
                    "file": file_name,
                    "summary_available": summary_available,
                    "frame_changed": frame_changed,
                    "cache_hit": op_info.get("cache_hit", False),
                    "latency_ms_increment": op_info.get("latency_ms_increment"),
                    "write_committed": salience_metrics.get("write_committed"),
                    "salience_probability": salience_metrics.get("salience_probability"),
                    "cumulative_latency_ms": memory_state.total_latency_ms,
                }
            )

        logits_tensor = torch.cat(logits_seq, dim=0) if logits_seq else torch.zeros(0, self.vocab.size, device=device)
        rollout = EpisodeRollout(
            logits=logits_tensor,
            executed_tokens=executed_tokens,
            memory_positions=memory_positions,
            action_positions=action_positions,
            metadata=metadata,
            vocabulary=self.vocab,
        )

        steps = max(total_steps, 1)
        encode_count = memory_state.memory_op_counts.get(MemoryOp.ENCODE, 0)
        assoc_count = memory_state.memory_op_counts.get(MemoryOp.ASSOC, 0)
        follow_count = memory_state.memory_op_counts.get(MemoryOp.FOLLOW, 0)
        write_count = memory_state.memory_op_counts.get(MemoryOp.WRITE, 0)
        halt_count = memory_state.memory_op_counts.get(MemoryOp.HALT, 0)
        query_count = assoc_count + follow_count
        skip_count = max(0, steps - encode_count - query_count)

        encode_fraction = encode_count / steps
        query_fraction = query_count / steps
        skip_fraction = max(0.0, 1.0 - encode_fraction - query_fraction)

        baseline_latency = steps * self.config.encode_latency_ms
        actual_latency = (
            encode_count * self.config.encode_latency_ms
            + assoc_count * self.config.assoc_latency_ms
            + follow_count * self.config.follow_latency_ms
            + write_count * self.config.write_latency_ms
            + halt_count * self.config.halt_latency_ms
            + max(0, steps - encode_count - assoc_count - follow_count - write_count - halt_count)
            * self.config.skip_latency_ms
        )
        observed_latency = episode.total_wall_time_ms if episode.total_wall_time_ms > 0 else actual_latency
        predicted_speedup = baseline_latency / max(actual_latency, 1e-6)
        observed_speedup = baseline_latency / max(observed_latency, 1e-6)

        self._last_gate_stats = {
            "encode_fraction": encode_fraction,
            "query_fraction": query_fraction,
            "skip_fraction": skip_fraction,
            "halt_fraction": halt_count / steps,
            "cache_hit_rate": memory_state.cache_hits / steps,
            "predicted_speedup": predicted_speedup,
            "observed_speedup": observed_speedup,
            "baseline_latency_ms": baseline_latency,
            "actual_latency_ms": actual_latency,
            "observed_latency_ms": observed_latency,
            "memory_op_counts": {op.name: int(count) for op, count in memory_state.memory_op_counts.items()},
            "write_commit_rate": (
                memory_state.salience_committed / max(memory_state.salience_attempts, 1)
            ),
            "salience_average": (
                (memory_state.salience_sum / max(memory_state.salience_attempts, 1))
                if memory_state.salience_attempts
                else 0.0
            ),
        }

        return rollout

    def predict_actions(self, episode: FrameActionEpisode) -> Tensor:
        rollout = self.forward_episode(episode, memory_labels=None, teacher_force_actions=False)
        action_logits = rollout.action_logits_env()
        if action_logits.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return action_logits.argmax(dim=1)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    @property
    def last_gate_stats(self) -> Optional[Dict[str, float]]:
        return self._last_gate_stats

    @property
    def last_gate_trace(self) -> Optional[List[Dict[str, Any]]]:
        return self._last_gate_trace
