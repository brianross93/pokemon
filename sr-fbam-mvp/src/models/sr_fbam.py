"""
Symbolic-Recurrence FBAM (SR-FBAM) core model.

Implements the Frame Head, Recurrent Integrator, and action loop specified in
Overview.md:101-174 and the Phase 1 budgeting goals in Overview.md:142-146.

The model supports multi-hop reasoning over the provided KnowledgeGraph using
discrete operators (ASSOC, FOLLOW, VOTE, WRITE, HALT) and emits HopTrace
records compatible with the ExperimentLogger.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.data import KnowledgeGraph, Query
from src.evaluation.logger_schema import HopTrace, current_gpu_memory_mb


class Action(IntEnum):
    """Discrete actions available to the integrator."""

    ASSOC = 0
    FOLLOW = 1
    VOTE = 2
    WRITE = 3
    HALT = 4
    START = 5  # Only used for the initial action embedding


ACTION_SEQUENCE: Sequence[Action] = [
    Action.ASSOC,
    Action.FOLLOW,
    Action.VOTE,
    Action.WRITE,
    Action.HALT,
]


@dataclass
class SRFBAMConfig:
    """Configuration parameters (Overview.md:142-146)."""

    token_vocab_size: int = 4096
    token_embedding_dim: int = 72
    frame_dim: int = 256
    node_vocab_size: int = 2048
    node_embedding_dim: int = 96
    relation_vocab_size: int = 128
    relation_embedding_dim: int = 64
    action_embedding_dim: int = 32
    integrator_hidden_dim: int = 192
    max_hops: int = 10
    tolerance: float = 0.10
    parameter_targets: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.parameter_targets is None:
            self.parameter_targets = {
                "frame_head": 300_000,
                "integrator": 500_000,
                "embeddings": 200_000,
            }


class SimpleTokenizer:
    """Whitespace and punctuation tokenizer used by the frame head."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.pattern = re.compile(r"[A-Za-z0-9]+")

    def tokenize(self, text: str) -> List[str]:
        return self.pattern.findall(text.lower())

    def token_to_index(self, token: str) -> int:
        # Hash to the vocabulary range; ensure deterministic non-negative indices.
        return hash(token) % self.vocab_size


class FrameHead(nn.Module):
    """
    Encodes the textual query into a sparse symbol representation.

    Parameter budget ~300K:
        - Embedding: vocab(4096) x dim(72) = 294,912
        - Projection: 72 -> 256 = 18,432
    """

    def __init__(self, config: SRFBAMConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = SimpleTokenizer(config.token_vocab_size)
        self.token_embed = nn.Embedding(config.token_vocab_size, config.token_embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(config.token_embedding_dim, config.frame_dim),
            nn.LayerNorm(config.frame_dim),
            nn.ReLU(),
        )
        nn.init.xavier_uniform_(self.token_embed.weight)

    def forward(self, query: Query, device: torch.device) -> Tuple[Tensor, Dict]:
        tokens = self.tokenizer.tokenize(query.natural_language)
        if not tokens:
            tokens = ["<blank>"]
        indices = [self.tokenizer.token_to_index(tok) for tok in tokens]
        token_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        embeddings = self.token_embed(token_tensor)
        pooled = embeddings.mean(dim=0)
        frame_embed = self.proj(pooled)
        context = {
            "tokens": tokens,
            "token_indices": indices,
        }
        return frame_embed, context


class SymbolEmbeddingBank(nn.Module):
    """
    Embedding tables for nodes and relations (~200K params).

    node_vocab_size(2048) x node_embedding_dim(96) = 196,608
    relation_vocab_size(128) x relation_embedding_dim(64) = 8,192
    """

    def __init__(self, config: SRFBAMConfig) -> None:
        super().__init__()
        self.node_vocab_size = config.node_vocab_size
        self.node_embedding_dim = config.node_embedding_dim
        self.relation_vocab_size = config.relation_vocab_size
        self.relation_embedding_dim = config.relation_embedding_dim

        self.node_embed = nn.Embedding(self.node_vocab_size, self.node_embedding_dim)
        self.relation_embed = nn.Embedding(self.relation_vocab_size, self.relation_embedding_dim)

        nn.init.xavier_uniform_(self.node_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

    @property
    def device(self) -> torch.device:
        return self.node_embed.weight.device

    def _node_index(self, node_id: str) -> int:
        return hash(f"node::{node_id}") % self.node_vocab_size

    def _relation_index(self, relation: str) -> int:
        return hash(f"rel::{relation}") % self.relation_vocab_size

    def embed_nodes(self, node_ids: Sequence[str]) -> Tensor:
        if not node_ids:
            return torch.zeros(self.node_embedding_dim, device=self.device)
        indices = torch.tensor(
            [self._node_index(nid) for nid in node_ids],
            dtype=torch.long,
            device=self.device,
        )
        return self.node_embed(indices).mean(dim=0)

    def embed_relation(self, relation: Optional[str]) -> Tensor:
        if not relation:
            return torch.zeros(self.relation_embedding_dim, device=self.device)
        idx = torch.tensor(self._relation_index(relation), dtype=torch.long, device=self.device)
        return self.relation_embed(idx)


class Integrator(nn.Module):
    """
    LSTM integrator with action and confidence heads (~500K params).

    Input dimension = frame_dim + node_embedding_dim + action_embedding_dim.
    Hidden dimension = config.integrator_hidden_dim (default 192).
    """

    def __init__(self, config: SRFBAMConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_dim = config.frame_dim
        self.node_dim = config.node_embedding_dim
        self.action_embedding_dim = config.action_embedding_dim
        self.hidden_dim = config.integrator_hidden_dim
        self.num_actions = len(ACTION_SEQUENCE)

        input_dim = self.frame_dim + self.node_dim + self.action_embedding_dim
        self.lstm_cell = nn.LSTMCell(input_dim, self.hidden_dim)
        self.action_embed = nn.Embedding(self.num_actions + 1, self.action_embedding_dim)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        nn.init.xavier_uniform_(self.action_embed.weight)
        for module in self.action_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.confidence_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def step(
        self,
        frame_embed: Tensor,
        node_embed: Tensor,
        prev_action_idx: int,
        state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one integrator step.

        Returns:
            h, c, action_logits, confidence
        """
        device = frame_embed.device
        action_vec = self.action_embed(
            torch.tensor([prev_action_idx], dtype=torch.long, device=device)
        )
        lstm_input = torch.cat(
            [frame_embed.unsqueeze(0), node_embed.unsqueeze(0), action_vec],
            dim=-1,
        )
        h, c = self.lstm_cell(lstm_input.squeeze(0), state)
        logits = self.action_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))
        return h, c, logits.squeeze(0), confidence.squeeze(0)


class MemoryContext:
    """Runtime view over the KnowledgeGraph using symbolic operators."""

    def __init__(self, kg: KnowledgeGraph, embeddings: SymbolEmbeddingBank) -> None:
        self.kg = kg
        self.embeddings = embeddings
        self.state: Dict[str, List[str]] = {
            "current_nodes": [],
            "candidate_history": [],
        }
        self.last_relation: Optional[str] = None
        self.last_target: Optional[str] = None

    def embed_current_nodes(self) -> Tensor:
        return self.embeddings.embed_nodes(self.state.get("current_nodes", []))

    def node_names(self, node_ids: Iterable[str]) -> List[str]:
        names = []
        for node_id in node_ids:
            node = self.kg.get_node(node_id)
            names.append(node.name if node else node_id)
        return names

    def execute(
        self,
        action: Action,
        plan_step: Optional[Dict],
    ) -> Tuple[List[str], str]:
        """Execute an action and return (candidate_ids, description)."""
        if action == Action.HALT:
            return self.state.get("current_nodes", []), "HALT"

        relation = None
        node_type = None
        source_id = None
        target_id = None

        if plan_step:
            relation = plan_step.get("relation") or plan_step.get("edge") or plan_step.get("rel")
            node_type = plan_step.get("node_type") or plan_step.get("type")
            source_id = plan_step.get("source_id") or plan_step.get("source")
            target_id = plan_step.get("target_id") or plan_step.get("target")

        if not source_id and self.state.get("current_nodes"):
            source_id = self.state["current_nodes"][0]
        if not target_id:
            target_id = self.last_target

        candidates: List[str] = []
        description = ""

        if action == Action.ASSOC:
            if relation:
                candidates = self.kg.assoc(
                    relation,
                    target_id=target_id,
                    source_id=source_id,
                    node_type=node_type,
                )
                description = f"ASSOC relation={relation}"
            self.last_relation = relation
            self.last_target = target_id

        elif action == Action.FOLLOW:
            relation = relation or self.last_relation
            if relation and source_id:
                candidates = self.kg.follow(
                    source_id=source_id,
                    relation=relation,
                    node_type=node_type,
                )
                description = f"FOLLOW relation={relation} source={source_id}"

        elif action == Action.VOTE:
            history = [node for candidates in self.state["candidate_history"] for node in candidates]
            if history:
                counts = Counter(history)
                top_node = max(counts.items(), key=lambda kv: kv[1])[0]
                candidates = [top_node]
                description = f"VOTE chose node={top_node}"
            else:
                candidates = self.state.get("current_nodes", [])
                description = "VOTE fallback"

        elif action == Action.WRITE:
            # Placeholder: in Phase 1 we only log the intent.
            candidates = self.state.get("current_nodes", [])
            description = "WRITE noop"

        self.state["candidate_history"].append(candidates)
        if candidates:
            self.state["current_nodes"] = candidates
        return candidates, description


@dataclass
class ReasoningOutput:
    prediction_id: Optional[str]
    prediction_name: Optional[str]
    confidence: float
    hop_traces: List[HopTrace]
    halted: bool
    halting_step: int
    candidate_ids: List[str]


class SRFBAM(nn.Module):
    """Top-level SR-FBAM model."""

    def __init__(self, config: Optional[SRFBAMConfig] = None) -> None:
        super().__init__()
        self.config = config or SRFBAMConfig()
        self.frame_head = FrameHead(self.config)
        self.embeddings = SymbolEmbeddingBank(self.config)
        self.integrator = Integrator(self.config)

        # Precompute action index mapping
        self.action_to_index = {action: idx for idx, action in enumerate(ACTION_SEQUENCE)}
        self.start_action_index = len(ACTION_SEQUENCE)

        # Budget verification cache
        self._budget_cache: Optional[Dict[str, int]] = None

    @property
    def device(self) -> torch.device:
        return self.frame_head.token_embed.weight.device

    def parameter_budgets(self) -> Dict[str, int]:
        """Return parameter counts per component."""
        if self._budget_cache is None:
            self._budget_cache = {
                "frame_head": sum(p.numel() for p in self.frame_head.parameters()),
                "integrator": sum(p.numel() for p in self.integrator.parameters()),
                "embeddings": sum(p.numel() for p in self.embeddings.parameters()),
            }
        return self._budget_cache

    def verify_parameter_budgets(self) -> Dict[str, Dict[str, float]]:
        """
        Check counts against targets (±10% tolerance by default).

        Returns dict with count, target, lower, upper, within_tolerance.
        """
        budgets = self.parameter_budgets()
        report: Dict[str, Dict[str, float]] = {}
        for name, count in budgets.items():
            target = self.config.parameter_targets.get(name, count)
            tol = self.config.tolerance
            lower = target * (1 - tol)
            upper = target * (1 + tol)
            report[name] = {
                "count": float(count),
                "target": float(target),
                "lower": float(lower),
                "upper": float(upper),
                "within_tolerance": float(lower) <= count <= float(upper),
            }
        return report

    def forward(self, *args, **kwargs):
        """Placeholder forward to maintain nn.Module interface."""
        raise NotImplementedError("Use reason(...) for inference-time execution.")

    @torch.no_grad()
    def reason(
        self,
        query: Query,
        kg: KnowledgeGraph,
        max_hops: Optional[int] = None,
    ) -> ReasoningOutput:
        """
        Execute the action loop for a single query.

        Args:
            query: Multi-hop query definition.
            kg: KnowledgeGraph with ASSOC/FOLLOW operations.
            max_hops: Optional override (defaults to config.max_hops).
        """
        self.eval()
        device = self.device
        max_steps = max_hops or self.config.max_hops

        frame_embed, _ = self.frame_head(query, device=device)
        memory = MemoryContext(kg, self.embeddings)
        node_embed = memory.embed_current_nodes()

        h, c = self.integrator.init_state(batch_size=1, device=device)
        prev_action_idx = self.start_action_index

        hop_traces: List[HopTrace] = []
        start_time = perf_counter()
        final_confidence = 0.0
        halted = False

        plan = query.symbolic_plan if isinstance(query.symbolic_plan, list) else []

        for hop in range(max_steps):
            h, c, logits, confidence = self.integrator.step(
                frame_embed, node_embed, prev_action_idx, (h, c)
            )
            action_index = int(torch.argmax(logits).item())
            action = ACTION_SEQUENCE[action_index]
            plan_step = plan[hop] if hop < len(plan) else None

            candidate_ids, description = memory.execute(action, plan_step)
            node_embed = memory.embed_current_nodes()

            timestamp_ms = (perf_counter() - start_time) * 1000.0
            node_names = memory.node_names(candidate_ids)
            hop_traces.append(
                HopTrace(
                    hop_number=hop + 1,
                    action=action.name,
                    query=description or query.natural_language,
                    result=", ".join(node_names) if node_names else "[none]",
                    confidence=float(confidence.item()),
                    timestamp_ms=float(timestamp_ms),
                    gpu_memory_mb=current_gpu_memory_mb(),
                )
            )

            final_confidence = float(confidence.item())
            prev_action_idx = action_index

            if action == Action.HALT:
                halted = True
                break

        current_nodes = memory.state.get("current_nodes", [])
        prediction_id = current_nodes[0] if current_nodes else None
        prediction_name = None
        if prediction_id:
            node = kg.get_node(prediction_id)
            prediction_name = node.name if node else prediction_id

        return ReasoningOutput(
            prediction_id=prediction_id,
            prediction_name=prediction_name,
            confidence=final_confidence,
            hop_traces=hop_traces,
            halted=halted,
            halting_step=len(hop_traces),
            candidate_ids=current_nodes,
        )


def create_srfbam(config: Optional[SRFBAMConfig] = None, device: Optional[str] = None) -> SRFBAM:
    """Factory helper to instantiate SR-FBAM with budget verification."""
    model = SRFBAM(config=config)
    if device:
        model = model.to(torch.device(device))
    # Log parameter budgets to stderr for quick inspection.
    budgets = model.verify_parameter_budgets()
    print("SR-FBAM parameter budgets:", budgets, file=sys.stderr)
    return model
