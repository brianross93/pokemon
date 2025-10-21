"""
Task wrapper that pairs the SR-FBAM core with battle-specific components.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from srfbam.core import EncodedFrame, SRFBAMCore, SrfbamStepSummary

from src.pkmn_battle.env import BattleObs, EnvAdapter, LegalAction
from src.pkmn_battle.extractor import Extractor
from src.pkmn_battle.graph import GraphMemory, WriteOp
from src.pkmn_battle.plan.interpreter import BattlePlanExecutor, PlanDecision
from src.plan.planner_llm import PlanletSpec
from src.pkmn_battle.policy import (
    ActionSpace,
    BattleControllerPolicy,
    BattleGateDecision,
    BattleGateMode,
)


PolicyFn = Callable[
    [
        BattleObs,
        Sequence[LegalAction],
        Sequence[float],
        Dict[int, LegalAction],
        GraphMemory,
        ActionSpace,
    ],
    Tuple[LegalAction, BattleGateDecision],
]
FrameEncoderFn = Callable[[BattleObs], EncodedFrame]


@dataclass
class BattleTelemetry:
    """Lightweight telemetry bundle for downstream logging."""

    last_summary: Optional[SrfbamStepSummary]
    last_writes: Sequence[WriteOp]
    last_action: Optional[LegalAction]
    legal_actions: Sequence[LegalAction]
    action_mask: Sequence[float]
    index_map: Dict[int, LegalAction]
    gate_decision: Optional[BattleGateDecision]
    hop_trace: Sequence[Dict[str, object]]
    fractions: Dict[str, float]
    speedup: Dict[str, Optional[float]]
    latency_ms: float
    fallback_required: bool
    plan_id: Optional[str] = None
    planlet_id: Optional[str] = None
    planlet_kind: Optional[str] = None
    plan_source: Optional[str] = None
    plan_cache_hit: Optional[bool] = None
    plan_metadata: Dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Dict[str, object]]:
        """Serialise telemetry into the shared schema namespaces."""

        mask_values: List[Optional[float]] = []
        for value in self.action_mask:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                mask_values.append(None)
                continue
            mask_values.append(numeric if math.isfinite(numeric) else None)

        core = {
            "legal_actions": [dict(action) for action in self.legal_actions],
            "action": dict(self.last_action) if self.last_action is not None else None,
            "action_mask": mask_values,
            "gate": _serialize_gate_decision(self.gate_decision),
            "fractions": dict(self.fractions),
            "speedup": dict(self.speedup),
            "latency_ms": float(self.latency_ms),
            "fallback_required": bool(self.fallback_required),
            "hop_trace": [dict(hop) for hop in self.hop_trace],
        }
        battle = {
            "writes": [_serialize_write_op(op) for op in self.last_writes],
            "index_map": {str(index): dict(action) for index, action in self.index_map.items()},
        }
        if self.last_summary is not None:
            battle["summary"] = _serialize_summary(self.last_summary)
        if self.planlet_id is not None or self.planlet_kind is not None or self.plan_id is not None:
            plan: Dict[str, object] = {
                "id": self.plan_id,
                "planlet_id": self.planlet_id,
                "planlet_kind": self.planlet_kind,
            }
            if self.plan_source is not None:
                plan["source"] = self.plan_source
            if self.plan_cache_hit is not None:
                plan["cache_hit"] = self.plan_cache_hit
            if self.plan_metadata:
                plan.update(dict(self.plan_metadata))
            core["plan"] = plan
        return {"core": core, "battle": battle}

    def plan_context(self) -> Dict[str, object]:
        """Return plan metadata suitable for the JSONL context block."""

        planlet_id = self.planlet_id or "NONE"
        planlet_kind = self.planlet_kind or "NONE"
        context: Dict[str, object] = {
            "id": self.plan_id,
            "planlet_id": planlet_id,
            "planlet_kind": planlet_kind,
        }
        if self.plan_source is not None:
            context["source"] = self.plan_source
        if self.plan_cache_hit is not None:
            context["cache_hit"] = self.plan_cache_hit
        if self.plan_metadata:
            context.update(dict(self.plan_metadata))
        return context


def _serialize_gate_decision(decision: Optional[BattleGateDecision]) -> Dict[str, object]:
    if decision is None:
        return {
            "mode": None,
            "encode_flag": None,
            "view": None,
            "confidence": None,
            "reason": None,
        }
    mode = decision.mode.value if isinstance(decision.mode, BattleGateMode) else str(decision.mode)
    payload: Dict[str, object] = {
        "mode": mode,
        "encode_flag": bool(decision.encode_flag),
        "view": None,
        "confidence": getattr(decision, "confidence", None),
        "reason": getattr(decision, "reason", None),
    }
    metadata = getattr(decision, "metadata", None)
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def _serialize_write_op(op: WriteOp) -> Dict[str, object]:
    base: Dict[str, object] = {
        "kind": op.kind,
        "confidence": getattr(op, "confidence", 1.0),
        "fallback_required": getattr(op, "fallback_required", False),
        "turn_hint": getattr(op, "turn_hint", None),
    }
    payload = getattr(op, "payload", None)
    if op.kind == "node" and payload is not None:
        base.update(
            {
                "type": getattr(payload, "type", None),
                "id": getattr(payload, "node_id", None),
                "attributes": getattr(payload, "attributes", {}),
            }
        )
    elif payload is not None:
        base.update(
            {
                "relation": getattr(payload, "relation", None),
                "src": getattr(payload, "src", None),
                "dst": getattr(payload, "dst", None),
                "attributes": getattr(payload, "attributes", {}),
            }
        )
    return base


def _serialize_summary(summary: SrfbamStepSummary) -> Dict[str, object]:
    """Strip tensors to scalars for telemetry logging."""

    gate_stats = dict(summary.gate_stats)
    return {
        "context_key": summary.context_key,
        "gate": gate_stats,
    }


class SRFBAMBattleAgent:
    """
    Glue code between the shared SR-FBAM core and battle-specific modules.

    The agent exposes a high-level ``act`` method that returns the chosen
    legal action after updating the symbolic graph. Policies can be swapped
    out by passing a custom ``policy_fn``.
    """

    def __init__(
        self,
        *,
        env: EnvAdapter,
        extractor: Extractor,
        graph: Optional[GraphMemory] = None,
        core: Optional[SRFBAMCore] = None,
        frame_encoder: Optional[FrameEncoderFn] = None,
        policy_fn: Optional[PolicyFn] = None,
    ) -> None:
        self.env = env
        self.extractor = extractor
        self.graph = graph or GraphMemory()
        self.core = core or SRFBAMCore()
        self.frame_encoder = frame_encoder
        self.action_space = ActionSpace()
        self._controller_policy = BattleControllerPolicy(action_space=self.action_space)
        self.policy_fn = policy_fn or self._controller_policy.select

        self._last_summary: Optional[SrfbamStepSummary] = None
        self._last_writes: List[WriteOp] = []
        self._last_action: Optional[LegalAction] = None
        self._last_legal: Tuple[LegalAction, ...] = ()
        self._last_mask: Tuple[float, ...] = ()
        self._index_map: Dict[int, LegalAction] = {}
        self._last_gate_decision: Optional[BattleGateDecision] = None
        self._last_latency_ms: float = 0.0
        self._gate_counts: Dict[str, int] = {}
        self._latency_stats: Dict[str, float] = {}
        self._metrics: Dict[str, Dict[str, Optional[float]]] = {}
        self._reset_metrics()
        self._fallback_required: bool = False
        self._latency_by_gate: Dict[str, Dict[str, float]] = {}
        self._plan_id: Optional[str] = None
        self._planlet_id: Optional[str] = None
        self._planlet_kind: Optional[str] = None
        self._plan_source: Optional[str] = None
        self._plan_cache_hit: Optional[bool] = None
        self._plan_metadata: Dict[str, object] = {}
        self._plan_executor = None
        self._plan_clear_after_step = False
        self._last_observation: Optional[BattleObs] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> BattleObs:
        obs = self.env.reset()
        self.core.reset_state()
        self._reset_metrics()
        self.clear_planlet_context()
        self._process_observation(obs)
        return obs

    def observe(self) -> BattleObs:
        obs = self.env.observe()
        self._process_observation(obs)
        return obs

    def set_planlet_context(
        self,
        *,
        plan_id: Optional[str] = None,
        planlet_id: Optional[str] = None,
        planlet_kind: Optional[str] = None,
        source: Optional[str] = None,
        cache_hit: Optional[bool] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Attach planlet metadata for subsequent telemetry payloads."""

        self._plan_id = plan_id
        self._planlet_id = planlet_id
        self._planlet_kind = planlet_kind
        self._plan_source = source
        self._plan_cache_hit = cache_hit
        self._plan_metadata = dict(metadata or {})

    def clear_planlet_context(self) -> None:
        """Remove any active plan metadata."""

        self._plan_id = None
        self._planlet_id = None
        self._planlet_kind = None
        self._plan_source = None
        self._plan_cache_hit = None
        self._plan_metadata = {}

    def planlet_context(self) -> Dict[str, object]:
        """Return plan metadata suitable for the JSONL context block."""

        planlet_id = self._planlet_id or "NONE"
        planlet_kind = self._planlet_kind or "NONE"
        context: Dict[str, object] = {
            "id": self._plan_id,
            "planlet_id": planlet_id,
            "planlet_kind": planlet_kind,
        }
        if self._plan_source is not None:
            context["source"] = self._plan_source
        if self._plan_cache_hit is not None:
            context["cache_hit"] = self._plan_cache_hit
        if self._plan_metadata:
            context.update(dict(self._plan_metadata))
        return context

    def _maybe_execute_plan(
        self,
        obs: BattleObs,
        legal_actions: Sequence[LegalAction],
    ) -> Optional[Tuple[LegalAction, BattleGateDecision]]:
        if self._plan_executor is None:
            return None

        decision = self._plan_executor.next_action(obs, legal_actions)
        status = decision.status
        metadata = {**self._plan_metadata, **dict(decision.metadata)}

        if status == "action" and decision.action is not None:
            metadata["status"] = "executing"
            self._plan_metadata = metadata
            gate = BattleGateDecision(
                mode=BattleGateMode.PLAN_STEP,
                reason="plan_step",
                encode_flag=False,
                metadata=metadata,
            )
            return decision.action, gate

        if status == "complete":
            metadata["status"] = "completed"
            self._plan_metadata = metadata
            self._plan_executor = None
            self._plan_clear_after_step = True
            return None

        if status == "abort":
            metadata["status"] = "aborted"
            if decision.reason:
                metadata.setdefault("reason", decision.reason)
            self._plan_metadata = metadata
            self._plan_executor = None
            self._plan_clear_after_step = True
            return None

        return None

    def planlet_context(self) -> Dict[str, object]:
        """Return plan metadata suitable for the JSONL context block."""

        planlet_id = self._planlet_id or "NONE"
        planlet_kind = self._planlet_kind or "NONE"
        context: Dict[str, object] = {
            "id": self._plan_id,
            "planlet_id": planlet_id,
            "planlet_kind": planlet_kind,
        }
        if self._plan_source is not None:
            context["source"] = self._plan_source
        if self._plan_cache_hit is not None:
            context["cache_hit"] = self._plan_cache_hit
        if self._plan_metadata:
            context.update(dict(self._plan_metadata))
        return context

    def load_planlet(
        self,
        planlet: PlanletSpec,
        *,
        plan_id: Optional[str] = None,
        source: str = "llm",
        cache_hit: Optional[bool] = None,
        cache_key: Optional[str] = None,
    ) -> bool:
        """Register a planlet for execution if preconditions hold."""

        if planlet.kind.upper() != "BATTLE":
            return False

        observation = self._last_observation
        if observation is None:
            observation = self.observe()

        executor = BattlePlanExecutor(planlet, plan_id=plan_id)
        executor.start(observation)
        if not executor.preconditions_satisfied:
            rejection_meta = {
                "status": "rejected",
                "errors": executor.precondition_errors(),
            }
            if cache_key is not None:
                rejection_meta["cache_key"] = cache_key
            self.set_planlet_context(
                plan_id=plan_id,
                planlet_id=planlet.id,
                planlet_kind=planlet.kind,
                source=source,
                cache_hit=cache_hit,
                metadata=rejection_meta,
            )
            self._plan_executor = None
            self._plan_clear_after_step = True
            return False

        metadata = {
            "status": "active",
            "step_index": 0,
            "steps_total": executor.steps_total,
        }
        if cache_key is not None:
            metadata["cache_key"] = cache_key

        self._plan_executor = executor
        self._plan_clear_after_step = False
        self.set_planlet_context(
            plan_id=plan_id,
            planlet_id=planlet.id,
            planlet_kind=planlet.kind,
            source=source,
            cache_hit=cache_hit,
            metadata=metadata,
        )
        return True

    def act(self) -> LegalAction:
        start = time.perf_counter()
        obs = self.observe()
        legal = self.env.legal_actions()
        mask, index_map = self.action_space.build_mask(legal)
        self._last_legal = tuple(legal)
        self._last_mask = mask
        self._index_map = index_map

        plan_result = self._maybe_execute_plan(obs, legal)
        if plan_result is not None:
            action, gate_decision = plan_result
            self._last_action = action
            self._last_gate_decision = gate_decision
            action_index = self.action_space.to_index(action)
            self.core.set_last_action_index(action_index)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._update_metrics(gate_decision, latency_ms)
            return action

        action, gate_decision = self.policy_fn(
            obs,
            legal,
            mask,
            index_map,
            self.graph,
            self.action_space,
        )
        self._last_action = action
        self._last_gate_decision = gate_decision
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._update_metrics(gate_decision, latency_ms)
        return action

    def step(self) -> BattleObs:
        action = self.act()
        return self.env.step(action)

    def telemetry(self) -> BattleTelemetry:
        hops = self.graph.drain_hops()
        return BattleTelemetry(
            last_summary=self._last_summary,
            last_writes=tuple(self._last_writes),
            last_action=self._last_action,
            legal_actions=self._last_legal,
            action_mask=self._last_mask,
            index_map=dict(self._index_map),
            gate_decision=self._last_gate_decision,
            hop_trace=tuple(hops),
            fractions=dict(self._metrics.get("fractions", {})),
            speedup=dict(self._metrics.get("speedup", {})),
            latency_ms=self._last_latency_ms,
            fallback_required=self._fallback_required,
            plan_id=self._plan_id,
            planlet_id=self._planlet_id,
            planlet_kind=self._planlet_kind,
            plan_source=self._plan_source,
            plan_cache_hit=self._plan_cache_hit,
            plan_metadata=dict(self._plan_metadata),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _process_observation(self, obs: BattleObs) -> None:
        self._last_observation = obs
        writes = list(self.extractor.extract(obs))
        for op in writes:
            self.graph.write(op)
        self._last_writes = writes
        self._fallback_required = any(getattr(op, "fallback_required", False) for op in writes)

        if self.frame_encoder is not None:
            encoded_frame = self.frame_encoder(obs)
            self._last_summary = self.core.encode_step(encoded_frame)

    def _default_policy(
        self,
        obs: BattleObs,
        legal_actions: Sequence[LegalAction],
        action_mask: Sequence[float],
        index_map: Dict[int, LegalAction],
        graph: GraphMemory,
        action_space: ActionSpace,
    ) -> Tuple[LegalAction, BattleGateDecision]:
        return self._controller_policy.select(
            obs,
            legal_actions,
            action_mask,
            index_map,
            graph,
            action_space,
        )

    def _reset_metrics(self) -> None:
        self._gate_counts = {
            "total": 0,
            "encode": 0,
            "assoc": 0,
            "follow": 0,
            "write": 0,
            "halt": 0,
            "skip": 0,
            "plan_lookup": 0,
            "plan_step": 0,
        }
        self._latency_stats = {"total_ms": 0.0, "steps": 0}
        self._metrics = {
            "fractions": {"encode": 0.0, "query": 0.0, "skip": 0.0},
            "speedup": {"predicted": None, "observed": None},
        }
        self._last_latency_ms = 0.0
        self._fallback_required = False
        self._latency_by_gate = {
            "encode": {"total_ms": 0.0, "count": 0},
            "assoc": {"total_ms": 0.0, "count": 0},
            "follow": {"total_ms": 0.0, "count": 0},
            "write": {"total_ms": 0.0, "count": 0},
            "skip": {"total_ms": 0.0, "count": 0},
            "plan_lookup": {"total_ms": 0.0, "count": 0},
            "plan_step": {"total_ms": 0.0, "count": 0},
        }

    def _update_metrics(self, decision: Optional[BattleGateDecision], latency_ms: float) -> None:
        self._last_latency_ms = float(latency_ms)
        self._latency_stats["total_ms"] += self._last_latency_ms
        self._latency_stats["steps"] += 1

        if decision is not None:
            self._gate_counts["total"] += 1
            if decision.encode_flag:
                self._gate_counts["encode"] += 1
                self._latency_by_gate["encode"]["total_ms"] += self._last_latency_ms
                self._latency_by_gate["encode"]["count"] += 1
            if decision.mode == BattleGateMode.ASSOC:
                self._gate_counts["assoc"] += 1
                self._latency_by_gate["assoc"]["total_ms"] += self._last_latency_ms
                self._latency_by_gate["assoc"]["count"] += 1
            elif decision.mode == BattleGateMode.FOLLOW:
                self._gate_counts["follow"] += 1
                self._latency_by_gate["follow"]["total_ms"] += self._last_latency_ms
                self._latency_by_gate["follow"]["count"] += 1
            elif decision.mode == BattleGateMode.PLAN_LOOKUP:
                self._gate_counts["plan_lookup"] += 1
                self._latency_by_gate["plan_lookup"]["total_ms"] += self._last_latency_ms
                self._latency_by_gate["plan_lookup"]["count"] += 1
            elif decision.mode == BattleGateMode.PLAN_STEP:
                self._gate_counts["plan_step"] += 1
                self._latency_by_gate["plan_step"]["total_ms"] += self._last_latency_ms
                self._latency_by_gate["plan_step"]["count"] += 1
            elif decision.mode == BattleGateMode.WRITE:
                self._gate_counts["write"] += 1
                self._latency_by_gate["write"]["total_ms"] += self._last_latency_ms
                self._latency_by_gate["write"]["count"] += 1
            elif decision.mode == BattleGateMode.HALT:
                self._gate_counts["halt"] += 1

            skip_exempt_modes = {
                BattleGateMode.ASSOC,
                BattleGateMode.FOLLOW,
                BattleGateMode.PLAN_LOOKUP,
                BattleGateMode.PLAN_STEP,
            }
            if not decision.encode_flag and decision.mode not in skip_exempt_modes:
                self._gate_counts["skip"] += 1
                self._latency_by_gate["skip"]["total_ms"] += self._last_latency_ms
                self._latency_by_gate["skip"]["count"] += 1

        self._metrics = self._compute_metrics()

    def _compute_metrics(self) -> Dict[str, Dict[str, Optional[float]]]:
        total = self._gate_counts["total"]
        encode = self._gate_counts["encode"]
        assoc = self._gate_counts["assoc"]
        follow = self._gate_counts["follow"]
        write = self._gate_counts["write"]
        skip = self._gate_counts["skip"]
        plan_lookup = self._gate_counts["plan_lookup"]
        plan_step = self._gate_counts["plan_step"]
        query = assoc + follow + plan_lookup + plan_step

        fractions = {"encode": 0.0, "query": 0.0, "skip": 0.0}
        if total > 0:
            fractions = {
                "encode": encode / total,
                "query": query / total,
                "skip": skip / total,
            }

        baseline = total * self.core.config.encode_latency_ms
        actual = (
            encode * self.core.config.encode_latency_ms
            + assoc * self.core.config.assoc_latency_ms
            + follow * self.core.config.follow_latency_ms
            + write * self.core.config.write_latency_ms
            + skip * self.core.config.skip_latency_ms
            + plan_lookup * self.core.config.assoc_latency_ms
            + plan_step * self.core.config.follow_latency_ms
        )
        predicted = baseline / actual if actual > 1e-6 else None

        observed_total = self._latency_stats["total_ms"]
        observed = baseline / observed_total if observed_total > 1e-6 else None

        return {
            "fractions": fractions,
            "speedup": {"predicted": predicted, "observed": observed},
        }

    def profile_stats(self) -> Dict[str, Dict[str, float]]:
        profile: Dict[str, Dict[str, float]] = {}
        for key, stats in self._latency_by_gate.items():
            total = stats["total_ms"]
            count = stats["count"]
            profile[key] = {
                "total_ms": total,
                "count": count,
                "avg_ms": total / count if count > 0 else 0.0,
            }
        return profile












