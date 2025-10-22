"""
Planlet-driven overworld executor that coordinates skills and memory updates.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor

from src.srfbam.core import EncodedFrame, SRFBAMCore, SrfbamStepSummary

from src.plan.compiler import CompiledPlan, CompiledPlanlet, PlanCompilationError, PlanCompiler
from src.plan.planner_llm import PlanBundle, validate_plan_bundle
from src.overworld import OverworldExtractor, OverworldMemory
from src.overworld.memory.hybrid_adapter import HybridMemoryAdapter
from src.overworld.memory.slot_bank import SlotBank
from src.overworld.action_space import OverworldActionSpace
from src.overworld.recording import OverworldTraceRecorder
from src.overworld.encounter import EncounterBridge, EncounterRequest, EncounterResult
from src.overworld.skills import (
    BaseSkill,
    EncounterSkill,
    HealSkill,
    InteractSkill,
    MenuSkill,
    NavigateSkill,
    PickupSkill,
    ShopSkill,
    SkillProgress,
    SkillStatus,
    TalkSkill,
    UseItemSkill,
    WaitSkill,
)
from .events import PlanletEvent
from .plan_monitor import PlanMonitor


SkillFactory = Type[BaseSkill]


def _build_encoded_frame(observation: Mapping[str, object]) -> EncodedFrame:
    data = observation.get("overworld") if isinstance(observation, Mapping) else None
    if not isinstance(data, Mapping):
        data = observation
    if not isinstance(data, Mapping):
        data = {}

    map_info = data.get("map") or {}
    player = data.get("player") or {}
    menus = data.get("menus") or []

    map_id = str(map_info.get("id", "unknown"))
    map_hash = (hash(map_id) % 256) / 255.0
    tile = player.get("tile") or [0, 0]
    player_x = int(tile[0]) & 0xFF
    player_y = int(tile[1]) & 0xFF
    max_coord = 255.0
    facing = player.get("facing", 0)
    menu_open = any(bool(getattr(menu, "get", lambda key, default=None: menu[key] if key in menu else default)("open", False)) for menu in menus)
    npc_count = len(data.get("npcs", []))
    warp_count = len(data.get("warps", []))

    features = torch.tensor(
        [
            map_hash,
            player_x / max_coord,
            player_y / max_coord,
            float(menu_open),
            float(npc_count % 32) / 32.0,
            float(warp_count % 32) / 32.0,
            float(bool(player.get("in_battle"))),
            (hash(facing) % 256) / 255.0,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    grid = torch.zeros((40, 120), dtype=torch.long)
    context_key = f"overworld:map:{map_id}"
    return EncodedFrame(grid=grid, features=features, context_key=context_key, extra={})


@dataclass
class ExecutorStepResult:
    action: Dict[str, object]
    telemetry: Dict[str, object]
    status: str


@dataclass
class ExecutorState:
    plan_queue: Deque[CompiledPlanlet] = field(default_factory=deque)
    current_planlet: Optional[CompiledPlanlet] = None
    current_skill: Optional[BaseSkill] = None
    step_index: int = 0


class OverworldExecutor:
    """
    Converts planlets into skill executions over the overworld graph.
    """

    _TRACE_LIMIT = 8

    DEFAULT_SKILL_REGISTRY: Mapping[str, SkillFactory] = {
        "NavigateSkill": NavigateSkill,
        "HealSkill": HealSkill,
        "ShopSkill": ShopSkill,
        "TalkSkill": TalkSkill,
        "MenuSkill": MenuSkill,
        "UseItemSkill": UseItemSkill,
        "InteractSkill": InteractSkill,
        "PickupSkill": PickupSkill,
        "WaitSkill": WaitSkill,
        "EncounterSkill": EncounterSkill,
    }

    def __init__(
        self,
        *,
        memory: Optional[OverworldMemory] = None,
        extractor: Optional[OverworldExtractor] = None,
        core: Optional[SRFBAMCore] = None,
        frame_encoder: Optional[Callable[[Mapping[str, object]], EncodedFrame]] = None,
        skill_registry: Optional[Mapping[str, SkillFactory]] = None,
    ) -> None:
        self.memory = memory or OverworldMemory()
        self.extractor = extractor or OverworldExtractor()
        self.skill_registry = dict(skill_registry or self.DEFAULT_SKILL_REGISTRY)
        self.state = ExecutorState()
        self.plan_id: Optional[str] = None
        self.action_space = OverworldActionSpace()
        self.monitor = PlanMonitor()
        self.core = core or SRFBAMCore()
        self.slot_bank = SlotBank(device=self.core.device)
        self.hybrid_adapter = HybridMemoryAdapter()
        self.device = self.core.device
        self.frame_encoder = frame_encoder or _build_encoded_frame
        self._last_summary: Optional[SrfbamStepSummary] = None
        self._last_gate_mode: Optional[str] = None
        self._last_gate_view: Optional[str] = None
        self._recovery_limit = 1
        self._trace: Deque[Dict[str, object]] = deque(maxlen=self._TRACE_LIMIT)
        self._event_sink: Optional[Callable[[PlanletEvent], None]] = None
        self._events: List[PlanletEvent] = []
        self._replan_handler: Optional[Callable[[PlanletEvent], Optional[Mapping[str, object]]]] = None
        self._max_replans = 3
        self._replan_attempts = 0
        self._gate_counts: Dict[str, int] = {}
        self._latency_stats: Dict[str, float] = {}
        self._metrics: Dict[str, Dict[str, Optional[float]]] = {}
        self._latency_by_gate: Dict[str, Dict[str, float]] = {}
        self._last_latency_ms: float = 0.0
        self._reset_metrics()
        self._reset_view_counts()
        self._trace_recorder: Optional[OverworldTraceRecorder] = None
        self._battle_handler: Optional[Callable[[EncounterRequest], EncounterResult]] = None
        self._encounter_bridge = EncounterBridge()
        self._last_observation: Optional[Mapping[str, object]] = None
        self._plan_metadata: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #

    def register_event_sink(self, sink: Callable[[PlanletEvent], None]) -> None:
        """Register a callback invoked every time a planlet event fires."""

        self._event_sink = sink

    def register_replan_handler(
        self, handler: Callable[[PlanletEvent], Optional[Mapping[str, object]]]
    ) -> None:
        """Register a handler that can supply replacement planlets after failures."""

        self._replan_handler = handler

    def register_battle_handler(
        self, handler: Callable[[EncounterRequest], EncounterResult]
    ) -> None:
        """Register a handler that executes encounters in the battle arena."""

        self._battle_handler = handler

    def register_trace_recorder(self, recorder: OverworldTraceRecorder) -> None:
        """Attach a trace recorder that receives per-step payloads."""

        self._trace_recorder = recorder

    def plan_history(self) -> List[PlanletEvent]:
        """Return a copy of the emitted planlet events."""

        return list(self._events)

    def set_max_replans(self, value: int) -> None:
        """Limit the number of planner calls triggered from the executor."""

        self._max_replans = max(0, int(value))

    def set_plan_metadata(self, metadata: Optional[Mapping[str, object]]) -> None:
        """Attach metadata about the active plan for telemetry and events."""

        self._plan_metadata = dict(metadata or {})

    def plan_metadata(self) -> Dict[str, object]:
        """Return a copy of the active plan metadata."""

        return dict(self._plan_metadata)

    # ------------------------------------------------------------------ #
    # Plan management
    # ------------------------------------------------------------------ #

    def load_plan_bundle(
        self,
        bundle: PlanBundle,
        compiler: Optional[PlanCompiler] = None,
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        compiled = (compiler or PlanCompiler()).compile(bundle)
        plan_metadata = dict(metadata or self._bundle_metadata(bundle))
        self._set_plan(compiled, reset_replans=True, metadata=plan_metadata)

    def load_compiled_plan(
        self,
        plan: CompiledPlan,
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._set_plan(plan, reset_replans=True, metadata=dict(metadata or {}))

    def _set_plan(
        self,
        plan: CompiledPlan,
        *,
        reset_replans: bool,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.state = ExecutorState(plan_queue=deque(plan.planlets))
        self.plan_id = plan.plan_id
        self.monitor.reset()
        self.slot_bank.clear()
        self._trace.clear()
        self._reset_view_counts()
        self._events.clear()
        if reset_replans:
            self._encounter_bridge.reset()
            self._replan_attempts = 0
        self._last_observation = None
        self._plan_metadata = dict(metadata or {})

    # ------------------------------------------------------------------ #
    # Execution loop
    # ------------------------------------------------------------------ #

    def step(self, observation: Mapping[str, object]) -> ExecutorStepResult:
        self._last_observation = observation
        writes = self.extractor.extract(observation)
        for op in writes:
            self.memory.write(op)

        if self.state.current_planlet is None:
            self._advance_planlet()

        if self.state.current_planlet is None:
            legal_actions = ({"kind": "wait"},)
            mask = self.action_space.build_mask(legal_actions)
            gate_info = {
                "mode": "IDLE",
                "encode_flag": False,
                "view": "typed",
                "confidence": 1.0,
                "reason": "plan-complete",
                "raw": "IDLE",
            }
            telemetry = self.memory.make_step_telemetry(
                planlet_id="NONE",
                planlet_kind="NONE",
                gate=gate_info,
                action={"kind": "wait"},
                latency_ms=0.0,
                fallback_required=False,
            )
            core = telemetry["core"]
            core["legal_actions"] = [dict(a) for a in legal_actions]
            core["action_mask"] = list(mask)
            core["fractions"] = {"encode": 0.0, "query": 0.0, "skip": 0.0}
            core["speedup"] = {"predicted": None, "observed": None}
            overworld_data = telemetry["overworld"]
            overworld_data["action_index"] = 0
            overworld_data["frame_features"] = []
            overworld_data["memory"] = self.memory.summarise_nodes()
            overworld_data["view_usage"] = dict(self._view_counts)
            overworld_data["hybrid"] = {"projected": 0, "ingested": 0}
            overworld_data["planlet_kind"] = "NONE"
            overworld_data["status"] = "PLAN_COMPLETE"
            return ExecutorStepResult(action={"kind": "wait"}, telemetry=telemetry, status="PLAN_COMPLETE")

        skill = self.state.current_skill
        assert skill is not None
        planlet = self.state.current_planlet

        start_time = time.perf_counter()

        summary: Optional[SrfbamStepSummary] = None
        gate_mode: Optional[str] = None
        encode_flag = False
        gate_view = "typed"
        gate_raw = "WRITE"
        hybrid_stats = {"projected": 0, "ingested": 0}
        frame_features: List[float] = []

        if self.core is not None and self.frame_encoder is not None:
            encoded = self.frame_encoder(observation)
            encoded.grid = encoded.grid.to(self.device)
            encoded.features = encoded.features.to(self.device)
            frame_features = encoded.features.detach().to("cpu").reshape(-1).tolist()
            summary = self.core.encode_step(encoded)
            gate_raw = str(summary.gate_stats.get("decision", "WRITE"))
            gate_mode, encode_flag = self._map_gate_decision(gate_raw)
            gate_view = self._choose_view(gate_mode, summary)
            summary.gate_stats["view"] = gate_view
            self._last_summary = summary
            self._last_gate_mode = gate_mode
            self._last_gate_view = gate_view
            self.slot_bank.add_from_summary(
                summary,
                confidence=float(summary.gate_stats.get("confidence", 1.0)),
                metadata={"context": summary.context_key},
            )
            hybrid_stats = self._sync_hybrid_views(gate_view, summary)
        else:
            gate_mode = "FOLLOW"
            self._last_summary = None
            self._last_gate_mode = gate_mode
            self._last_gate_view = gate_view
            hybrid_stats = self._sync_hybrid_views(gate_view, summary)

        self._record_view_usage(gate_view)
        if hasattr(skill, "update_context"):
            try:
                skill.update_context(summary=summary, memory=self.memory, slot_bank=self.slot_bank)
            except TypeError:
                # Backwards compatibility for skills without keyword signatures.
                skill.update_context(summary)  # type: ignore[misc]

        legal_actions = tuple(skill.legal_actions(observation, self.memory))
        if not legal_actions:
            legal_actions = ({"kind": "wait"},)
        mask = self.action_space.build_mask(legal_actions)
        mask_values = [None if not math.isfinite(value) else float(value) for value in mask]

        action = skill.select_action(observation, self.memory)
        try:
            action_index = self.action_space.to_index(action)
        except KeyError:
            action = dict(legal_actions[0])
            action_index = self.action_space.to_index(action)
        if self.core is not None:
            self.core.set_last_action_index(action_index)

        progress = skill.progress(self.memory)
        reason = progress.reason if progress.status is SkillStatus.STALLED else None
        encounter_events: List[Dict[str, object]] = []
        if progress.status is SkillStatus.STALLED and reason == "RANDOM_BATTLE":
            progress, encounter_events = self._handle_encounter(planlet, observation)
            reason = progress.reason if progress.status is SkillStatus.STALLED else None

        gate_info: Dict[str, object] = {
            "mode": gate_mode or "FOLLOW",
            "raw": gate_raw,
            "encode_flag": encode_flag,
            "view": gate_view,
            "view_counts": dict(self._view_counts),
            "slots": len(self.slot_bank),
        }
        if summary is not None:
            gate_info["stats"] = dict(summary.gate_stats)
            gate_info.setdefault("confidence", summary.gate_stats.get("confidence"))
            gate_info.setdefault("reason", summary.gate_stats.get("reason"))

        telemetry = self.memory.make_step_telemetry(
            planlet_id=planlet.spec.id,
            planlet_kind=planlet.spec.kind,
            gate=gate_info,
            action=action,
            latency_ms=0.0,
            fallback_required=False,
        )
        core = telemetry["core"]
        overworld_data = telemetry["overworld"]
        core["legal_actions"] = [dict(a) for a in legal_actions]
        core["action_mask"] = mask_values
        overworld_data["frame_features"] = list(frame_features)
        overworld_data["action_index"] = int(action_index)
        overworld_data["memory"] = self.memory.summarise_nodes()
        overworld_data["view_usage"] = dict(self._view_counts)
        overworld_data["hybrid"] = dict(hybrid_stats)
        menu_snapshot = self._extract_menu_snapshot(observation)
        if menu_snapshot is not None:
            overworld_data["menu_state"] = int(menu_snapshot.get("state", 0))
            if "cursor" in menu_snapshot:
                overworld_data["menu_cursor"] = menu_snapshot["cursor"]
            overworld_data["is_menu"] = bool(menu_snapshot.get("is_menu", False))
            overworld_data["menu_snapshot"] = menu_snapshot

        if isinstance(skill, NavigateSkill):
            overworld_data["navigate"] = {
                "target": list(skill.target) if skill.target else None,
                "path_remaining": len(skill.path),
            }

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self._update_metrics(gate_mode, encode_flag, latency_ms)

        core["latency_ms"] = latency_ms
        core["fractions"] = dict(self._metrics["fractions"])
        core["speedup"] = dict(self._metrics["speedup"])
        core["fallback_required"] = bool(core.get("fallback_required", False))
        if encounter_events:
            overworld_data["encounter"] = [self._convert_for_json(event) for event in encounter_events]

        recovery_reason = self.monitor.record_step(planlet.spec.id, skill, self.memory)
        if recovery_reason == "BLOCKED_PATH" and isinstance(skill, NavigateSkill):
            if self.monitor.recovery_count(planlet.spec.id) <= self._recovery_limit:
                skill.on_enter(planlet.spec, self.memory)
                overworld_data["recovery"] = {"reason": recovery_reason, "action": "replan"}
            else:
                progress = SkillProgress(status=SkillStatus.STALLED, reason=recovery_reason)
                reason = recovery_reason

        status = "IN_PROGRESS"
        step_index = self.state.step_index
        planlet_id = planlet.spec.id
        planlet_kind = planlet.spec.kind
        overworld_data["planlet_kind"] = planlet_kind
        plan_context = {
            "id": self.plan_id,
            "planlet_id": planlet_id,
            "planlet_kind": planlet_kind,
        }
        if self._plan_metadata:
            plan_context.update(self._plan_metadata)
        telemetry.setdefault("core", {})["plan"] = plan_context
        if progress.status is SkillStatus.SUCCEEDED:
            skill.on_exit(self.memory)
            self.monitor.clear_planlet(planlet.spec.id)
            self.state.current_planlet = None
            self.state.current_skill = None
            status = "PLANLET_COMPLETE"
            overworld_data["status"] = status
            self._trace.append(self._convert_for_json(telemetry))
            event = self._build_event(planlet, status, reason, telemetry, step_index)
            self._emit_event(event)
            self._trace.clear()
        elif progress.status is SkillStatus.STALLED:
            status = "PLANLET_STALLED"
            self.monitor.clear_planlet(planlet.spec.id)
            overworld_data["status"] = status
            self._trace.append(self._convert_for_json(telemetry))
            event = self._build_event(planlet, status, reason, telemetry, step_index)
            self._emit_event(event)
            self._trace.clear()
            self._handle_replan(event)
            self.state.current_planlet = None
            self.state.current_skill = None
        else:
            overworld_data["status"] = status
            self._trace.append(self._convert_for_json(telemetry))

        hint_payload = None
        if hasattr(skill, "planner_hint") and callable(getattr(skill, "planner_hint")):
            hint_payload = skill.planner_hint()
        if hint_payload:
            overworld_data.setdefault("skill", {})["hint"] = hint_payload

        recovery_hint = None
        if hasattr(skill, "recovery_hint") and callable(getattr(skill, "recovery_hint")):
            recovery_hint = skill.recovery_hint()
        if recovery_hint:
            overworld_data.setdefault("recovery", recovery_hint)

        observation_payload = self._convert_for_json(observation)
        telemetry_payload = self._convert_for_json(telemetry)
        self._record_trace(
            planlet_id=planlet_id,
            planlet_kind=planlet_kind,
            status=status,
            step_index=step_index,
            observation=observation_payload,
            telemetry=telemetry_payload,
            reason=reason,
        )

        self.state.step_index += 1
        return ExecutorStepResult(action=action, telemetry=telemetry, status=status)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _advance_planlet(self) -> None:
        if not self.state.plan_queue:
            self.state.current_planlet = None
            self.state.current_skill = None
            return
        next_planlet = self.state.plan_queue.popleft()
        skill_cls = self.skill_registry.get(next_planlet.skill)
        if skill_cls is None:
            raise ValueError(f"No skill registered for '{next_planlet.skill}'")
        skill = skill_cls()
        skill.on_enter(next_planlet.spec, self.memory)
        self.state.current_planlet = next_planlet
        self.state.current_skill = skill
        self.monitor.notify_new_planlet(next_planlet.spec.id)
        self._trace.clear()

    def _build_event(
        self,
        planlet: CompiledPlanlet,
        status: str,
        reason: Optional[str],
        telemetry: Dict[str, object],
        step_index: int,
    ) -> PlanletEvent:
        trace = [dict(entry) for entry in self._trace]
        return PlanletEvent(
            plan_id=self.plan_id,
            planlet_id=planlet.spec.id,
            planlet_kind=planlet.spec.kind,
            status=status,
            reason=reason,
            step_index=step_index,
            telemetry=dict(telemetry),
            trace=trace,
            metadata=dict(self._plan_metadata),
        )

    def _emit_event(self, event: PlanletEvent) -> None:
        self._events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)

    def _handle_replan(self, event: PlanletEvent) -> None:
        if self._replan_handler is None:
            return
        if self._replan_attempts >= self._max_replans:
            return

        response = self._replan_handler(event)
        if response is None:
            self._replan_attempts += 1
            return

        if isinstance(response, PlanBundle):
            bundle = response
        elif isinstance(response, Mapping):
            bundle = validate_plan_bundle(response)
        else:
            raise TypeError("Replan handler must return PlanBundle or mapping.")

        compiler = PlanCompiler()
        try:
            compiled = compiler.compile(bundle)
        except PlanCompilationError:
            raise

        metadata = self._bundle_metadata(bundle)
        self._replan_attempts += 1
        self._set_plan(compiled, reset_replans=False, metadata=metadata)

    def _bundle_metadata(self, bundle: PlanBundle) -> Dict[str, object]:
        raw = getattr(bundle, "raw", None)
        if isinstance(raw, Mapping):
            meta = raw.get("metadata")
            if isinstance(meta, Mapping):
                return dict(meta)
        return {}

    def _extract_menu_snapshot(self, observation: Mapping[str, object]) -> Optional[Dict[str, object]]:
        overworld = observation.get("overworld") if isinstance(observation, Mapping) else None
        if not isinstance(overworld, Mapping):
            return None
        extras = overworld.get("extra")
        menus = overworld.get("menus")
        state = self._coerce_int(overworld.get("menu_state"))
        cursor = self._coerce_int(overworld.get("menu_cursor"))
        joy_ignore = self._coerce_int(overworld.get("joy_ignore"))
        if isinstance(extras, Mapping):
            if state is None:
                state = self._coerce_int(extras.get("menu_state"))
            if cursor is None:
                cursor = self._coerce_int(extras.get("menu_cursor"))
            if joy_ignore is None:
                joy_ignore = self._coerce_int(extras.get("joy_ignore"))
        has_menu_struct = isinstance(menus, (list, tuple)) and len(menus) > 0
        if state is None and cursor is None and not has_menu_struct:
            return None
        snapshot: Dict[str, object] = {}
        state_value = 0 if state is None else int(state)
        snapshot["state"] = state_value
        if cursor is not None:
            snapshot["cursor"] = int(cursor)
        if joy_ignore is not None:
            snapshot["joy_ignore"] = int(joy_ignore)
        if has_menu_struct:
            snapshot["menus"] = [dict(menu) if isinstance(menu, Mapping) else menu for menu in menus]
        snapshot["is_menu"] = bool(state_value) or has_menu_struct
        return snapshot

    @staticmethod
    def _coerce_int(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    def _reset_view_counts(self) -> None:
        self._view_counts = {"typed": 0, "slots": 0}

    def _reset_metrics(self) -> None:
        self._gate_counts = {"total": 0, "encode": 0, "query": 0, "skip": 0}
        self._latency_stats = {"total_ms": 0.0, "steps": 0}
        self._metrics = {
            "fractions": {"encode": 0.0, "query": 0.0, "skip": 0.0},
            "speedup": {"predicted": None, "observed": None},
        }
        self._latency_by_gate = {
            "encode": {"total_ms": 0.0, "count": 0},
            "query": {"total_ms": 0.0, "count": 0},
            "skip": {"total_ms": 0.0, "count": 0},
        }
        self._last_latency_ms = 0.0

    def _handle_encounter(
        self,
        planlet: CompiledPlanlet,
        observation: Mapping[str, object],
    ) -> Tuple[SkillProgress, List[Dict[str, object]]]:
        events: List[Dict[str, object]] = []
        mode = str(planlet.spec.args.get("mode", "escape_first")).lower()
        if mode not in {"escape_first", "fast_fight", "conserve_pp", "win_if_easy"}:
            mode = "escape_first"
        timeout_steps = max(1, int(planlet.spec.timeout_steps or 300))
        rng_state = b""
        request, entry = self._encounter_bridge.build_request(
            observation=observation,
            memory=self.memory,
            mode=mode,
            timeout_steps=timeout_steps,
            rng_state=rng_state,
            party_summary=None,
        )
        events.append(entry)

        if self._battle_handler is None:
            progress = SkillProgress(status=SkillStatus.STALLED, reason="RANDOM_BATTLE")
            return progress, events

        try:
            result = self._battle_handler(request)
        except Exception as exc:  # pragma: no cover - defensive
            events.append(self._encounter_bridge.fail(reason=str(exc)))
            return SkillProgress(status=SkillStatus.STALLED, reason="ENCOUNTER_ERROR"), events

        if result.snapshot_return is None:
            result.snapshot_return = request.snapshot_overworld

        invariants, exit_event = self._encounter_bridge.complete(
            result=result,
            observation=observation,
            memory=self.memory,
        )
        events.append(exit_event)

        status = result.status
        if status in {"WIN", "ESCAPED"}:
            if all(invariants.values()):
                return SkillProgress(status=SkillStatus.IN_PROGRESS), events
            return SkillProgress(status=SkillStatus.STALLED, reason="ENCOUNTER_INVARIANT"), events
        if status == "TIMEOUT":
            return SkillProgress(status=SkillStatus.STALLED, reason="ENCOUNTER_TIMEOUT"), events
        if status == "FAINTED":
            return SkillProgress(status=SkillStatus.STALLED, reason="PARTY_FAINTED"), events
        return SkillProgress(status=SkillStatus.STALLED, reason="ENCOUNTER_ERROR"), events

    def _record_trace(
        self,
        *,
        planlet_id: Optional[str],
        planlet_kind: Optional[str],
        status: str,
        step_index: int,
        observation: object,
        telemetry: object,
        reason: Optional[str],
    ) -> None:
        if self._trace_recorder is None or planlet_id is None:
            return
        context = {
            "domain": "overworld",
            "plan": {
                "id": self.plan_id,
                "planlet_id": planlet_id,
                "planlet_kind": planlet_kind,
            },
            "status": status,
            "step_index": int(step_index),
        }
        if self._plan_metadata:
            context["plan"].update(dict(self._plan_metadata))
        if reason:
            context["reason"] = reason
        payload = {
            "source": "sr-fbam.overworld.executor",
            "context": context,
            "observation": observation,
            "telemetry": telemetry,
        }
        self._trace_recorder.record(payload)

    def _convert_for_json(self, value: object) -> object:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Mapping):
            return {str(key): self._convert_for_json(sub_value) for key, sub_value in value.items()}
        if isinstance(value, (list, tuple, set, deque)):
            return [self._convert_for_json(item) for item in value]
        if isinstance(value, bytes):
            return list(value)
        if hasattr(value, "_asdict"):
            return self._convert_for_json(value._asdict())
        return str(value)

    def _choose_view(self, gate_mode: Optional[str], summary: Optional[SrfbamStepSummary]) -> str:
        if gate_mode is None:
            return "typed"

        context = summary.context_key if summary is not None else None
        if gate_mode in {"ASSOC", "FOLLOW", "WRITE"} and context is not None:
            try:
                next(self.slot_bank.match_metadata("context", context))
                return "slots"
            except StopIteration:
                pass
        return "typed"

    def _record_view_usage(self, view: str) -> None:
        key = view if view in self._view_counts else "typed"
        self._view_counts[key] = self._view_counts.get(key, 0) + 1

    def _sync_hybrid_views(
        self,
        gate_view: str,
        summary: Optional[SrfbamStepSummary],
    ) -> Dict[str, int]:
        context = summary.context_key if summary is not None else None
        stats = {"projected": 0, "ingested": 0}
        if gate_view == "slots":
            stats["projected"] = self.hybrid_adapter.project_slots_to_graph(
                self.memory, self.slot_bank, context=context
            )
        else:
            stats["ingested"] = self.hybrid_adapter.ingest_graph_slots(
                self.memory, self.slot_bank, context=context
            )
        return stats

    def _map_gate_decision(self, decision: str) -> Tuple[str, bool]:
        decision = decision.upper()
        if decision == "EXTRACT":
            return "HALT", True
        if decision == "CACHE_HIT":
            return "ASSOC", False
        if decision == "REUSE":
            return "FOLLOW", False
        if decision == "WRITE":
            return "WRITE", False
        return decision or "FOLLOW", False

    def _update_metrics(self, mode: Optional[str], encode_flag: bool, latency_ms: float) -> None:
        self._last_latency_ms = float(latency_ms)
        self._latency_stats["total_ms"] += self._last_latency_ms
        self._latency_stats["steps"] += 1

        if mode is None:
            return

        self._gate_counts["total"] += 1
        category = None
        if encode_flag:
            category = "encode"
        elif mode in {"ASSOC", "FOLLOW"}:
            category = "query"
        else:
            category = "skip"

        self._gate_counts[category] += 1
        slot = self._latency_by_gate[category]
        slot["total_ms"] += self._last_latency_ms
        slot["count"] += 1

        self._metrics = self._compute_metrics()

    def _compute_metrics(self) -> Dict[str, Dict[str, Optional[float]]]:
        total = self._gate_counts["total"]
        encode = self._gate_counts["encode"]
        query = self._gate_counts["query"]
        skip = self._gate_counts["skip"]

        fractions = {"encode": 0.0, "query": 0.0, "skip": 0.0}
        if total > 0:
            fractions = {
                "encode": encode / total,
                "query": query / total,
                "skip": skip / total,
            }

        predicted = None
        observed = None
        if total > 0 and self.core is not None:
            baseline = total * self.core.config.encode_latency_ms
            actual = (
                encode * self.core.config.encode_latency_ms
                + query * self.core.config.assoc_latency_ms
                + skip * self.core.config.skip_latency_ms
            )
            if actual > 1e-6:
                predicted = baseline / actual
            observed_total = self._latency_stats["total_ms"]
            if observed_total > 1e-6:
                observed = baseline / observed_total

        return {"fractions": fractions, "speedup": {"predicted": predicted, "observed": observed}}









