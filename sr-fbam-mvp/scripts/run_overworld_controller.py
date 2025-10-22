#!/usr/bin/env python3
"""Run the SR-FBAM overworld controller end-to-end on PyBoy."""

from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.capture_overworld_telemetry import _policy_bootstrap  # reuse menu navigation helper

from src.llm.llm_client import LLMConfig, MockLLMClient, create_llm_client
from src.llm.planlets import FakeOverworldLLM, PlanletProposer, PlanletService
from src.middleware.pokemon_adapter import PokemonAction
from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter
from src.overworld.env.overworld_adapter import OverworldAdapter
from src.overworld.recording import OverworldTraceRecorder
from src.plan.cache import PlanCache
from src.plan.planner_llm import PlanBundle, PlanletSpec
from src.plan.compiler import PlanCompilationError, PlanCompiler
from src.plan.storage import PlanletStore
from src.srfbam.tasks.overworld import OverworldExecutor


LOGGER = logging.getLogger("run_overworld_controller")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the SR-FBAM overworld executor inside PyBoy.")
    parser.add_argument("--rom", type=Path, default=Path("Pokemon Blue.gb"), help="Path to ROM.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of controller steps to run.")
    parser.add_argument(
        "--telemetry-out",
        type=Path,
        help="Optional JSONL file to record per-step telemetry.",
    )
    parser.add_argument(
        "--policy-boot",
        action="store_true",
        help="Use menu policy to reach the overworld instead of deterministic boot macros.",
    )
    parser.add_argument("--window", type=str, default="null", help="PyBoy window backend (null|SDL2).")
    parser.add_argument("--frames-per-step", type=int, default=12, help="Frames to wait on wait actions.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for exploration targets.")
    parser.add_argument(
        "--planner-backend",
        type=str,
        default="fake",
        choices=["none", "fake", "mock", "openai", "anthropic"],
        help="Planner backend to use for planlet generation.",
    )
    parser.add_argument("--planner-model", type=str, help="Override planner model identifier.")
    parser.add_argument("--planner-api-key", type=str, help="API key for planner backend (env fallback).")
    parser.add_argument("--planner-base-url", type=str, help="Optional API base URL override.")
    parser.add_argument("--planner-store", type=Path, help="Directory to persist emitted planlets.")
    parser.add_argument(
        "--planner-cache-size",
        type=int,
        default=64,
        help="Maximum cache entries retained for planlets.",
    )
    parser.add_argument(
        "--planner-cache-ttl",
        type=float,
        default=900.0,
        help="Cache TTL in seconds (<=0 disables expiry).",
    )
    parser.add_argument(
        "--disable-planner-cache",
        action="store_true",
        help="Disable runtime planlet cache even if configured.",
    )
    parser.add_argument(
        "--planner-nearby-limit",
        type=int,
        default=5,
        help="Entity limit when summarising overworld context for the planner.",
    )
    parser.add_argument(
        "--planner-allow-search",
        action="store_true",
        help="Allow planner backend to issue retrieval/search tool calls.",
    )
    return parser.parse_args()


def to_executor_observation(observation, adapter: PyBoyPokemonAdapter) -> Dict[str, object]:
    data = {
        "frame": observation.frame,
        "overworld": dict(observation.overworld),
        "ram": adapter.snapshot_overworld_ram(),
    }
    return data


class RandomPlanManager:
    """Simple exploration plan generator that produces NAVIGATE_TO planlets."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._counter = 0

    def build_bundle(self, observation: Dict[str, object]) -> PlanBundle:
        overworld = observation.get("overworld", {}) or {}
        ram = observation.get("ram", {}) or {}

        map_byte = ram.get(0xD35E) or overworld.get("area_id", 0)
        map_id = f"{int(map_byte) & 0xFF:02X}"

        player_x = int(overworld.get("x", 0))
        player_y = int(overworld.get("y", 0))
        player = overworld.get("player")
        if isinstance(player, dict):
            tile = player.get("tile")
            if isinstance(tile, (list, tuple)) and len(tile) >= 2:
                player_x = int(tile[0])
                player_y = int(tile[1])

        dx = self._rng.choice([-3, -2, -1, 0, 1, 2, 3])
        dy = self._rng.choice([-3, -2, -1, 0, 1, 2, 3])
        if dx == 0 and dy == 0:
            dx = 1
        target = [max(0, min(255, player_x + dx)), max(0, min(255, player_y + dy))]

        planlet_id = f"nav_{self._counter}"
        planlet = PlanletSpec(
            id=planlet_id,
            kind="NAVIGATE_TO",
            args={"target": {"map": map_id, "tile": target}},
            timeout_steps=240,
        )
        self._counter += 1
        return PlanBundle(
            plan_id=f"auto_{int(time.time())}_{self._counter}",
            goal=f"Explore {map_id} -> {target}",
            planlets=[planlet],
        )


class PlanCoordinator:
    """Bridge overworld executor HALTs to planner/LLM requests with random fallback."""

    def __init__(
        self,
        *,
        executor: "OverworldExecutor",
        service: Optional[PlanletService],
        fallback: RandomPlanManager,
        allow_search: bool,
        nearby_limit: int,
        backend_label: str,
        logger: logging.Logger,
    ) -> None:
        self.executor = executor
        self.service = service
        self.fallback = fallback
        self.allow_search = bool(allow_search)
        self.nearby_limit = max(1, int(nearby_limit))
        self._backend_label = backend_label
        self._logger = logger
        self._plan_seq = 0
        self._planlet_seq = 0
        self._last_source = backend_label if service is not None else "random"

    def has_planner(self) -> bool:
        return self.service is not None

    def describe_last_source(self) -> str:
        return self._last_source

    def request_bundle(self, observation: Dict[str, object], *, reason: str) -> PlanBundle:
        if self.service is None:
            return self._fallback_bundle(observation, reason, "planner-disabled")
        try:
            proposal = self.service.request_overworld_planlet(
                self.executor.memory,
                nearby_limit=self.nearby_limit,
                allow_search=self.allow_search,
            )
        except Exception:
            self._logger.exception("Planlet service failed (reason=%s)", reason)
            return self._fallback_bundle(observation, reason, "planner-error")
        bundle = self._bundle_from_proposal(proposal, reason)
        kinds = [planlet.kind for planlet in bundle.planlets]
        unsupported = [kind for kind in kinds if kind not in PlanCompiler.DEFAULT_REGISTRY]
        if unsupported:
            self._logger.warning(
                "Planner emitted unsupported kinds %s; falling back to random plan.",
                unsupported,
            )
            return self._fallback_bundle(observation, reason, "planner-unsupported")
        return bundle

    def fallback_bundle(self, observation: Dict[str, object], *, reason: str) -> PlanBundle:
        return self._fallback_bundle(observation, reason, "fallback-random")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _bundle_from_proposal(self, proposal: Any, reason: str) -> PlanBundle:
        payload = dict(getattr(proposal, "planlet", {}) or {})
        planlet_id = str(
            payload.get("id")
            or payload.get("planlet_id")
            or self._next_planlet_id()
        )
        payload["id"] = planlet_id
        kind = str(payload.get("kind") or "NAVIGATE_TO").upper()
        args = dict(payload.get("args") or {})
        pre = list(payload.get("pre") or payload.get("preconditions") or [])
        post = list(payload.get("post") or payload.get("postconditions") or [])
        hints = dict(payload.get("hints") or {})
        script = list(payload.get("script") or [])
        recovery = list(payload.get("recovery") or payload.get("recovery_plan") or [])
        timeout_raw = payload.get("timeout_steps", payload.get("timeout"))
        try:
            timeout_steps = max(1, int(timeout_raw)) if timeout_raw is not None else 600
        except (TypeError, ValueError):
            timeout_steps = 600

        spec = PlanletSpec(
            id=planlet_id,
            kind=kind,
            args=args,
            pre=pre,
            post=post,
            hints=hints,
            script=script,
            timeout_steps=timeout_steps,
            recovery=recovery,
        )

        plan_id = str(payload.get("plan_id") or self._next_plan_id())
        raw_bundle = {
            "planlet": payload,
            "reason": reason,
            "source": getattr(proposal, "source", "llm"),
            "cache_hit": bool(getattr(proposal, "cache_hit", False)),
            "cache_key": getattr(proposal, "cache_key", None),
        }
        token_usage = getattr(proposal, "token_usage", None)
        if token_usage:
            raw_bundle["token_usage"] = token_usage
        retrieved_docs = getattr(proposal, "retrieved_docs", None)
        if retrieved_docs:
            raw_bundle["retrieved_docs"] = retrieved_docs
        summary = getattr(proposal, "summary", None)
        if summary is not None and hasattr(summary, "to_payload"):
            try:
                raw_bundle["summary"] = summary.to_payload()
            except Exception:  # pragma: no cover - defensive
                pass

        cache_hit = raw_bundle.get("cache_hit")
        source = raw_bundle.get("source", "llm")
        suffix = "[cache]" if cache_hit else ""
        self._last_source = f"{self._backend_label}:{source}{suffix}"

        goal = payload.get("goal")
        metadata = {
            "plan_source": self._last_source,
            "planner_origin": source,
            "planner_reason": reason,
            "planner_cache_hit": bool(cache_hit),
        }
        cache_key = raw_bundle.get("cache_key")
        if cache_key:
            metadata["planner_cache_key"] = cache_key
        if token_usage:
            metadata["planner_token_usage"] = token_usage
        raw_bundle["metadata"] = metadata
        return PlanBundle(plan_id=plan_id, goal=goal, planlets=[spec], raw=raw_bundle)

    def _fallback_bundle(self, observation: Dict[str, object], reason: str, tag: str) -> PlanBundle:
        bundle = self.fallback.build_bundle(observation)
        self._last_source = tag
        raw_bundle = dict(bundle.raw)
        raw_bundle.update(
            {
                "source": "random",
                "reason": reason,
                "cache_hit": False,
                "cache_key": None,
                "metadata": {
                    "plan_source": tag,
                    "planner_origin": "random",
                    "planner_reason": reason,
                    "planner_cache_hit": False,
                },
            }
        )
        return PlanBundle(plan_id=bundle.plan_id, goal=bundle.goal, planlets=list(bundle.planlets), raw=raw_bundle)

    def _next_plan_id(self) -> str:
        self._plan_seq += 1
        return f"ow_plan_{int(time.time() * 1000)}_{self._plan_seq}"

    def _next_planlet_id(self) -> str:
        self._planlet_seq += 1
        return f"ow_planlet_{int(time.time() * 1000)}_{self._planlet_seq}"


def action_to_pokemon(action: Dict[str, object], *, frames_per_step: int) -> PokemonAction:
    kind = str(action.get("kind", "")).lower()
    if kind == "button":
        label = str(action.get("label", "")) or "A"
        frames = int(action.get("frames", 6))
        return PokemonAction("SCRIPT", {"inputs": [label.upper()], "frames": frames})
    if kind == "script":
        inputs = [str(value).upper() for value in action.get("inputs", [])]
        if not inputs:
            inputs = [str(action.get("label", "A")).upper()]
        frames = int(action.get("frames", 6))
        return PokemonAction("SCRIPT", {"inputs": inputs, "frames": frames})
    if kind == "wait":
        frames = int(action.get("frames", frames_per_step))
        return PokemonAction("WAIT", {"frames": frames})
    if kind == "menu":
        label = str(action.get("label", "A")).upper()
        frames = int(action.get("frames", 6))
        return PokemonAction("SCRIPT", {"inputs": [label], "frames": frames})
    if kind == "button_sequence":
        inputs = [str(value).upper() for value in action.get("inputs", [])]
        frames = int(action.get("frames", 6))
        return PokemonAction("SCRIPT", {"inputs": inputs, "frames": frames})
    return PokemonAction("WAIT", {"frames": frames_per_step})


def build_planlet_service(
    args: argparse.Namespace,
) -> tuple[Optional[PlanletService], str]:
    backend = (args.planner_backend or "none").lower()
    backend_label = backend

    if backend == "none":
        return None, "random"

    proposer = PlanletProposer()
    store = PlanletStore(args.planner_store) if args.planner_store else None

    cache: Optional[PlanCache] = None
    if not args.disable_planner_cache and args.planner_cache_size > 0:
        ttl = None if args.planner_cache_ttl <= 0 else float(args.planner_cache_ttl)
        cache = PlanCache(max_size=int(args.planner_cache_size), ttl_seconds=ttl)

    client: Any
    if backend == "fake":
        client = FakeOverworldLLM()
        backend_label = "fake"
    elif backend == "mock":
        client = MockLLMClient(LLMConfig(model="mock"))
        backend_label = "mock"
    else:
        model = args.planner_model
        if not model:
            if backend == "openai":
                model = "gpt-5"
            elif backend == "anthropic":
                model = "claude-3-sonnet-20240229"
            else:
                model = "gpt-5"

        api_key = args.planner_api_key
        if not api_key:
            if backend == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif backend == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                api_key = os.getenv("PLANNER_API_KEY")

        base_url = args.planner_base_url
        if not base_url and backend == "openai":
            base_url = os.getenv("OPENAI_BASE_URL")

        config = LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        config.response_format = {"type": "json_object"}
        client = create_llm_client(config)
        backend_label = backend

    service = PlanletService(proposer=proposer, client=client, store=store, cache=cache)
    return service, backend_label


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = PyBoyConfig(rom_path=str(args.rom), window_type=args.window)
    adapter = PyBoyPokemonAdapter(cfg)
    overworld = OverworldAdapter(adapter)

    executor = OverworldExecutor()
    rng = random.Random(args.seed)
    random_manager = RandomPlanManager(rng)
    planner_service, backend_label = build_planlet_service(args)
    compiler = PlanCompiler()
    plan_coordinator = PlanCoordinator(
        executor=executor,
        service=planner_service,
        fallback=random_manager,
        allow_search=args.planner_allow_search,
        nearby_limit=args.planner_nearby_limit,
        backend_label=backend_label,
        logger=LOGGER,
    )
    if planner_service is None:
        LOGGER.info("Planner backend disabled; using random exploration planlets.")
    else:
        LOGGER.info("Planner backend initialised (%s).", backend_label)

    recorder: Optional[OverworldTraceRecorder] = None
    if args.telemetry_out:
        recorder = OverworldTraceRecorder(args.telemetry_out)
        executor.register_trace_recorder(recorder)

    running = True

    def handle_signal(signum, frame):  # type: ignore[override]
        nonlocal running
        LOGGER.info("Received signal %s, stopping run...", signum)
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    observation = overworld.reset()
    if args.policy_boot:
        observation = _policy_bootstrap(overworld, observation, frames_per_step=args.frames_per_step)

    obs_payload = to_executor_observation(observation, adapter)
    last_observation = obs_payload

    def load_new_plan(reason: str, current_observation: Dict[str, object]) -> None:
        bundle = plan_coordinator.request_bundle(current_observation, reason=reason)
        if bundle is None:
            LOGGER.error("Planner returned no plan bundle; skipping load (reason=%s).", reason)
            return
        plan_metadata = dict(getattr(bundle, "raw", {}).get("metadata") or {})
        plan_metadata.setdefault("planner_reason", reason)
        plan_metadata.setdefault("plan_source", plan_coordinator.describe_last_source())
        try:
            compiled = compiler.compile(bundle)
        except PlanCompilationError:
            LOGGER.exception(
                "Failed to compile plan bundle; attempting random fallback (reason=%s).",
                reason,
            )
            fallback_bundle = plan_coordinator.fallback_bundle(
                current_observation, reason=f"{reason}:compile_error"
            )
            try:
                plan_metadata = dict(getattr(fallback_bundle, "raw", {}).get("metadata") or {})
                plan_metadata.setdefault("planner_reason", f"{reason}:compile_error")
                plan_metadata.setdefault("plan_source", plan_coordinator.describe_last_source())
                compiled = compiler.compile(fallback_bundle)
            except PlanCompilationError:
                LOGGER.exception("Fallback bundle also failed compilation; skipping load.")
                return
        executor.load_compiled_plan(compiled, metadata=plan_metadata)
        LOGGER.info(
            "Loaded plan %s (%d planlets) source=%s reason=%s",
            compiled.plan_id,
            len(compiled.planlets),
            plan_coordinator.describe_last_source(),
            reason,
        )

    def ensure_plan(current_observation: Dict[str, object], *, reason: str) -> None:
        if executor.state.current_planlet is not None or executor.state.plan_queue:
            return
        load_new_plan(reason, current_observation)

    def replan_handler(event) -> PlanBundle:
        status = getattr(event, "status", None)
        reason = getattr(event, "reason", None)
        LOGGER.info(
            "Executor requested replan (planlet=%s status=%s reason=%s)",
            getattr(event, "planlet_id", None),
            status,
            reason,
        )
        tag_components = [str(part) for part in (status, reason) if part]
        tag = ":".join(tag_components) if tag_components else "replan"
        return plan_coordinator.request_bundle(last_observation, reason=tag)

    executor.register_replan_handler(replan_handler)
    if recorder is not None:
        def _plan_event_logger(event) -> None:
            plan_payload = {
                "id": event.plan_id,
                "planlet_id": event.planlet_id,
                "planlet_kind": event.planlet_kind,
            }
            metadata = getattr(event, "metadata", None)
            if isinstance(metadata, dict):
                plan_payload.update(metadata)
            context = {
                "domain": "overworld",
                "status": event.status,
                "step_index": event.step_index,
                "plan": plan_payload,
            }
            if event.reason:
                context["reason"] = event.reason
            payload = {
                "source": "overworld.controller.plan_event",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context,
                "telemetry": event.telemetry,
            }
            if event.trace:
                payload["observation"] = {"trace": event.trace}
            try:
                recorder.record(payload)
            except Exception:
                LOGGER.exception("Failed to record plan event telemetry.")

        executor.register_event_sink(_plan_event_logger)
    ensure_plan(obs_payload, reason="initial")

    step = 0
    try:
        while running and step < args.steps:
            try:
                result = executor.step(obs_payload)
            except Exception:
                LOGGER.exception("Executor step failed; forcing replan.")
                load_new_plan("executor_error", obs_payload)
                continue

            pokemon_action = action_to_pokemon(result.action, frames_per_step=args.frames_per_step)
            try:
                observation = overworld.step(pokemon_action)
            except Exception:
                LOGGER.exception("PyBoy adapter step failed; forcing replan.")
                time.sleep(0.1)
                load_new_plan("adapter_error", obs_payload)
                continue

            obs_payload = to_executor_observation(observation, adapter)
            last_observation = obs_payload

            if result.status in {"PLANLET_COMPLETE", "PLANLET_STALLED", "PLAN_COMPLETE"}:
                ensure_plan(obs_payload, reason=f"executor_status:{result.status}")

            step += 1
    finally:
        if recorder is not None:
            recorder.close()
        adapter.close()

    LOGGER.info("Completed %s steps", step)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual testing script
    sys.exit(main())
