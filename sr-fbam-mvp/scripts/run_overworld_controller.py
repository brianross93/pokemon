#!/usr/bin/env python3
"""Run the SR-FBAM overworld controller end-to-end on PyBoy."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from concurrent.futures import ThreadPoolExecutor, Future

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.capture_overworld_telemetry import _policy_bootstrap  # reuse menu navigation helper

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from src.llm.llm_client import LLMConfig, MockLLMClient, create_llm_client
from src.llm.planlets import FakeOverworldLLM, PlanletProposer, PlanletService
from src.middleware.pokemon_adapter import PokemonAction
from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter
from src.overworld.env.overworld_adapter import OverworldAdapter, OverworldObservation
from src.overworld.recording import OverworldTraceRecorder
from src.overworld.recording.trace_recorder import TraceValidationError
from src.overworld.ram_map import DEFAULT_OVERWORLD_RAM_MAP
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
    parser.add_argument(
        "--emulation-speed",
        type=float,
        default=1.0,
        help="PyBoy emulation speed multiplier (1.0 approximates real-time, 0 for unlimited).",
    )
    parser.add_argument(
        "--max-frame-skip",
        type=int,
        default=5,
        help="Maximum frames PyBoy may skip to maintain speed.",
    )
    parser.add_argument("--frames-per-step", type=int, default=12, help="Frames to wait on wait actions.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for exploration targets.")
    parser.add_argument(
        "--planner-backend",
        type=str,
        default="openai",
        choices=["none", "fake", "mock", "openai", "anthropic"],
        help="Planner backend to use for planlet generation.",
    )
    parser.add_argument("--planner-model", type=str, default="gpt-5", help="Override planner model identifier.")
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
    parser.add_argument(
        "--metadata-out",
        type=Path,
        help="Optional JSON file to record run configuration and environment metadata.",
    )
    parser.add_argument(
        "--planlet-watchdog-steps",
        type=int,
        default=900,
        help="Trigger replan if a planlet stays active for N controller steps (0 disables).",
    )
    parser.add_argument(
        "--stall-watchdog-steps",
        type=int,
        default=600,
        help="Reload save-state if overworld step counter stops advancing for N controller steps (0 disables).",
    )
    parser.add_argument(
        "--watchdog-save-slot",
        type=int,
        default=None,
        help="Optional PyBoy save-state slot used by the stall watchdog (requires PyBoy save/load support).",
    )
    return parser.parse_args()


def ingest_overworld_observation(
    executor: OverworldExecutor,
    observation: OverworldObservation,
) -> Mapping[str, object]:
    writes = executor.extractor.extract(observation)
    for op in writes:
        executor.memory.write(op)
    executor._last_observation = observation  # keep executor memo in sync
    snapshot = executor.extractor.last_payload or {}
    return dict(snapshot) if isinstance(snapshot, Mapping) else {}


def extract_step_counter_from_observation(observation: OverworldObservation) -> Optional[int]:
    metadata = observation.metadata or {}
    for key in ("step_counter", "overworld_step_counter", "progress_step", "stepCount"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    ram_snapshot = observation.ram
    if isinstance(ram_snapshot, Mapping):
        addr = DEFAULT_OVERWORLD_RAM_MAP.get("step_counter")
        if addr is not None:
            raw_value = ram_snapshot.get(addr)
            if raw_value is not None:
                try:
                    return int(raw_value)
                except (TypeError, ValueError):
                    pass
    return None


DEFAULT_MISSION_PLAN: Dict[str, Any] = {
    "mission": "Defeat the Elite Four and collect all eight badges.",
    "progress": {
        "badges_collected": 0,
        "current_objective": "Leave the bedroom and reach Professor Oak's lab."
    },
    "planlets": {
        "pending": [],
        "completed": []
    },
    "notes": []
}


class PlanCoordinator:
    """Bridge overworld executor HALTs to planner/LLM requests."""

    def __init__(
        self,
        *,
        executor: "OverworldExecutor",
        service: Optional[PlanletService],
        allow_search: bool,
        nearby_limit: int,
        backend_label: str,
        logger: logging.Logger,
        mission_plan: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.executor = executor
        self.service = service
        self.allow_search = bool(allow_search)
        self.nearby_limit = max(1, int(nearby_limit))
        self._backend_label = backend_label
        self._logger = logger
        self._plan_seq = 0
        self._planlet_seq = 0
        self._last_source = backend_label if service is not None else "planner-unavailable"
        self._mission_plan: Dict[str, Any] = copy.deepcopy(mission_plan or DEFAULT_MISSION_PLAN)
        self._search_notice_logged = False

    def has_planner(self) -> bool:
        return self.service is not None

    def describe_last_source(self) -> str:
        return self._last_source

    def request_bundle(
        self,
        context: Optional[Mapping[str, object]],
        *,
        reason: str,
        observation: Optional[OverworldObservation] = None,
    ) -> PlanBundle:
        if self.service is None:
            raise RuntimeError("Planner backend not configured; cannot produce planlets.")
        if self.allow_search and not self._search_notice_logged:
            self._logger.info(
                "Web search tool enabled; planner may issue external queries (incurring additional API cost)."
            )
            self._search_notice_logged = True
        frame_bytes = self._encode_observation(observation)
        mission_plan_payload = self._mission_plan_for_prompt()
        try:
            proposal = self.service.request_overworld_planlet(
                self.executor.memory,
                nearby_limit=self.nearby_limit,
                allow_search=self.allow_search,
                frame_image=frame_bytes,
                mission_plan=mission_plan_payload,
            )
        except Exception as exc:
            self._logger.exception("Planlet service failed (reason=%s)", reason)
            raise RuntimeError(f"Planner backend error: {exc}") from exc
        bundle = self._bundle_from_proposal(proposal, reason)
        kinds = [planlet.kind for planlet in bundle.planlets]
        unsupported = [kind for kind in kinds if kind not in PlanCompiler.DEFAULT_REGISTRY]
        if unsupported:
            message = f"Planner emitted unsupported kinds {unsupported}."
            self._logger.warning(message)
            raise RuntimeError(message)
        return bundle

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _bundle_from_proposal(self, proposal: Any, reason: str) -> PlanBundle:
        payload = dict(getattr(proposal, "planlet", {}) or {})
        updates = payload.pop("updates", None)
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
        if updates:
            raw_bundle["updates"] = updates
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

    def _next_plan_id(self) -> str:
        self._plan_seq += 1
        return f"ow_plan_{int(time.time() * 1000)}_{self._plan_seq}"

    def _next_planlet_id(self) -> str:
        self._planlet_seq += 1
        return f"ow_planlet_{int(time.time() * 1000)}_{self._planlet_seq}"

    @staticmethod
    def _encode_observation(observation: Optional[OverworldObservation]) -> Optional[bytes]:
        if observation is None:
            return None
        try:
            from io import BytesIO
            from PIL import Image
        except ImportError:
            return None

        frame = getattr(observation, "raw_framebuffer", None)
        if frame is None:
            frame = getattr(observation, "framebuffer", None)
        if frame is None:
            return None
        try:
            image = Image.fromarray(frame.astype("uint8", copy=False))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception:
            return None

    def _mission_plan_for_prompt(self) -> Dict[str, Any]:
        plan = copy.deepcopy(self._mission_plan)
        planlets_section = plan.setdefault("planlets", {})
        state = getattr(self.executor, "state", None)
        active: List[Dict[str, Any]] = []
        if state is not None:
            if state.current_planlet is not None:
                spec = state.current_planlet.spec
                active.append(
                    {
                        "id": spec.id,
                        "kind": spec.kind,
                        "timeout_steps": spec.timeout_steps,
                    }
                )
            for queued in list(state.plan_queue or []):
                spec = queued.spec
                active.append(
                    {
                        "id": spec.id,
                        "kind": spec.kind,
                        "timeout_steps": spec.timeout_steps,
                    }
                )
        planlets_section["active"] = active
        planlets_section.setdefault("pending", [])
        planlets_section.setdefault("completed", [])
        return plan

    def register_plan_bundle(self, plan: "CompiledPlan", reason: str) -> None:
        pending = self._mission_plan.setdefault("planlets", {}).setdefault("pending", [])
        timestamp = datetime.now(timezone.utc).isoformat()
        for compiled_planlet in plan.planlets:
            entry = {
                "id": compiled_planlet.spec.id,
                "kind": compiled_planlet.spec.kind,
                "goal": plan.goal,
                "loaded_at": timestamp,
                "reason": reason,
            }
            pending.append(entry)

    def apply_updates(self, updates: Optional[Mapping[str, Any]]) -> None:
        if not updates:
            return
        _deep_merge(self._mission_plan, updates)

    def mark_planlet_completed(self, planlet_id: Optional[str]) -> None:
        if not planlet_id:
            return
        planlets_section = self._mission_plan.setdefault("planlets", {})
        pending = planlets_section.setdefault("pending", [])
        completed = planlets_section.setdefault("completed", [])
        for index, entry in enumerate(list(pending)):
            if entry.get("id") == planlet_id:
                pending.pop(index)
                entry = dict(entry)
                entry["completed_at"] = datetime.now(timezone.utc).isoformat()
                completed.append(entry)
                break

    def handle_plan_event(self, event: Any) -> None:
        status = getattr(event, "status", "")
        if status == "PLANLET_COMPLETE":
            self.mark_planlet_completed(getattr(event, "planlet_id", None))


def _deep_merge(target: Dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            _deep_merge(target[key], value)  # type: ignore[index]
        elif isinstance(value, list) and isinstance(target.get(key), list):
            target[key] = list(value)
        else:
            target[key] = copy.deepcopy(value)


def _serialise_args(args: argparse.Namespace) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(val) for key, val in value.items()}
        return value

    return {key: convert(val) for key, val in vars(args).items()}


def _detect_git_metadata() -> Dict[str, Optional[str]]:
    metadata: Dict[str, Optional[str]] = {"commit": None, "branch": None}
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        metadata["commit"] = commit
    except Exception:
        pass
    try:
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        metadata["branch"] = branch
    except Exception:
        pass
    return metadata


def _collect_run_metadata(
    args: argparse.Namespace,
    planner_service: Optional[PlanletService],
    coordinator: PlanCoordinator,
) -> Dict[str, Any]:
    planner_backend = args.planner_backend or ("none" if planner_service is None else "unknown")
    metadata: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli": _serialise_args(args),
        "run": {
            "rom": str(Path(args.rom).resolve()),
            "seed": args.seed,
            "frames_per_step": args.frames_per_step,
            "steps": args.steps,
            "window": args.window,
        },
        "planner": {
            "backend": planner_backend,
            "model": args.planner_model,
            "allow_search": args.planner_allow_search,
            "cache_enabled": not args.disable_planner_cache and args.planner_cache_size > 0,
            "nearby_limit": args.planner_nearby_limit,
            "last_plan_source": coordinator.describe_last_source(),
        },
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "git": _detect_git_metadata(),
    }
    if planner_service is not None:
        metadata["planner"]["client"] = planner_service.client.__class__.__name__
    return metadata


def _write_run_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


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
        return None, "none"

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

        if backend in {"openai", "anthropic"} and not api_key:
            raise RuntimeError(
                f"{backend.title()} API key not provided. Set the appropriate environment variable or pass --planner-api-key."
            )

        base_url = args.planner_base_url
        if not base_url and backend == "openai":
            base_url = os.getenv("OPENAI_BASE_URL")

        config = LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        client = create_llm_client(config)
        backend_label = backend

    service = PlanletService(proposer=proposer, client=client, store=store, cache=cache)
    return service, backend_label


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = PyBoyConfig(
        rom_path=str(args.rom),
        window_type=args.window,
        speed=float(args.emulation_speed),
        max_frame_skip=int(args.max_frame_skip),
    )
    adapter = PyBoyPokemonAdapter(cfg)
    overworld = OverworldAdapter(adapter)

    executor = OverworldExecutor()
    mission_plan = copy.deepcopy(DEFAULT_MISSION_PLAN)
    planner_service, backend_label = build_planlet_service(args)
    if planner_service is None:
        LOGGER.error("Planner backend is required; configure --planner-backend with a supported value.")
        return 1
    compiler = PlanCompiler()
    plan_coordinator = PlanCoordinator(
        executor=executor,
        service=planner_service,
        allow_search=args.planner_allow_search,
        nearby_limit=args.planner_nearby_limit,
        backend_label=backend_label,
        logger=LOGGER,
        mission_plan=mission_plan,
    )
    if planner_service is None:
        LOGGER.info("Planner backend disabled; relying on deterministic menu plans only.")
    else:
        LOGGER.info("Planner backend initialised (%s).", backend_label)

    if args.metadata_out:
        metadata = _collect_run_metadata(args, planner_service, plan_coordinator)
        try:
            _write_run_metadata(Path(args.metadata_out), metadata)
            LOGGER.info("Wrote run metadata to %s", Path(args.metadata_out))
        except Exception:
            LOGGER.exception("Failed to persist run metadata to %s", args.metadata_out)

    if args.watchdog_save_slot is not None:
        try:
            adapter.save_state(args.watchdog_save_slot)
            LOGGER.info("Saved initial PyBoy state to slot %s", args.watchdog_save_slot)
        except Exception:
            LOGGER.exception("Failed to save initial PyBoy state to slot %s.", args.watchdog_save_slot)

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

    last_observation_obj: OverworldObservation = observation
    last_snapshot = ingest_overworld_observation(executor, last_observation_obj)
    planlet_watchdog_threshold = max(0, int(args.planlet_watchdog_steps or 0))
    stall_watchdog_threshold = max(0, int(args.stall_watchdog_steps or 0))
    planlet_watchdog_steps = 0
    stall_watchdog_steps = 0
    last_planlet_id: Optional[str] = None
    last_step_counter = extract_step_counter_from_observation(observation)
    planner_failure_count = 0
    planner_failure_limit = 5
    planner_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner")
    pending_plan: Optional[Dict[str, Any]] = None

    def load_plan_bundle(
        bundle: PlanBundle,
        planner_snapshot: Mapping[str, object],
        *,
        reason: str,
    ) -> bool:
        plan_metadata = dict(getattr(bundle, "raw", {}).get("metadata") or {})
        plan_metadata.setdefault("planner_reason", reason)
        plan_metadata.setdefault("plan_source", plan_coordinator.describe_last_source())
        try:
            compiled = compiler.compile(bundle)
        except PlanCompilationError:
            LOGGER.exception("Failed to compile plan bundle (reason=%s).", reason)
            return False
        executor.load_compiled_plan(compiled, metadata=plan_metadata)
        LOGGER.info(
            "Loaded plan %s (%d planlets) source=%s reason=%s",
            compiled.plan_id,
            len(compiled.planlets),
            plan_coordinator.describe_last_source(),
            reason,
        )
        plan_coordinator.register_plan_bundle(compiled, reason)
        updates = getattr(bundle, "raw", {}).get("updates") if hasattr(bundle, "raw") else None
        plan_coordinator.apply_updates(updates)
        return True

    def load_new_plan(
        reason: str,
        planner_snapshot: Mapping[str, object],
        *,
        observation_obj: OverworldObservation,
    ) -> None:
        nonlocal pending_plan
        if pending_plan is not None:
            return
        snapshot_copy = copy.deepcopy(planner_snapshot)
        observation_ref = observation_obj
        started_at = time.perf_counter()

        def _invoke() -> PlanBundle:
            return plan_coordinator.request_bundle(
                snapshot_copy,
                reason=reason,
                observation=observation_ref,
            )

        future = planner_pool.submit(_invoke)
        pending_plan = {
            "future": future,
            "reason": reason,
            "snapshot": snapshot_copy,
            "started_at": started_at,
        }
        LOGGER.debug("Scheduled planner request (reason=%s)", reason)

    def finalize_pending_plan(*, block: bool = False) -> Optional[bool]:
        nonlocal pending_plan, planner_failure_count, planlet_watchdog_steps, last_planlet_id
        if pending_plan is None:
            return None
        future: Future = pending_plan["future"]
        if not future.done():
            if not block:
                return None
        try:
            bundle = future.result()
        except Exception as exc:
            reason = pending_plan.get("reason", "unknown")
            LOGGER.error("No plan available after planner failure (reason=%s): %s", reason, exc)
            planner_failure_count += 1
            pending_plan = None
            return False
        reason = pending_plan.get("reason", "unknown")
        snapshot_payload = pending_plan.get("snapshot", last_snapshot)
        started_at = pending_plan.get("started_at")
        pending_plan = None
        if isinstance(started_at, (int, float)):
            elapsed = time.perf_counter() - started_at
            LOGGER.info("Planner completed (reason=%s) in %.2fs", reason, elapsed)
        if not load_plan_bundle(bundle, snapshot_payload, reason=reason):
            planner_failure_count += 1
            return False
        planner_failure_count = 0
        planlet_watchdog_steps = 0
        last_planlet_id = None
        return True

    def ensure_plan(
        planner_snapshot: Mapping[str, object],
        *,
        reason: str,
        observation_obj: OverworldObservation,
    ) -> None:
        if executor.state.current_planlet is not None or executor.state.plan_queue:
            return
        load_new_plan(reason, planner_snapshot, observation_obj=observation_obj)

    def replan_handler(event) -> Optional[PlanBundle]:
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
        observation_ref = executor._last_observation or last_observation_obj
        load_new_plan(tag, last_snapshot, observation_obj=observation_ref)
        return None

    executor.register_replan_handler(replan_handler)
    def _plan_event_logger(event) -> None:
        plan_coordinator.handle_plan_event(event)
        if recorder is None:
            return
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
    try:
        ensure_plan(last_snapshot, reason="initial", observation_obj=last_observation_obj)
    except Exception:
        LOGGER.exception("Planner failed during initial plan request.")
        return 1

    step = 0
    try:
        while running and step < args.steps:
            finalize_outcome = finalize_pending_plan()
            if finalize_outcome is False and planner_failure_count >= planner_failure_limit:
                LOGGER.error("Planner failure threshold reached; aborting run.")
                break
            try:
                result = executor.step(observation)
            except TraceValidationError as exc:
                LOGGER.error("Trace recorder validation failed: %s", exc)
                break
            except Exception:
                LOGGER.exception("Executor step failed; forcing replan.")
                last_snapshot = ingest_overworld_observation(executor, observation)
                last_observation_obj = observation
                load_new_plan("executor_error", last_snapshot, observation_obj=observation)
                continue

            snapshot_view = executor.extractor.last_payload
            if isinstance(snapshot_view, Mapping):
                last_snapshot = dict(snapshot_view)

            pokemon_action = action_to_pokemon(result.action, frames_per_step=args.frames_per_step)
            try:
                observation = overworld.step(pokemon_action)
            except Exception:
                LOGGER.exception("PyBoy adapter step failed; forcing replan.")
                time.sleep(0.1)
                last_snapshot = ingest_overworld_observation(executor, observation)
                last_observation_obj = observation
                last_step_counter = extract_step_counter_from_observation(observation)
                load_new_plan("adapter_error", last_snapshot, observation_obj=observation)
                continue

            current_counter = extract_step_counter_from_observation(observation)
            last_observation_obj = observation

            active_planlet = getattr(executor.state.current_planlet, "spec", None)
            if active_planlet is not None:
                if active_planlet.id == last_planlet_id:
                    planlet_watchdog_steps += 1
                else:
                    last_planlet_id = active_planlet.id
                    planlet_watchdog_steps = 0
            else:
                last_planlet_id = None
                planlet_watchdog_steps = 0

            if planlet_watchdog_threshold > 0 and planlet_watchdog_steps >= planlet_watchdog_threshold:
                LOGGER.warning(
                    "Planlet watchdog triggered for planlet %s after %s steps; requesting new plan.",
                    last_planlet_id,
                    planlet_watchdog_steps,
                )
                last_snapshot = ingest_overworld_observation(executor, observation)
                last_observation_obj = observation
                load_new_plan("watchdog_planlet", last_snapshot, observation_obj=observation)
                planlet_watchdog_steps = 0
                last_planlet_id = None
                stall_watchdog_steps = 0
                last_step_counter = current_counter
                continue

            if pending_plan is not None:
                stall_watchdog_steps = 0
            elif stall_watchdog_threshold > 0:
                if current_counter is None or last_step_counter is None:
                    stall_watchdog_steps = 0
                elif current_counter == last_step_counter:
                    stall_watchdog_steps += 1
                else:
                    stall_watchdog_steps = 0
                last_step_counter = current_counter
                if stall_watchdog_steps >= stall_watchdog_threshold:
                    LOGGER.warning(
                        "Stall watchdog triggered after %s steps without progress; attempting recovery.",
                        stall_watchdog_steps,
                    )
                    stall_watchdog_steps = 0
                    recovered = False
                    if args.watchdog_save_slot is not None:
                        try:
                            adapter.load_state(args.watchdog_save_slot)
                            observation = overworld.observe()
                            last_snapshot = ingest_overworld_observation(executor, observation)
                            last_observation_obj = observation
                            last_step_counter = extract_step_counter_from_observation(observation)
                            ensure_plan(last_snapshot, reason="watchdog_stall_reload", observation_obj=observation)
                            LOGGER.info(
                                "Reloaded PyBoy save-state slot %s after stall.",
                                args.watchdog_save_slot,
                            )
                            recovered = True
                        except Exception:
                            LOGGER.exception(
                                "Failed to reload save-state slot %s.",
                                args.watchdog_save_slot,
                            )
                    if not recovered:
                        last_snapshot = ingest_overworld_observation(executor, observation)
                        last_observation_obj = observation
                        load_new_plan("watchdog_stall_replan", last_snapshot, observation_obj=observation)
                        last_step_counter = extract_step_counter_from_observation(observation)
                    continue

            if result.status in {"PLANLET_COMPLETE", "PLANLET_STALLED", "PLAN_COMPLETE"}:
                last_snapshot = ingest_overworld_observation(executor, observation)
                last_observation_obj = observation
                ensure_plan(last_snapshot, reason=f"executor_status:{result.status}", observation_obj=observation)

            last_step_counter = current_counter

            step += 1
    finally:
        try:
            finalize_pending_plan(block=False)
        except Exception:
            LOGGER.exception("Failed to finalise pending planner request during shutdown.")
        if pending_plan is not None:
            future_obj = pending_plan.get("future")
            if isinstance(future_obj, Future):
                future_obj.cancel()
            pending_plan = None
        planner_pool.shutdown(wait=False)
        if recorder is not None:
            recorder.close()
        adapter.close()

    LOGGER.info("Completed %s steps", step)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual testing script
    sys.exit(main())
