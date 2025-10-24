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
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional
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
    parser.add_argument("--planner-model", type=str, default="gpt-5-mini", help="Override planner model identifier.")
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
    if isinstance(snapshot, Mapping):
        try:
            overworld = snapshot.get("overworld", {})
            if isinstance(overworld, Mapping):
                tile_count = len(overworld.get("tiles", [])) if isinstance(overworld.get("tiles"), list) else 0
                adjacency = overworld.get("tile_adjacency")
                adjacency_edges = 0
                if isinstance(adjacency, Mapping):
                    adjacency_edges = sum(len(neigh) for neigh in adjacency.values() if isinstance(neigh, list))
                map_id = None
                player_tile = None
                map_info = overworld.get("map")
                if isinstance(map_info, Mapping):
                    map_id = map_info.get("id")
                player_info = overworld.get("player")
                if isinstance(player_info, Mapping):
                    player_tile = player_info.get("tile")
                LOGGER.debug(
                    "Overworld graph snapshot: tiles=%s adjacency_edges=%s map=%s player_tile=%s",
                    tile_count,
                    adjacency_edges,
                    map_id,
                    player_tile,
                )
        except Exception:
            LOGGER.exception("Failed to summarise overworld graph snapshot.")
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
        self._recent_dialog: deque[Dict[str, Any]] = deque(maxlen=6)
        self._dialog_facts: deque[Dict[str, Any]] = deque(maxlen=12)
        self._dialog_fact_hashes: set[str] = set()

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
        if mission_plan_payload:
            snapshot = mission_plan_payload.get("environment", {}).get("overworld_snapshot")
            if isinstance(snapshot, Mapping):
                naming_payload = snapshot.get("naming_screen")
                overlay_payload = snapshot.get("overlay_state")
                self._logger.info(
                    "Mission plan snapshot: naming=%s overlay=%s cursor=%s presets=%s",
                    bool(naming_payload),
                    bool(overlay_payload),
                    naming_payload.get("cursor") if isinstance(naming_payload, Mapping) else None,
                    naming_payload.get("presets") if isinstance(naming_payload, Mapping) else None,
                )
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
        snapshot = self._overworld_snapshot_for_prompt()
        extractor_payload = getattr(self.executor, "extractor", None)
        if extractor_payload is not None:
            last_payload = getattr(extractor_payload, "last_payload", None)
            if isinstance(last_payload, Mapping):
                ow = last_payload.get("overworld")
                naming_present = isinstance(ow.get("naming_screen"), Mapping) if isinstance(ow, Mapping) else False
                map_id = ow.get("map", {}).get("id") if isinstance(ow, Mapping) else None
                menus = ow.get("menus") if isinstance(ow, Mapping) else None
                self._logger.info(
                    "Last payload map=%s naming=%s menus=%s",
                    map_id,
                    naming_present,
                    menus[:3] if isinstance(menus, list) else menus,
                )
        if snapshot:
            plan.setdefault("environment", {})["overworld_snapshot"] = snapshot
        return plan

    def _overworld_snapshot_for_prompt(self) -> Optional[Dict[str, Any]]:
        extractor = getattr(self.executor, "extractor", None)
        payload = getattr(extractor, "last_payload", None)
        if not isinstance(payload, Mapping):
            return None
        overworld = payload.get("overworld")
        if not isinstance(overworld, Mapping):
            return None

        player = overworld.get("player") or {}
        try:
            px = int(player.get("tile", [0, 0])[0])
            py = int(player.get("tile", [0, 0])[1])
        except Exception:
            px = 0
            py = 0

        tiles = overworld.get("tiles")
        interesting: List[Dict[str, Any]] = []
        if isinstance(tiles, Iterable):
            for tile in tiles:
                if not isinstance(tile, Mapping):
                    continue
                passable = bool(tile.get("passable", True))
                terrain = str(tile.get("terrain", "unknown"))
                special = tile.get("special")
                if not passable and terrain not in {"door", "stairs"} and not special:
                    continue
                if terrain not in {"door", "stairs"} and not special:
                    continue
                try:
                    tx = int(tile.get("x", 0))
                    ty = int(tile.get("y", 0))
                except Exception:
                    continue
                dist = abs(px - tx) + abs(py - ty)
                interesting.append(
                    {
                        "tile": [tx, ty],
                        "terrain": terrain,
                        "special": special,
                        "passable": passable,
                        "screen": dict(tile.get("screen", {})) if isinstance(tile.get("screen"), Mapping) else None,
                        "distance": dist,
                    }
                )
        interesting.sort(key=lambda item: item["distance"])
        interesting = interesting[:8]

        entities_payload: List[Dict[str, Any]] = []
        entities = overworld.get("entities")
        if isinstance(entities, Iterable):
            for entity in entities:
                if not isinstance(entity, Mapping):
                    continue
                entry = {
                    "id": str(entity.get("id")),
                    "tile": list(entity.get("tile", [])) if isinstance(entity.get("tile"), Iterable) else None,
                    "tile_id": entity.get("tile_id"),
                    "screen": dict(entity.get("screen", {})) if isinstance(entity.get("screen"), Mapping) else None,
                }
                entities_payload.append(entry)
        if len(entities_payload) > 8:
            entities_payload = entities_payload[:8]

        snapshot: Dict[str, Any] = {
            "player": {
                "tile": [px, py],
                "facing": player.get("facing"),
                "map_id": player.get("map_id"),
            },
            "targets": interesting,
            "entities": entities_payload,
        }

        dialog_lines = overworld.get("dialog_lines")
        if isinstance(dialog_lines, list):
            for line in dialog_lines:
                if isinstance(line, str):
                    text = line.strip()
                    if text:
                        self._recent_dialog.append({"text": text, "map_id": snapshot["player"]["map_id"]})
            self._handle_dialog_lines(dialog_lines, map_id=snapshot["player"]["map_id"])
        if self._recent_dialog:
            snapshot["recent_dialog"] = list(self._recent_dialog)

        naming = overworld.get("naming_screen")
        if isinstance(naming, Mapping):
            naming_snapshot: Dict[str, Any] = {
                "grid_letters": naming.get("grid_letters"),
                "cursor": naming.get("cursor"),
            }
            if isinstance(naming.get("cursor_history"), list):
                naming_snapshot["cursor_history"] = naming.get("cursor_history")[:8]
            if isinstance(naming.get("presets"), list):
                naming_snapshot["presets"] = naming.get("presets")
            if isinstance(naming.get("dialog_lines"), list):
                naming_snapshot["dialog_lines"] = naming.get("dialog_lines")
            snapshot["naming_screen"] = naming_snapshot
            self._logger.info(
                "Snapshot naming cursor=%s presets=%s",
                naming_snapshot.get("cursor"),
                naming_snapshot.get("presets"),
            )

        menus = overworld.get("menus")
        if isinstance(menus, list):
            snapshot["menus"] = [
                {"id": menu.get("id"), "path": menu.get("path"), "open": menu.get("open")}
                for menu in menus[:5]
                if isinstance(menu, Mapping)
            ]

        overlay_state: Dict[str, Any] = {}
        if isinstance(overworld.get("dialog"), Mapping):
            overlay_state["dialog_open"] = True
        if isinstance(menus, list):
            overlay_state["menu_overlay"] = any(
                bool(menu.get("open"))
                and any("OVERLAY" in str(part).upper() for part in (menu.get("path") or []))
                for menu in menus
                if isinstance(menu, Mapping)
            )
        if isinstance(naming, Mapping):
            overlay_state["naming_active"] = True
        if overlay_state:
            snapshot["overlay_state"] = overlay_state

        adjacency = overworld.get("tile_adjacency")
        if isinstance(adjacency, Mapping) and adjacency:
            degrees: List[int] = []
            sample: List[Dict[str, Any]] = []
            for key, neighbors in adjacency.items():
                if not isinstance(neighbors, Iterable):
                    continue
                neighbor_list = list(neighbors)
                deg = len(neighbor_list)
                degrees.append(deg)
                if len(sample) < 4 and isinstance(key, tuple) and len(key) == 3:
                    sample.append(
                        {
                            "tile": {"map_id": key[0], "x": key[1], "y": key[2]},
                            "neighbors": deg,
                        }
                    )
            if degrees:
                avg_degree = sum(degrees) / len(degrees)
                adjacency_stats = {
                    "tracked_tiles": len(degrees),
                    "avg_degree": round(avg_degree, 2),
                    "max_degree": max(degrees),
                    "min_degree": min(degrees),
                    "isolated_tiles": sum(1 for value in degrees if value == 0),
                }
                if sample:
                    adjacency_stats["sample"] = sample
                snapshot["tile_adjacency_stats"] = adjacency_stats

        if self._dialog_facts:
            snapshot["dialog_facts"] = list(self._dialog_facts)

        return snapshot

    def _handle_dialog_lines(self, lines: Iterable[str], *, map_id: Any) -> None:
        cleaned = tuple(str(line).strip() for line in lines if isinstance(line, str))
        if not cleaned:
            return
        hash_key = json.dumps({"map": map_id, "lines": cleaned}, ensure_ascii=False)
        if hash_key in self._dialog_fact_hashes:
            return
        self._dialog_fact_hashes.add(hash_key)
        facts = self._summarise_dialog_lines(cleaned, map_id=map_id)
        for fact in facts:
            entry = {
                "map_id": map_id,
                "fact": fact,
                "source_lines": list(cleaned),
            }
            self._dialog_facts.append(entry)

    def _summarise_dialog_lines(self, lines: Iterable[str], *, map_id: Any) -> List[Dict[str, Any]]:
        if self.service is None:
            return []
        client = getattr(self.service, "client", None)
        if client is None:
            return []

        system_prompt = (
            "You convert Pokémon NPC dialog into structured facts for an agent. "
            "Always respond with a JSON array (possibly empty) of objects, with fields such as type, action, item, "
            "location, summary, or actor. Use concise lowercase identifiers (e.g., type:\"clue\", action:\"learn_move\"). "
            "If no actionable information is present, return an empty array []. Never include comments or prose outside the JSON array."
        )
        user_payload = {
            "map_id": map_id,
            "dialog_lines": list(lines),
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

        try:
            raw = client.generate_response(messages)
        except Exception as exc:
            self._logger.exception("Dialog summariser call failed: %s", exc)
            return []

        if isinstance(raw, str):
            content = raw.strip()
        elif isinstance(raw, Mapping):
            content = json.dumps(raw)
        else:
            content = str(raw)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            self._logger.warning("Dialog summariser returned non-JSON payload: %s", content)
            return []

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        facts: List[Dict[str, Any]] = []
        for entry in data:
            if isinstance(entry, Mapping):
                facts.append({key: value for key, value in entry.items() if isinstance(key, str)})
        return facts

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
                model = "gpt-5-mini"
            elif backend == "anthropic":
                model = "claude-3-sonnet-20240229"
            else:
                model = "gpt-5-mini"

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

    # -------------------- Overworld readiness predicates -------------------- #
    def _overlay_open(snapshot: Mapping[str, object]) -> bool:
        ow = snapshot.get("overworld") if isinstance(snapshot, Mapping) else None
        if not isinstance(ow, Mapping):
            return False
        menus = ow.get("menus")
        if not isinstance(menus, Iterable):
            return False
        for menu in menus:
            if not isinstance(menu, Mapping):
                continue
            if not menu.get("open"):
                continue
            path = menu.get("path") or []
            if any(str(part).upper() in {"DIALOG", "OVERLAY"} for part in path):
                return True
        return False

    def _anchored_world(snapshot: Mapping[str, object]) -> bool:
        ow = snapshot.get("overworld") if isinstance(snapshot, Mapping) else None
        if not isinstance(ow, Mapping):
            return False
        map_info = ow.get("map")
        map_id = map_info.get("id") if isinstance(map_info, Mapping) else None
        tiles = ow.get("tiles")
        tile_count = len(tiles) if isinstance(tiles, list) else 0
        if not map_id or map_id == "unknown":
            return False
        if map_id == "screen_local":
            if tile_count < 200:
                return False
        else:
            if tile_count < 120:
                return False
        adjacency = ow.get("tile_adjacency")
        if not isinstance(adjacency, Mapping):
            return False
        adjacency_edges = 0
        for neighbors in adjacency.values():
            if isinstance(neighbors, Iterable):
                adjacency_edges += sum(1 for _ in neighbors)
        min_edges = 200 if map_id == "screen_local" else 50
        if adjacency_edges < min_edges:
            return False
        return True

    def _overworld_ready(snapshot: Mapping[str, object]) -> bool:
        return _anchored_world(snapshot) and not _overlay_open(snapshot)
    # ----------------------------------------------------------------------- #


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
        primary_kind = None
        try:
            planlets = getattr(bundle, "planlets", None)
            if planlets:
                primary_kind = getattr(planlets[0], "kind", None)
        except Exception:
            primary_kind = None
        if not _overworld_ready(snapshot_payload) and primary_kind != "MENU_SEQUENCE":
            LOGGER.info(
                "Discarding %s planlet while overlay/intro active; requesting MENU_SEQUENCE instead.",
                primary_kind or "unknown",
            )
            load_new_plan("overlay_force_menu", snapshot_payload, observation_obj=last_observation_obj)
            return None
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
        request_reason = reason if _overworld_ready(planner_snapshot) else "overlay_bootstrap"
        load_new_plan(request_reason, planner_snapshot, observation_obj=observation_obj)

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
        gated_tag = tag if _overworld_ready(last_snapshot) else "overlay_replan"
        load_new_plan(gated_tag, last_snapshot, observation_obj=observation_ref)
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

    # ------------------------------------------------------------------ #
    # While waiting on an asynchronous planner response, pump PyBoy with
    # single-frame waits so SDL stays responsive without progressing state.
    # ------------------------------------------------------------------ #
    def _idle_pump_until_plan(*, max_seconds: float = 12.0, frames_per_pump: int = 1) -> None:
        nonlocal observation, last_snapshot, last_observation_obj
        nonlocal stall_watchdog_steps, pending_plan
        if pending_plan is None:
            return
        start = time.perf_counter()
        frames = max(1, int(frames_per_pump))
        while pending_plan is not None and (time.perf_counter() - start) < max_seconds:
            try:
                idle_action = PokemonAction("WAIT", {"frames": frames})
                observation = overworld.step(idle_action)
            except Exception:
                LOGGER.exception("Idle pump failed while waiting for planner; aborting wait loop.")
                break
            last_snapshot = ingest_overworld_observation(executor, observation)
            last_observation_obj = observation
            stall_watchdog_steps = 0
            outcome = finalize_pending_plan(block=False)
            if outcome is not None:
                break
    def _naming_overlay_open(snapshot: Mapping[str, object]) -> bool:
        ow = snapshot.get("overworld")
        return isinstance(ow, Mapping) and isinstance(ow.get("naming_screen"), Mapping)

    def _naming_signature(snapshot: Mapping[str, object]) -> Optional[str]:
        if not isinstance(snapshot, Mapping):
            return None
        overworld = snapshot.get("overworld")
        if not isinstance(overworld, Mapping):
            return None
        naming = overworld.get("naming_screen")
        if not isinstance(naming, Mapping):
            return None
        signature: Dict[str, Any] = {}
        cursor = naming.get("cursor")
        if isinstance(cursor, Mapping):
            signature["cursor"] = {
                "row": cursor.get("row"),
                "col": cursor.get("col"),
                "letter": cursor.get("letter"),
            }
        dialog_lines = naming.get("dialog_lines")
        if isinstance(dialog_lines, list):
            signature["dialog_lines"] = [str(line) for line in dialog_lines[:4]]
        history = naming.get("cursor_history")
        if isinstance(history, list) and history:
            tail = history[-1]
            if isinstance(tail, Mapping):
                signature["cursor_history_tail"] = {
                    "row": tail.get("row"),
                    "col": tail.get("col"),
                    "letter": tail.get("letter"),
                }
        overlay_dialog = overworld.get("dialog_lines")
        if isinstance(overlay_dialog, list):
            signature["overlay_dialog"] = [str(line) for line in overlay_dialog[:4]]
        if not signature:
            return None
        try:
            return json.dumps(signature, sort_keys=True)
        except TypeError:
            def _to_serialisable(value: Any) -> Any:
                if isinstance(value, (list, tuple)):
                    return [_to_serialisable(item) for item in value]
                if isinstance(value, Mapping):
                    return {str(key): _to_serialisable(val) for key, val in value.items()}
                return value
            return json.dumps(_to_serialisable(signature), sort_keys=True)

    def _is_probably_a_spam(planlet_spec: object) -> bool:
        try:
            script = getattr(planlet_spec, "script", None)
            if not isinstance(script, Iterable):
                return False
            for entry in script:
                if not isinstance(entry, Mapping):
                    continue
                if str(entry.get("op")).upper() != "MENU_SEQUENCE":
                    continue
                buttons = entry.get("buttons")
                if not isinstance(buttons, Iterable):
                    continue
                unique_buttons = {str(btn).upper() for btn in buttons if btn is not None}
                if unique_buttons and unique_buttons <= {"A"}:
                    return True
        except Exception:
            return False
        return False

    NAMING_STALL_THRESHOLD = 5
    naming_signature_prev = _naming_signature(last_snapshot)
    naming_repeat_planlet_id: Optional[str] = None
    naming_repeat_count = 0

    prev_overworld_ready = _overworld_ready(last_snapshot)

    try:
        while running and step < args.steps:
            finalize_outcome = finalize_pending_plan(block=False)
            if finalize_outcome is False and planner_failure_count >= planner_failure_limit:
                LOGGER.error("Planner failure threshold reached; aborting run.")
                break
            current_overworld_ready = _overworld_ready(last_snapshot)
            if current_overworld_ready and not prev_overworld_ready:
                ow = last_snapshot.get("overworld") if isinstance(last_snapshot, Mapping) else {}
                tiles = ow.get("tiles")
                tile_count = len(tiles) if isinstance(tiles, list) else 0
                map_id = (ow.get("map") or {}).get("id") if isinstance(ow.get("map"), Mapping) else None
                LOGGER.info(
                    "Overworld now ready (map_id=%s tiles=%s pending_plan=%s)",
                    map_id,
                    tile_count,
                    bool(pending_plan),
                )
            prev_overworld_ready = current_overworld_ready
            if _overworld_ready(last_snapshot) and pending_plan is None:
                current_planlet_obj = getattr(executor.state, "current_planlet", None)
                current_spec = getattr(current_planlet_obj, "spec", None) if current_planlet_obj is not None else None
                current_kind = getattr(current_spec, "kind", None) if current_spec is not None else None
                if current_kind == "MENU_SEQUENCE":
                    LOGGER.info(
                        "Overlay cleared while MENU_SEQUENCE %s active; cancelling macro and requesting navigation plan.",
                        current_spec.id if current_spec is not None else None,
                    )
                    try:
                        executor.monitor.clear_planlet(current_spec.id)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    executor.state.current_planlet = None
                    executor.state.current_skill = None
                    load_new_plan("overlay_cleared", last_snapshot, observation_obj=last_observation_obj)
                    _idle_pump_until_plan(max_seconds=6.0, frames_per_pump=1)
                    continue
            if _naming_overlay_open(last_snapshot):
                current_planlet = getattr(executor.state, "current_planlet", None)
                planlet_spec = getattr(current_planlet, "spec", None)
                active_kind = getattr(planlet_spec, "kind", None) if planlet_spec is not None else None
                active_format = getattr(planlet_spec, "format", None) if planlet_spec is not None else None
                wrong_kind = active_kind != "MENU_SEQUENCE"
                wrong_format = not (isinstance(active_format, str) and "name" in active_format.lower())
                bad_macro = _is_probably_a_spam(planlet_spec)
                if pending_plan is not None or wrong_kind or wrong_format or bad_macro:
                    if pending_plan is None:
                        load_new_plan("overlay:naming", last_snapshot, observation_obj=last_observation_obj)
                    _idle_pump_until_plan(max_seconds=12.0, frames_per_pump=1)
                    stall_watchdog_steps = 0
                    continue
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
                last_snapshot = ingest_overworld_observation(executor, observation)
                last_observation_obj = observation
                last_step_counter = extract_step_counter_from_observation(observation)
                load_new_plan("adapter_error", last_snapshot, observation_obj=observation)
                _idle_pump_until_plan(max_seconds=3.0, frames_per_pump=args.frames_per_step)
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

            current_naming_signature = _naming_signature(last_snapshot)
            if _naming_overlay_open(last_snapshot) and active_planlet is not None and current_naming_signature is not None:
                if (
                    active_planlet.id == naming_repeat_planlet_id
                    and naming_signature_prev == current_naming_signature
                ):
                    naming_repeat_count += 1
                else:
                    naming_repeat_planlet_id = active_planlet.id
                    naming_repeat_count = 1
                if NAMING_STALL_THRESHOLD > 0 and naming_repeat_count >= NAMING_STALL_THRESHOLD:
                    LOGGER.warning(
                        "Naming stall detected for planlet %s after %s unchanged overlays; invalidating cached macro.",
                        active_planlet.id,
                        naming_repeat_count,
                    )
                    if plan_coordinator.has_planner():
                        service = plan_coordinator.service
                        if service is not None:
                            service.record_feedback(active_planlet.id, success=False)
                            service.invalidate_cached_planlet(planlet_id=active_planlet.id)
                    last_snapshot = ingest_overworld_observation(executor, observation)
                    last_observation_obj = observation
                    load_new_plan("naming_stall", last_snapshot, observation_obj=observation)
                    naming_repeat_planlet_id = None
                    naming_repeat_count = 0
                    naming_signature_prev = _naming_signature(last_snapshot)
                    planlet_watchdog_steps = 0
                    stall_watchdog_steps = 0
                    last_planlet_id = None
                    last_step_counter = current_counter
                    _idle_pump_until_plan(max_seconds=6.0, frames_per_pump=1)
                    continue
            else:
                naming_repeat_planlet_id = None
                naming_repeat_count = 0
            naming_signature_prev = current_naming_signature

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
