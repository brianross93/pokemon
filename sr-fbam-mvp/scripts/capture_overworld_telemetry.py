"""
Capture overworld telemetry either from a live PyBoy session or via a synthetic dry-run.

Examples:
    # Synthetic trace (no emulator required)
    python scripts/capture_overworld_telemetry.py --output data/overworld/sample_trace.jsonl --steps 32 --dry-run

    # Live capture (requires PyBoy + ROM path + configured memory addresses)
    python scripts/capture_overworld_telemetry.py --rom Pokemon\\ Blue.gb --output data/overworld/run01.jsonl --steps 200

    # Corridor sweep with deterministic seeds and gate/adherence export
    python scripts/capture_overworld_telemetry.py \
        --rom Pokemon\\ Blue.gb \
        --output data/overworld/corridor_run.jsonl \
        --gate-jsonl data/overworld/corridor_gates.jsonl \
        --metadata-out runs/metadata/corridor_run.json \
        --seed-set corridor_a corridor_b corridor_c \
        --traces-per-seed 2 \
        --steps 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.middleware.pokemon_adapter import PokemonAction, PokemonTelemetry
from src.overworld.env.overworld_adapter import OverworldAdapter, OverworldObservation
from src.overworld.recording.trace_recorder import OverworldTraceRecorder
from src.pkmn_overworld.world_graph import WorldGraph

try:  # Optional dependency
    from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter
except RuntimeError:  # pragma: no cover - raised when pyboy missing
    PyBoyPokemonAdapter = None  # type: ignore[assignment]
    PyBoyConfig = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture overworld telemetry for training.")
    parser.add_argument(
        "--rom",
        type=str,
        default="Pokemon Blue.gb",
        help="Path to the Pokemon Blue ROM (only used when not running with --dry-run).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/overworld/sample_trace.jsonl"),
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of decision steps to record.",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=20,
        help="Frames to advance between observations when capturing from PyBoy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic telemetry without launching PyBoy.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="null",
        help="PyBoy window type (use 'SDL2' for visible window).",
    )
    parser.add_argument(
        "--debug-addresses",
        action="store_true",
        help="Include raw memory reads for configured addresses in the telemetry output (PyBoy only).",
    )
    parser.add_argument(
        "--seed-set",
        nargs="+",
        default=["corridor_a", "corridor_b"],
        help="Deterministic seed identifiers applied per trace run (PyBoy only).",
    )
    parser.add_argument(
        "--traces-per-seed",
        type=int,
        default=1,
        help="Number of captures to record for each seed identifier.",
    )
    parser.add_argument(
        "--gate-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file to store per-step gate/adherence events.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Optional metadata JSON summarising the capture run (seeds, counts, files).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier injected into plan context and metadata.",
    )
    return parser.parse_args()


def make_plan_context(run_id: str) -> Mapping[str, object]:
    return {
        "id": f"plan-run-{run_id}",
        "planlet_id": f"pl-overworld-{run_id}",
        "planlet_kind": "OVERWORLD",
        "timeout": 900,
    }


TelemetryLike = Union[PokemonTelemetry, OverworldObservation, Mapping[str, Any]]


class GateEventLogger:
    """Helper to persist gate/adherence events to a side-car JSONL file."""

    def __init__(self, path: Optional[Path]) -> None:
        self._handle = None
        self.count = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8")

    def write(self, event: Mapping[str, object]) -> None:
        self.count += 1
        if self._handle is None:
            return
        json.dump(event, self._handle, ensure_ascii=False)
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()


def _seed_value(seed: str, trace_index: int) -> int:
    """
    Deterministically derive a 16-bit seed from the provided identifier.
    """
    digest = hashlib.blake2s(f"{seed}:{trace_index}".encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, byteorder="big") & 0xFFFF


def _extract_overworld_fields(
    telemetry: TelemetryLike,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int]]:
    """
    Normalise different telemetry payloads into overworld data, extra metadata, and an optional frame index.
    """
    if isinstance(telemetry, PokemonTelemetry):
        overworld = {
            "area_id": telemetry.area_id,
            "x": telemetry.x,
            "y": telemetry.y,
            "in_grass": telemetry.in_grass,
            "in_battle": telemetry.in_battle,
            "step_counter": telemetry.step_counter,
            "method": telemetry.method,
        }
        extras = dict(telemetry.extra or {})
        frame = extras.get("frame")
        return overworld, extras, frame

    if isinstance(telemetry, OverworldObservation):
        overworld = dict(telemetry.overworld or {})
        extras = dict(overworld.get("extra") or {})
        frame = getattr(telemetry, "frame", None)
        return overworld, extras, frame

    if isinstance(telemetry, Mapping):
        mapping = telemetry
        if "overworld" in mapping:
            overworld = dict(mapping.get("overworld") or {})
            extras = dict(overworld.get("extra") or mapping.get("extra") or {})
        else:
            overworld = {
                "area_id": mapping.get("area_id"),
                "x": mapping.get("x"),
                "y": mapping.get("y"),
                "in_grass": mapping.get("in_grass"),
                "in_battle": mapping.get("in_battle"),
                "step_counter": mapping.get("step_counter"),
                "method": mapping.get("method"),
            }
            extras = dict(mapping.get("extra") or {})
        frame = mapping.get("frame")
        return overworld, extras, frame

    raise TypeError(f"Unsupported telemetry payload: {type(telemetry)!r}")


def frame_features_from_telemetry(telemetry: TelemetryLike, step: int) -> List[float]:
    """Very small handcrafted feature vector until the real encoder lands."""

    overworld, extras, _ = _extract_overworld_fields(telemetry)
    area_id = int(overworld.get("area_id") or 0)
    x = int(overworld.get("x") or 0)
    y = int(overworld.get("y") or 0)
    in_grass = 1.0 if overworld.get("in_grass") else 0.0
    in_battle = 1.0 if overworld.get("in_battle") else 0.0
    step_counter = int(overworld.get("step_counter") or 0)
    time_bucket = str(extras.get("time_bucket") or "")

    return [
        float(area_id) / 255.0,
        float(x) / 255.0,
        float(y) / 255.0,
        in_grass,
        in_battle,
        float(step_counter % 256) / 255.0,
        float(step % 128) / 127.0,
        1.0 if time_bucket == "night" else 0.0,
    ]


def core_gate_payload(step: int) -> Mapping[str, object]:
    confidence = 0.75 + 0.2 * math.sin(step / 5.0)
    adherence_score = max(0.0, min(1.0, 0.65 + 0.3 * math.cos((step + 1) / 6.0)))
    adherence_code = "ADHERED" if adherence_score >= 0.6 else "OFF_PLAN"
    mode = "PLAN_STEP" if step % 3 else "PLAN_LOOKUP"
    return {
        "mode": mode,
        "encode_flag": False,
        "confidence": round(confidence, 3),
        "reason": "synthetic" if step % 7 else "cache_hit",
        "adherence": {"score": round(adherence_score, 3), "code": adherence_code},
        "adherence_flag": 1 if adherence_score >= 0.6 else 0,
    }


def legal_actions_stub() -> List[Mapping[str, object]]:
    return [
        {"id": "MOVE_UP", "index": 0},
        {"id": "MOVE_DOWN", "index": 1},
        {"id": "MOVE_LEFT", "index": 2},
        {"id": "MOVE_RIGHT", "index": 3},
        {"id": "WAIT", "index": 4},
    ]


def action_for_index(index: int, frames_per_step: int) -> PokemonAction:
    if index == 0:
        return PokemonAction("SCRIPT", {"inputs": ["UP"], "frames": 6})
    if index == 1:
        return PokemonAction("SCRIPT", {"inputs": ["DOWN"], "frames": 6})
    if index == 2:
        return PokemonAction("SCRIPT", {"inputs": ["LEFT"], "frames": 6})
    if index == 3:
        return PokemonAction("SCRIPT", {"inputs": ["RIGHT"], "frames": 6})
    return PokemonAction("WAIT", {"frames": frames_per_step})


def build_payload(
    *,
    telemetry: TelemetryLike,
    step: int,
    action_index: int,
    plan_context: Mapping[str, object],
    seed_id: str,
    trace_index: int,
    seed_value: int,
    run_id: str,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    overworld, extras, frame = _extract_overworld_fields(telemetry)
    frame_features = frame_features_from_telemetry(telemetry, step)
    gate_payload = core_gate_payload(step)
    frame_value = frame if frame is not None else int(extras.get("frame", step))
    debug_addrs = extras.get("debug_addrs")
    fractions = {"encode": 0.15 + 0.05 * math.sin(step / 4.0), "query": 0.7}
    fractions["encode"] = round(max(0.0, min(1.0, fractions["encode"])), 3)
    fractions["query"] = round(max(0.0, min(1.0, fractions["query"])), 3)
    fractions["skip"] = round(max(0.0, min(1.0, 1.0 - fractions["encode"] - fractions["query"])), 3)
    payload = {
        "source": "capture.overworld",
        "timestamp": timestamp,
        "context": {
            "domain": "overworld",
            "status": "PLANLET_ACTIVE",
            "mode": "explore",
            "step_index": step,
            "plan": plan_context,
            "seed": seed_id,
            "trace_index": trace_index,
            "run_id": run_id,
            "seed_value": seed_value,
        },
        "observation": {
            "frame": frame_value,
            "overworld": {
                "area_id": int(overworld.get("area_id") or 0),
                "x": int(overworld.get("x") or 0),
                "y": int(overworld.get("y") or 0),
            },
        },
        "telemetry": {
            "core": {
                "legal_actions": legal_actions_stub(),
                "action": {"id": "MOVE_RIGHT" if action_index == 3 else "WAIT", "index": action_index},
                "action_mask": [0.0 for _ in range(5)],
                "gate": gate_payload,
                "fractions": fractions,
                "speedup": {"predicted": 0.05, "observed": 0.02},
                "latency_ms": 12.0 + step * 0.1,
                "fallback_required": False,
                "hop_trace": [],
                "plan_metrics": {
                    "plan_step": step,
                    "steps_total": 256,
                    "confidence": gate_payload["confidence"],
                },
            },
            "overworld": {
                "mode": "explore",
                "frame_features": frame_features,
                "action_index": action_index,
                "status": "RUNNING",
            },
        },
    }
    if isinstance(debug_addrs, Mapping) and debug_addrs:
        payload["telemetry"]["overworld"]["debug_addresses"] = {
            str(name): int(value) for name, value in debug_addrs.items()
        }
    gate_event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "planlet_id": plan_context.get("planlet_id"),
        "run_id": run_id,
        "seed": seed_id,
        "seed_value": seed_value,
        "trace_index": trace_index,
        "step": step,
        "gate": gate_payload,
        "fractions": fractions,
        "adherence": gate_payload.get("adherence"),
        "domain": "overworld",
    }
    return payload, gate_event


def capture_with_pyboy(
    args: argparse.Namespace,
    gate_logger: Optional[GateEventLogger],
    stats: Dict[str, Any],
    run_id: str,
) -> Iterable[Mapping[str, object]]:
    if PyBoyPokemonAdapter is None or PyBoyConfig is None:
        raise RuntimeError("PyBoy is not available. Install pyboy or run with --dry-run.")

    cfg = PyBoyConfig(
        rom_path=args.rom,
        window_type=args.window,
        debug_addresses=args.debug_addresses,
    )
    adapter = PyBoyPokemonAdapter(cfg)
    overworld = OverworldAdapter(adapter)

    seeds: List[str] = list(dict.fromkeys(args.seed_set)) if args.seed_set else ["default"]

    try:
        for seed_index, seed_name in enumerate(seeds):
            for trace_index in range(max(1, args.traces_per_seed)):
                seed_value = _seed_value(seed_name, trace_index)
                if hasattr(adapter, "set_rng_seed"):
                    try:
                        adapter.set_rng_seed(seed_value)  # type: ignore[attr-defined]
                    except Exception as exc:  # pragma: no cover - defensive guard
                        print(f"[warn] Failed to apply RNG seed {seed_value} ({seed_name}): {exc}")
                telemetry = overworld.reset()
                trace_run_id = f"{run_id}:{seed_name}:{trace_index}"
                plan_context = make_plan_context(trace_run_id)

                payload, gate_event = build_payload(
                    telemetry=telemetry,
                    step=0,
                    action_index=4,
                    plan_context=plan_context,
                    seed_id=seed_name,
                    trace_index=trace_index,
                    seed_value=seed_value,
                    run_id=trace_run_id,
                )
                if gate_logger:
                    gate_logger.write(gate_event)
                stats["records"] += 1
                yield payload

                for step in range(1, args.steps):
                    action_index = step % 5
                    action = action_for_index(action_index, args.frames_per_step)
                    telemetry = overworld.step(action)
                    payload, gate_event = build_payload(
                        telemetry=telemetry,
                        step=step,
                        action_index=action_index,
                        plan_context=plan_context,
                        seed_id=seed_name,
                        trace_index=trace_index,
                        seed_value=seed_value,
                        run_id=trace_run_id,
                    )
                    if gate_logger:
                        gate_logger.write(gate_event)
                    stats["records"] += 1
                    yield payload

                stats["trace_details"].append(
                    {
                        "seed": seed_name,
                        "seed_value": seed_value,
                        "trace_index": trace_index,
                        "steps": args.steps,
                        "planlet_id": plan_context.get("planlet_id"),
                        "run_id": trace_run_id,
                    }
                )
    finally:
        adapter.close()


def synthetic_trace(
    args: argparse.Namespace,
    gate_logger: Optional[GateEventLogger],
    stats: Dict[str, Any],
    run_id: str,
) -> Iterable[Mapping[str, object]]:
    graph = WorldGraph(load_static=True)
    # Follow a simple loop in Pallet Town using static warp hints.
    coords: List[Tuple[int, int]] = [(x, 4) for x in range(4)] + [(3, y) for y in range(5, 8)] + [
        (x, 7) for x in range(2, -1, -1)
    ]
    area_id = 1  # Arbitrary map identifier for Pallet Town.
    seeds: List[str] = list(dict.fromkeys(args.seed_set)) if args.seed_set else ["synthetic"]

    for seed_name in seeds:
        for trace_index in range(max(1, args.traces_per_seed)):
            seed_value = _seed_value(seed_name, trace_index)
            trace_run_id = f"{run_id}:{seed_name}:{trace_index}"
            plan_context = make_plan_context(trace_run_id)
            for step in range(args.steps):
                x, y = coords[(step + trace_index) % len(coords)]
                telemetry = PokemonTelemetry(
                    area_id=area_id,
                    x=x,
                    y=y,
                    in_grass=False,
                    in_battle=False,
                    step_counter=step,
                    elapsed_ms=step * 16.0,
                    method="walk",
                    extra={"frame": step, "time_bucket": "day"},
                )
                action_index = (step + 1) % 5
                payload, gate_event = build_payload(
                    telemetry=telemetry,
                    step=step,
                    action_index=action_index,
                    plan_context=plan_context,
                    seed_id=seed_name,
                    trace_index=trace_index,
                    seed_value=seed_value,
                    run_id=trace_run_id,
                )
                if gate_logger:
                    gate_logger.write(gate_event)
                stats["records"] += 1
                yield payload
            stats["trace_details"].append(
                {
                    "seed": seed_name,
                    "seed_value": seed_value,
                    "trace_index": trace_index,
                    "steps": args.steps,
                    "planlet_id": plan_context.get("planlet_id"),
                    "run_id": trace_run_id,
                }
            )
    # Silence unused variable warning.
    _ = graph


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gate_logger = GateEventLogger(args.gate_jsonl)
    stats: Dict[str, Any] = {
        "run_id": run_id,
        "records": 0,
        "trace_details": [],
        "seed_set": list(dict.fromkeys(args.seed_set)) if args.seed_set else [],
        "traces_per_seed": max(1, args.traces_per_seed),
        "steps_per_trace": args.steps,
        "frames_per_step": args.frames_per_step,
        "dry_run": bool(args.dry_run),
    }
    capture_fn = capture_with_pyboy if not args.dry_run else synthetic_trace
    entries = capture_fn(args, gate_logger, stats, run_id)
    with OverworldTraceRecorder(args.output, validate=False) as recorder:
        for payload in entries:
            recorder.record(payload)
    gate_logger.close()

    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output": str(args.output),
            "gate_jsonl": str(args.gate_jsonl) if args.gate_jsonl else None,
            "rom": None if args.dry_run else args.rom,
            "dry_run": bool(args.dry_run),
            "seed_set": stats["seed_set"],
            "traces_per_seed": stats["traces_per_seed"],
            "steps_per_trace": stats["steps_per_trace"],
            "frames_per_step": stats["frames_per_step"],
            "records": stats["records"],
            "gate_events": gate_logger.count,
            "trace_details": stats["trace_details"],
        }
        args.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
