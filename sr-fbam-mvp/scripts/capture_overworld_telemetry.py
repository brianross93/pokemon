"""
Capture overworld telemetry either from a live PyBoy session or via a synthetic dry-run.

Examples:
    # Synthetic trace (no emulator required)
    python scripts/capture_overworld_telemetry.py --output data/overworld/sample_trace.jsonl --steps 32 --dry-run

    # Live capture (requires PyBoy + ROM path + configured memory addresses)
    python scripts/capture_overworld_telemetry.py --rom Pokemon\\ Blue.gb --output data/overworld/run01.jsonl --steps 200
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.middleware.pokemon_adapter import PokemonAction, PokemonTelemetry
from src.overworld.env.overworld_adapter import OverworldAdapter
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
    return parser.parse_args()


def make_plan_context(run_id: str) -> Mapping[str, object]:
    return {
        "id": f"plan-run-{run_id}",
        "planlet_id": f"pl-overworld-{run_id}",
        "planlet_kind": "OVERWORLD",
        "timeout": 900,
    }


def frame_features_from_telemetry(telemetry: PokemonTelemetry, step: int) -> List[float]:
    """Very small handcrafted feature vector until the real encoder lands."""

    return [
        float(telemetry.area_id) / 255.0,
        float(telemetry.x) / 255.0,
        float(telemetry.y) / 255.0,
        1.0 if telemetry.in_grass else 0.0,
        1.0 if telemetry.in_battle else 0.0,
        float(telemetry.step_counter % 256) / 255.0,
        float(step % 128) / 127.0,
        1.0 if telemetry.extra.get("time_bucket") == "night" else 0.0,
    ]


def core_gate_payload(step: int) -> Mapping[str, object]:
    confidence = 0.75 + 0.2 * math.sin(step / 5.0)
    return {
        "mode": "PLAN_STEP",
        "encode_flag": False,
        "confidence": round(confidence, 3),
        "reason": "synthetic" if step % 7 else "cache_hit",
    }


def legal_actions_stub() -> List[Mapping[str, object]]:
    return [
        {"id": "MOVE_UP", "index": 0},
        {"id": "MOVE_DOWN", "index": 1},
        {"id": "MOVE_LEFT", "index": 2},
        {"id": "MOVE_RIGHT", "index": 3},
        {"id": "WAIT", "index": 4},
    ]


def build_payload(
    *,
    telemetry: PokemonTelemetry,
    step: int,
    action_index: int,
    plan_context: Mapping[str, object],
) -> Mapping[str, object]:
    timestamp = datetime.now(timezone.utc).isoformat()
    frame_features = frame_features_from_telemetry(telemetry, step)
    gate_payload = core_gate_payload(step)
    return {
        "source": "capture.overworld",
        "timestamp": timestamp,
        "context": {
            "domain": "overworld",
            "status": "PLANLET_ACTIVE",
            "mode": "explore",
            "step_index": step,
            "plan": plan_context,
        },
        "observation": {
            "frame": telemetry.extra.get("frame", int(step)),
            "overworld": {
                "area_id": telemetry.area_id,
                "x": telemetry.x,
                "y": telemetry.y,
            },
        },
        "telemetry": {
            "core": {
                "legal_actions": legal_actions_stub(),
                "action": {"id": "MOVE_RIGHT" if action_index == 3 else "WAIT", "index": action_index},
                "action_mask": [0.0 for _ in range(5)],
                "gate": gate_payload,
                "fractions": {"encode": 0.15, "query": 0.7, "skip": 0.15},
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


def capture_with_pyboy(args: argparse.Namespace) -> Iterable[Mapping[str, object]]:
    if PyBoyPokemonAdapter is None or PyBoyConfig is None:
        raise RuntimeError("PyBoy is not available. Install pyboy or run with --dry-run.")

    cfg = PyBoyConfig(
        rom_path=args.rom,
        window_type=args.window,
    )
    adapter = PyBoyPokemonAdapter(cfg)
    overworld = OverworldAdapter(adapter)

    plan_context = make_plan_context(run_id="pyboy")
    telemetry = overworld.reset()
    yield build_payload(telemetry=telemetry, step=0, action_index=4, plan_context=plan_context)

    for step in range(1, args.steps):
        action_index = step % 5
        # Cycle through cardinal moves with occasional wait to keep things deterministic.
        if action_index == 0:
            action = PokemonAction("SCRIPT", {"inputs": ["UP"], "frames": 6})
        elif action_index == 1:
            action = PokemonAction("SCRIPT", {"inputs": ["DOWN"], "frames": 6})
        elif action_index == 2:
            action = PokemonAction("SCRIPT", {"inputs": ["LEFT"], "frames": 6})
        elif action_index == 3:
            action = PokemonAction("SCRIPT", {"inputs": ["RIGHT"], "frames": 6})
        else:
            action = PokemonAction("WAIT", {"frames": args.frames_per_step})
        telemetry = overworld.step(action)
        yield build_payload(telemetry=telemetry, step=step, action_index=action_index, plan_context=plan_context)

    adapter.close()


def synthetic_trace(args: argparse.Namespace) -> Iterable[Mapping[str, object]]:
    plan_context = make_plan_context(run_id="synthetic")
    graph = WorldGraph(load_static=True)
    # Follow a simple loop in Pallet Town using static warp hints.
    coords: List[Tuple[int, int]] = [(x, 4) for x in range(4)] + [(3, y) for y in range(5, 8)] + [(x, 7) for x in range(2, -1, -1)]
    area_id = 1  # Arbitrary map identifier for Pallet Town.
    for step in range(args.steps):
        x, y = coords[step % len(coords)]
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
        yield build_payload(telemetry=telemetry, step=step, action_index=action_index, plan_context=plan_context)
    # Silence unused variable warning.
    _ = graph


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries = capture_with_pyboy(args) if not args.dry_run else synthetic_trace(args)
    with OverworldTraceRecorder(args.output, validate=False) as recorder:
        for payload in entries:
            recorder.record(payload)


if __name__ == "__main__":
    main()
