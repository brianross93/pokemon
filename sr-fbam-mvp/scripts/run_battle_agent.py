#!/usr/bin/env python3
"""
Run the SR-FBAM battle agent against a live PyBoy session.

The harness wires the PyBoy adapter to the symbolic battle stack
(BlueRAMAdapter -> BlueExtractor -> GraphMemory -> SRFBAMBattleAgent)
and executes the selected moves or switches via deterministic button scripts.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.middleware.pokemon_adapter import PokemonAction
from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter
from src.pkmn_battle.env import DEFAULT_BLUE_BATTLE_RAM_MAP
from src.pkmn_battle.env.blue_ram_adapter import BlueRAMAdapter
from src.pkmn_battle.extractor.blue_extractor import BlueExtractor
from src.pkmn_battle.graph.schema import WriteOp
from src.srfbam.tasks.battle import SRFBAMBattleAgent


DEFAULT_ADDRESSES: Dict[str, int] = {
    "map_id": 0xD35E,
    "player_x": 0xD361,
    "player_y": 0xD362,
    "in_battle": 0xD057,
    "species_id": 0xD058,
    "in_grass": 0xD5A5,
}


def _format_action(action: Optional[Dict[str, object]]) -> str:
    if not action:
        return "none"
    kind = action.get("kind", "unknown")
    index = action.get("index")
    if index is None:
        return str(kind)
    return f"{kind}{index}"


def _percentile(data: List[float], q: float) -> float:
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _print_profile(profile: Dict[str, Dict[str, float]]) -> None:
    order = ["encode", "assoc", "follow", "write", "skip"]

    def fmt(key: str) -> str:
        stats = profile.get(key, {})
        avg = stats.get("avg_ms", 0.0)
        count = int(stats.get("count", 0))
        return f"{key}=avg{avg:.1f}ms/{count}"

    parts = [fmt(key) for key in order]
    print("Profile:", ", ".join(parts))


def translate_action_to_pokemon(action: dict, wait_frames: int) -> PokemonAction:
    """Translate a legal battle action into a PokemonAction understood by PyBoy."""

    kind = action.get("kind")
    index = int(action.get("index", 0))
    if kind == "move":
        return PokemonAction("BATTLE_MOVE", {"slot": index})
    if kind == "switch":
        return PokemonAction("BATTLE_SWITCH", {"slot": index})
    return PokemonAction("WAIT", {"frames": wait_frames})


def build_battle_env(
    adapter: PyBoyPokemonAdapter,
    *,
    wait_frames: int,
) -> BlueRAMAdapter:
    def executor(action: dict) -> None:
        pokemon_action = translate_action_to_pokemon(action, wait_frames)
        adapter.step(pokemon_action)

    return BlueRAMAdapter(
        read_u8=adapter.read_u8,
        ram_map=DEFAULT_BLUE_BATTLE_RAM_MAP,
        snapshot_extra=lambda: {"frame": int(adapter.pyboy.frame_count)},
        action_executor=executor,
        reset_callback=lambda: adapter.reset(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SR-FBAM battle agent via PyBoy.")
    parser.add_argument("--rom", required=True, help="Path to Pokemon Blue ROM.")
    parser.add_argument("--steps", type=int, default=100, help="Number of agent steps to execute.")
    parser.add_argument("--visual", action="store_true", help="Open PyBoy window instead of running headless.")
    parser.add_argument("--wait-frames", type=int, default=90, help="Frames to wait after battle actions.")
    parser.add_argument("--log-interval", type=int, default=1, help="How often to print telemetry (in steps).")
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to a JSONL file where per-step telemetry will be appended.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-gate latency breakdown (average ms and counts) at the end of the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PyBoyConfig(
        rom_path=args.rom,
        window_type="SDL2" if args.visual else "null",
        addresses=dict(DEFAULT_ADDRESSES),
    )

    adapter = PyBoyPokemonAdapter(cfg)
    battle_env = build_battle_env(adapter, wait_frames=args.wait_frames)
    extractor = BlueExtractor()
    agent = SRFBAMBattleAgent(env=battle_env, extractor=extractor)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")

    try:
        obs = agent.reset()
        print("Initial observation:", obs)
        start_time = time.time()
        latency_history: List[float] = []
        for step in range(1, args.steps + 1):
            obs = agent.step()
            telemetry = agent.telemetry()
            latency_history.append(telemetry.latency_ms)
            turn = obs.get("turn", step)

            gate = telemetry.gate_decision.mode if telemetry.gate_decision else "n/a"
            reason = telemetry.gate_decision.reason if telemetry.gate_decision else "-"
            encode_flag = telemetry.gate_decision.encode_flag if telemetry.gate_decision else False
            confidence = telemetry.gate_decision.confidence if telemetry.gate_decision else None
            fractions = telemetry.fractions
            speedup = telemetry.speedup
            pred = speedup.get("predicted")
            obs_speed = speedup.get("observed")
            pred_str = f"{pred:.2f}" if pred is not None else "n/a"
            obs_str = f"{obs_speed:.2f}" if obs_speed is not None else "n/a"
            action_desc = _format_action(telemetry.last_action)
            p50 = _percentile(latency_history, 0.5)
            p95 = _percentile(latency_history, 0.95)
            hop_count = len(telemetry.hop_trace)
            legal_count = len(telemetry.legal_actions)

            if step % args.log_interval == 0:
                print(
                    f"T={turn:04d} op={gate} encode={encode_flag} reason={reason} "
                    f"e/q/s={fractions.get('encode', 0.0):.2f}/"
                    f"{fractions.get('query', 0.0):.2f}/"
                    f"{fractions.get('skip', 0.0):.2f} "
                    f"latency={telemetry.latency_ms:.1f}ms "
                    f"p50={p50:.1f}ms p95={p95:.1f}ms "
                    f"speed(pred/obs)={pred_str}/{obs_str} "
                    f"act={action_desc} legal={legal_count} hops={hop_count} "
                    f"fallback={telemetry.fallback_required}"
                )

            payload = telemetry.to_payload()
            record = {
                "source": "sr-fbam.battle.agent",
                "context": {
                    "domain": "battle",
                    "battle": {"turn": int(turn), "step": int(step)},
                },
                "observation": obs,
                "telemetry": payload,
            }
            json.dump(record, log_handle)
            log_handle.write("\n")

        elapsed = time.time() - start_time
        print(f"Completed {args.steps} steps in {elapsed:.2f}s")
        if args.profile:
            _print_profile(agent.profile_stats())
    finally:
        log_handle.close()
        with contextlib.suppress(Exception):
            adapter.close()


if __name__ == "__main__":
    main()
