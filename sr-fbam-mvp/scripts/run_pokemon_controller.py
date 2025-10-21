"""
Run the symbolic controller against a live PyBoy emulator session.

The script expects a Pokemon Blue ROM and the memory offsets discovered via
``scripts/inspect_pyboy_memory.py``. It issues simple random-walk actions to
illustrate how the middleware updates its knowledge base with real telemetry.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

from src.knowledge.knowledge_graph import Context
from src.middleware import PokemonAction, SymbolicController
from src.middleware.controller import GateDecision, GateMode
from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter


DEFAULT_ADDRESSES: Dict[str, int] = {
    "map_id": 0xD35E,
    "player_x": 0xD361,
    "player_y": 0xD362,
    "in_battle": 0xD057,
    "species_id": 0xD058,
    "in_grass": 0xD5A5,
    # step_counter left unset; adapter will fall back to frame count.
}


def parse_address(arg: str) -> Tuple[str, int]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"Invalid address '{arg}'. Expected name=0xVALUE.")
    name, value = arg.split("=", 1)
    try:
        offset = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer '{value}' in address spec.") from exc
    return name, offset


def random_walk_action() -> PokemonAction:
    """Generate a simple exploratory action."""
    choice = random.choice(["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "WAIT"])
    if choice == "WAIT":
        return PokemonAction("WAIT", {"frames": random.randint(30, 90)})
    return PokemonAction(choice, {"frames": random.randint(4, 10)})


def log_decision(step: int, decision: GateDecision, summary: Dict[str, float]) -> None:
    context_info = (
        f"area={summary.get('area_id', 'n/a')} "
        f"rate={summary.get('mean_rate', 0.0):.3f} "
        f"share={summary.get('mean_share', 0.0):.3f} "
        f"conf={summary.get('confidence', 0.0)}"
    )
    print(f"[step {step:04d}] gate={decision.mode.value:6s} reason={decision.reason:20s} {context_info}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run symbolic controller on Pokemon Blue via PyBoy.")
    parser.add_argument("--rom", required=True, help="Path to Pokemon Blue ROM.")
    parser.add_argument("--steps", type=int, default=500, help="Controller iterations to run.")
    parser.add_argument(
        "--address",
        action="append",
        type=parse_address,
        default=[],
        help="Override memory offset (name=0xVALUE). Can be repeated.",
    )
    parser.add_argument("--species-id", type=int, default=25, help="Target species (default: 25 Pikachu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for action sampling.")
    args = parser.parse_args()

    random.seed(args.seed)
    addresses = dict(DEFAULT_ADDRESSES)
    for name, offset in args.address:
        addresses[name] = offset

    cfg = PyBoyConfig(rom_path=args.rom, addresses=addresses)
    adapter = PyBoyPokemonAdapter(cfg)
    controller = SymbolicController(
        target_species_id=args.species_id,
        target_entity_name=f"Species{args.species_id}",
        game="blue",
        min_confidence=0.7,
        min_rule_samples=5,
    )

    telemetry = adapter.reset()
    candidate_contexts: List[Context] = [
        Context(game="blue", area_id=telemetry.area_id, method=telemetry.method),
    ]

    for step in range(args.steps):
        controller.observe(telemetry)
        decision = controller.decide(telemetry, candidate_contexts)
        summary = controller.summarise(telemetry)
        summary["area_id"] = telemetry.area_id
        log_decision(step, decision, summary)

        if decision.mode is GateMode.ASSOC and decision.target_context:
            # TODO: Implement navigation to target_context; placeholder wait.
            telemetry = adapter.step(PokemonAction("WAIT", {"frames": 60}))
        else:
            action = random_walk_action()
            telemetry = adapter.step(action)

        # Refresh candidate contexts with the latest area.
        if not any(ctx.area_id == telemetry.area_id for ctx in candidate_contexts):
            candidate_contexts.append(Context(game="blue", area_id=telemetry.area_id, method=telemetry.method))

    print("Final summary:", controller.summarise(telemetry))


if __name__ == "__main__":
    main()
