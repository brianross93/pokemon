"""
Demonstration script that wires the symbolic controller to a mock adapter.

This version uses synthetic telemetry to showcase the flow until a real emulator
adapter (PyBoy) is available. It prints gate decisions and the controller's
belief summaries for a short horizon.
"""
from __future__ import annotations

import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

# Add repository root so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.middleware import PokemonAction, PokemonTelemetry, SymbolicController
from src.middleware.pokemon_adapter import PokemonBlueAdapter
from src.knowledge.knowledge_graph import Context


@dataclass
class MockConfig:
    areas: List[int]
    target_species: int


class MockPokemonAdapter(PokemonBlueAdapter):
    """Minimal adapter that produces deterministic telemetry samples."""

    def __init__(self, config: MockConfig) -> None:
        self.config = config
        self.step_id = 0
        self._rng = random.Random(42)
        self._area_cycle: Iterator[int] = itertools.cycle(config.areas)
        self._current_area = next(self._area_cycle)

    def reset(self) -> PokemonTelemetry:
        self.step_id = 0
        self._current_area = next(self._area_cycle)
        return self._telemetry(in_battle=False, species=None)

    def step(self, action: PokemonAction) -> PokemonTelemetry:
        self.step_id += 1
        if action.name == "ASSOC" and action.payload.get("area_id") is not None:
            self._current_area = action.payload["area_id"]
        encounter = (self.step_id % 5 == 0)
        species = self.config.target_species if encounter and self._rng.random() < 0.4 else None
        return self._telemetry(in_battle=encounter, species=species)

    def _telemetry(self, in_battle: bool, species: int | None) -> PokemonTelemetry:
        return PokemonTelemetry(
            area_id=self._current_area,
            x=self.step_id % 16,
            y=(self.step_id // 4) % 16,
            in_grass=True,
            in_battle=in_battle,
            encounter_species_id=species,
            step_counter=self.step_id,
            elapsed_ms=self.step_id * 100.0,
            method="grass",
        )


def main() -> None:
    target_species = 25  # Pikachu
    controller = SymbolicController(
        target_species_id=target_species,
        target_entity_name="Pikachu",
        game="blue",
        min_confidence=0.7,
        min_rule_samples=3,
    )

    adapter = MockPokemonAdapter(MockConfig(areas=[7, 9], target_species=target_species))
    telemetry = adapter.reset()

    candidate_contexts = [
        Context(game="blue", area_id=area_id, method="grass")
        for area_id in adapter.config.areas
    ]

    for step in range(20):
        controller.observe(telemetry)
        decision = controller.decide(telemetry, candidate_contexts)
        summary = controller.summarise(telemetry)

        print(
            f"[step {step:02d}] gate={decision.mode.value:6s} "
            f"reason={decision.reason:24s} "
            f"area={telemetry.area_id} "
            f"confidence={summary['confidence']}"
        )

        if decision.mode.value == "ASSOC" and decision.target_context:
            next_area = decision.target_context.area_id
            telemetry = adapter.step(PokemonAction("ASSOC", {"area_id": next_area}))
        else:
            telemetry = adapter.step(PokemonAction("WAIT", {"frames": 20}))

    print("Final summary:", controller.summarise(telemetry))


if __name__ == "__main__":
    main()
