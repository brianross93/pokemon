from __future__ import annotations

from src.overworld.graph.overworld_memory import OverworldMemory
from src.plan.planner_llm import PlanletSpec
from src.plan.prompt_builder import PlannerPromptBuilder
from src.srfbam.tasks.events import PlanletEvent


def _make_event(planlet_id: str, kind: str, status: str, reason: str | None) -> PlanletEvent:
    return PlanletEvent(
        plan_id="p",
        planlet_id=planlet_id,
        planlet_kind=kind,
        status=status,
        reason=reason,
        step_index=0,
        telemetry={},
        trace=[],
    )


def test_prompt_builder_summarises_history_and_entities() -> None:
    memory = OverworldMemory()
    memory.write(OverworldMemory.make_map_region("01", "Pallet"))
    memory.write(OverworldMemory.make_tile("01", 0, 0, passable=True, terrain="path"))
    memory.write(OverworldMemory.make_tile("01", 1, 0, passable=True, terrain="path"))

    history = [
        _make_event("p1", "NAVIGATE_TO", "PLANLET_COMPLETE", None),
        _make_event("p2", "HEAL_AT_CENTER", "PLANLET_STALLED", "MENU_DESYNC"),
    ]
    pending = [
        PlanletSpec(id="p3", kind="BUY_ITEM", args={}, pre=[], post=[], hints={}, timeout_steps=30, recovery=[]),
    ]

    builder = PlannerPromptBuilder(max_history=5)
    prompt = builder.build(goal="Test goal", memory=memory, history=history, pending=pending)

    assert prompt.goal == "Test goal"
    assert any(entry["type"] == "MapRegion" for entry in prompt.entity_summary)
    assert prompt.outcomes["failures"].get("MENU_DESYNC") == 1
    assert prompt.outcomes["delta"] == 0
    assert prompt.pending_planlets[0]["timeout_steps"] == 30

    formatted = builder.format_prompt(prompt)
    assert "Failure Taxonomy" in formatted
    assert "MENU_DESYNC" in formatted
    assert "delta=0" in formatted
