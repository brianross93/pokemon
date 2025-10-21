from __future__ import annotations

import copy

from src.plan import PlanCompiler
from src.plan.planner_llm import PlanBundle, PlanletSpec, validate_plan_bundle
from src.srfbam.tasks.overworld import OverworldExecutor
from src.overworld.encounter import EncounterResult, OverworldSnapshot
from src.overworld.skills.base import BaseSkill, SkillProgress, SkillStatus

DIRECTION_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


class RandomBattleSkill(BaseSkill):
    name = "RandomBattleSkill"
    _GLOBAL_TRIGGERED = False

    def legal_actions(self, observation, graph):
        return ({"kind": "wait"},)

    def select_action(self, observation, graph):
        return {"kind": "wait"}

    def progress(self, graph):
        if not RandomBattleSkill._GLOBAL_TRIGGERED:
            RandomBattleSkill._GLOBAL_TRIGGERED = True
            return SkillProgress(status=SkillStatus.STALLED, reason="RANDOM_BATTLE")
        return SkillProgress(status=SkillStatus.SUCCEEDED)


def run_executor_sequence(
    executor: OverworldExecutor,
    base_obs: dict,
    start_tile: list[int],
    *,
    warp_destination: list[int] | None = None,
    max_steps: int = 10,
):
    current_tile = list(start_tile)
    results = []
    for _ in range(max_steps):
        obs = copy.deepcopy(base_obs)
        obs["overworld"]["player"]["tile"] = current_tile.copy()
        result = executor.step(obs)
        results.append(result)
        if result.status in {"PLANLET_COMPLETE", "PLAN_COMPLETE", "PLANLET_STALLED"}:
            break
        action = result.action
        label = action.get("label")
        if action.get("kind") == "button" and label in DIRECTION_DELTAS:
            dx, dy = DIRECTION_DELTAS[label]
            current_tile[0] += dx
            current_tile[1] += dy
        elif action.get("kind") == "wait" and action.get("meta", {}).get("reason") == "warp_exit" and warp_destination is not None:
            current_tile = list(warp_destination)
    return results


def test_overworld_executor_navigate_planlet_completes() -> None:
    raw_plan = {
        "plan_id": "T1",
        "goal": "Move one tile south",
        "planlets": [
            {
                "id": "P1",
                "kind": "NAVIGATE_TO",
                "args": {"target": {"map": "01", "tile": [0, 1]}},
            }
        ],
    }
    plan = PlanCompiler().compile(validate_plan_bundle(raw_plan))
    executor = OverworldExecutor()
    executor.load_compiled_plan(plan)

    base_obs = {
        "overworld": {
            "map": {"id": "01", "name": "Map_01"},
            "player": {"map_id": "01", "tile": [0, 0], "facing": "south"},
            "tiles": [
                {"map_id": "01", "x": 0, "y": 0, "passable": True, "terrain": "path", "special": ""},
                {"map_id": "01", "x": 0, "y": 1, "passable": True, "terrain": "path", "special": ""},
            ],
            "warps": [],
            "npcs": [],
            "menus": [],
        }
    }

    results = run_executor_sequence(executor, base_obs, [0, 0])
    assert results
    assert results[-1].status in {"PLANLET_COMPLETE", "PLAN_COMPLETE"}


def test_navigate_skill_detours_around_obstacle() -> None:
    raw_plan = {
        "plan_id": "T2",
        "goal": "Navigate around obstacle",
        "planlets": [
            {
                "id": "P1",
                "kind": "NAVIGATE_TO",
                "args": {"target": {"map": "01", "tile": [0, 2]}},
            }
        ],
    }
    plan = PlanCompiler().compile(validate_plan_bundle(raw_plan))
    executor = OverworldExecutor()
    executor.load_compiled_plan(plan)

    tiles = [
        {"map_id": "01", "x": 0, "y": 0, "passable": True, "terrain": "path", "special": ""},
        {"map_id": "01", "x": 0, "y": 1, "passable": False, "terrain": "rock", "special": ""},
        {"map_id": "01", "x": 0, "y": 2, "passable": True, "terrain": "path", "special": ""},
        {"map_id": "01", "x": 1, "y": 0, "passable": True, "terrain": "path", "special": ""},
        {"map_id": "01", "x": 1, "y": 1, "passable": True, "terrain": "path", "special": ""},
        {"map_id": "01", "x": 1, "y": 2, "passable": True, "terrain": "path", "special": ""},
    ]
    base_obs = {
        "overworld": {
            "map": {"id": "01", "name": "Map_01"},
            "player": {"map_id": "01", "tile": [0, 0], "facing": "south"},
            "tiles": tiles,
            "warps": [],
            "npcs": [],
            "menus": [],
        }
    }

    results = run_executor_sequence(executor, base_obs, [0, 0])
    assert results[-1].status in {"PLANLET_COMPLETE", "PLAN_COMPLETE"}


def test_navigate_skill_uses_warp_exit() -> None:
    raw_plan = {
        "plan_id": "T3",
        "goal": "Use warp to reach destination",
        "planlets": [
            {
                "id": "P1",
                "kind": "NAVIGATE_TO",
                "args": {"target": {"map": "01", "tile": [0, 2]}},
            }
        ],
    }
    plan = PlanCompiler().compile(validate_plan_bundle(raw_plan))
    executor = OverworldExecutor()
    executor.load_compiled_plan(plan)

    base_obs = {
        "overworld": {
            "map": {"id": "01", "name": "Map_01"},
            "player": {"map_id": "01", "tile": [0, 0], "facing": "south"},
            "tiles": [
                {"map_id": "01", "x": 0, "y": 0, "passable": True, "terrain": "path", "special": ""},
                {"map_id": "01", "x": 0, "y": 2, "passable": True, "terrain": "path", "special": ""},
            ],
            "warps": [
                {"id": "0", "src_tile": [0, 0], "dst_map_id": "01", "dst_tile": [0, 2], "src_map_id": "01"},
            ],
            "npcs": [],
            "menus": [],
        }
    }

    results = run_executor_sequence(executor, base_obs, [0, 0], warp_destination=[0, 2])
    assert results[-1].status in {"PLANLET_COMPLETE", "PLAN_COMPLETE"}


def test_navigate_skill_reports_stalled_when_blocked() -> None:
    raw_plan = {
        "plan_id": "T4",
        "goal": "Hit dead end",
        "planlets": [
            {
                "id": "P1",
                "kind": "NAVIGATE_TO",
                "args": {"target": {"map": "01", "tile": [2, 0]}},
            }
        ],
    }
    plan = PlanCompiler().compile(validate_plan_bundle(raw_plan))
    executor = OverworldExecutor()
    executor.load_compiled_plan(plan)

    base_obs = {
        "overworld": {
            "map": {"id": "01", "name": "Map_01"},
            "player": {"map_id": "01", "tile": [0, 0], "facing": "south"},
            "tiles": [
                {"map_id": "01", "x": 0, "y": 0, "passable": True, "terrain": "path", "special": ""},
                {"map_id": "01", "x": 1, "y": 0, "passable": False, "terrain": "wall", "special": ""},
                {"map_id": "01", "x": 2, "y": 0, "passable": True, "terrain": "path", "special": ""},
            ],
            "warps": [],
            "npcs": [],
            "menus": [],
        }
    }

    results = run_executor_sequence(executor, base_obs, [0, 0])
    assert results[-1].status == "PLANLET_STALLED"


def test_encounter_bridge_handles_clean_exit() -> None:
    RandomBattleSkill._GLOBAL_TRIGGERED = False
    raw_plan = {
        "plan_id": "encounter",
        "goal": "Trigger battle and resume",
        "planlets": [
            {
                "id": "nav",
                "kind": "NAVIGATE_TO",
                "args": {"target": {"map": "01", "tile": [0, 1]}},
            }
        ],
    }
    plan = PlanCompiler().compile(validate_plan_bundle(raw_plan))
    executor = OverworldExecutor(skill_registry={"NavigateSkill": RandomBattleSkill})
    executor.load_compiled_plan(plan)

    def battle_handler(request):
        return EncounterResult(
            status="ESCAPED",
            turns=2,
            reason=None,
            snapshot_return=request.snapshot_overworld,
        )

    executor.register_battle_handler(battle_handler)

    base_obs = {
        "overworld": {
            "map": {"id": "01", "name": "Map_01"},
            "player": {"map_id": "01", "tile": [0, 0], "facing": "south"},
            "tiles": [
                {"map_id": "01", "x": 0, "y": 0, "passable": True, "terrain": "path", "special": ""},
                {"map_id": "01", "x": 0, "y": 1, "passable": True, "terrain": "path", "special": ""},
            ],
            "warps": [],
            "npcs": [],
            "menus": [],
        }
    }

    result = executor.step(copy.deepcopy(base_obs))
    encounter_events = result.telemetry.get("encounter", [])
    phases = [event.get("phase") for event in encounter_events]
    assert phases == ["battle.entry", "battle.exit"]
    assert result.status == "IN_PROGRESS"
    assert executor.state.current_planlet is not None


def test_encounter_bridge_detects_invariant_violation() -> None:
    RandomBattleSkill._GLOBAL_TRIGGERED = False
    raw_plan = {
        "plan_id": "encounter_violation",
        "goal": "Trigger battle and violate invariants",
        "planlets": [
            {
                "id": "nav",
                "kind": "NAVIGATE_TO",
                "args": {"target": {"map": "01", "tile": [0, 1]}},
            }
        ],
    }
    plan = PlanCompiler().compile(validate_plan_bundle(raw_plan))
    executor = OverworldExecutor(skill_registry={"NavigateSkill": RandomBattleSkill})
    executor.load_compiled_plan(plan)

    def battle_handler(request):
        snapshot = OverworldSnapshot(
            rng_state=request.snapshot_overworld.rng_state,
            map_id=request.snapshot_overworld.map_id,
            tile_xy=request.snapshot_overworld.tile_xy,
            facing=request.snapshot_overworld.facing,
            menu_open=True,
            party_summary=dict(request.snapshot_overworld.party_summary),
            graph_snapshot=request.snapshot_overworld.graph_snapshot,
        )
        return EncounterResult(
            status="ESCAPED",
            turns=3,
            reason=None,
            snapshot_return=snapshot,
        )

    executor.register_battle_handler(battle_handler)

    base_obs = {
        "overworld": {
            "map": {"id": "01", "name": "Map_01"},
            "player": {"map_id": "01", "tile": [0, 0], "facing": "south"},
            "tiles": [
                {"map_id": "01", "x": 0, "y": 0, "passable": True, "terrain": "path", "special": ""},
                {"map_id": "01", "x": 0, "y": 1, "passable": True, "terrain": "path", "special": ""},
            ],
            "warps": [],
            "npcs": [],
            "menus": [],
        }
    }

    result = executor.step(copy.deepcopy(base_obs))
    encounter_events = result.telemetry.get("encounter", [])
    phases = [event.get("phase") for event in encounter_events]
    assert phases == ["battle.entry", "battle.exit"]
    assert result.status == "PLANLET_STALLED"
    exit_event = [event for event in encounter_events if event.get("phase") == "battle.exit"][0]
    assert exit_event["result"]["status"] == "ESCAPED"
    assert exit_event["overworld_invariants"]["menu_closed"] is False
