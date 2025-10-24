from __future__ import annotations

import copy
import logging
import types

import numpy as np

from src.plan import PlanCompiler
from src.plan.planner_llm import PlanBundle, PlanletSpec, validate_plan_bundle
from src.srfbam.tasks.overworld import OverworldExecutor
from src.overworld.encounter import EncounterResult, OverworldSnapshot
from src.overworld.env.overworld_adapter import OverworldObservation
from src.overworld.extractor.overworld_extractor import OverworldExtractor
from src.overworld.skills.base import BaseSkill, SkillProgress, SkillStatus
from scripts.run_overworld_controller import PlanCoordinator

DIRECTION_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def make_observation(
    overworld_payload: dict[str, object],
    *,
    metadata: dict[str, object] | None = None,
) -> OverworldObservation:
    frame = np.zeros((40, 120, 3), dtype=np.uint8)
    meta = {
        "visual_overworld": copy.deepcopy(overworld_payload),
        "map_id": str(overworld_payload.get("map", {}).get("id", "unknown")),
        "map_name": overworld_payload.get("map", {}).get("name"),
        "player_tile": list(overworld_payload.get("player", {}).get("tile", [0, 0])),
        "player_facing": overworld_payload.get("player", {}).get("facing", "south"),
    }
    if metadata:
        meta.update(metadata)
    return OverworldObservation(framebuffer=frame, ram=None, metadata=meta)


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
    base_overworld: dict[str, object],
    start_tile: list[int],
    *,
    warp_destination: list[int] | None = None,
    max_steps: int = 10,
) -> list:
    current_tile = list(start_tile)
    results = []
    base_payload = (
        copy.deepcopy(base_overworld["overworld"]) if isinstance(base_overworld, dict) and "overworld" in base_overworld else copy.deepcopy(base_overworld)
    )
    for _ in range(max_steps):
        payload = copy.deepcopy(base_payload)
        player = payload.setdefault("player", {})
        player["tile"] = current_tile.copy()
        observation = make_observation(payload)
        result = executor.step(observation)
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

    result = executor.step(make_observation(copy.deepcopy(base_obs["overworld"])))
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

    result = executor.step(make_observation(copy.deepcopy(base_obs["overworld"])))
    encounter_events = result.telemetry.get("encounter", [])
    phases = [event.get("phase") for event in encounter_events]
    assert phases == ["battle.entry", "battle.exit"]
    assert result.status == "PLANLET_STALLED"
    exit_event = [event for event in encounter_events if event.get("phase") == "battle.exit"][0]
    assert exit_event["result"]["status"] == "ESCAPED"
    assert exit_event["overworld_invariants"]["menu_closed"] is False


def test_naming_snapshot_emits_presets_and_cursor_history() -> None:
    extractor = OverworldExtractor()
    frame_hash = "frame_001"
    tiles = []
    # RED row
    red_letters = [0x80 + ord(letter) - ord("A") for letter in "RED"]
    for offset, tile_id in enumerate(red_letters):
        tiles.append(
            {
                "map_id": "FF",
                "x": offset,
                "y": 0,
                "passable": True,
                "terrain": "menu",
                "special": "",
                "screen": {"x": 2 + offset, "y": 8},
                "tile_id": tile_id,
            }
        )
    # BLUE row
    blue_letters = [0x80 + ord(letter) - ord("A") for letter in "BLUE"]
    for offset, tile_id in enumerate(blue_letters):
        tiles.append(
            {
                "map_id": "FF",
                "x": offset,
                "y": 1,
                "passable": True,
                "terrain": "menu",
                "special": "",
                "screen": {"x": 2 + offset, "y": 9},
                "tile_id": tile_id,
            }
        )

    overworld = {
        "frame": {"hash": frame_hash},
        "menus": [{"id": "overlay:naming", "path": ["OVERLAY"], "open": True}],
        "tiles": tiles,
        "dialog_lines": ["NAME?"],
    }
    metadata = {
        "frame_shape": (40, 120),
        "game_area": {
            "sprites": [
                {
                    "index": 0,
                    "on_screen": True,
                    "screen": {"x": 3, "y": 8},
                }
            ]
        },
    }

    naming_state = extractor._decode_naming_screen(overworld, metadata)
    assert naming_state is not None
    cursor = naming_state.get("cursor", {})
    assert cursor.get("letter") == "E"
    assert cursor.get("source") == "sprite"
    history = naming_state.get("cursor_history", [])
    assert history and history[0]["letter"] == "E"
    preset_labels = {entry["label"] for entry in naming_state.get("presets", [])}
    assert {"RED", "BLUE"} <= preset_labels
    assert naming_state.get("dialog_lines") == ["NAME?"]

    # Second frame without sprites should fall back to cursor history.
    overworld2 = copy.deepcopy(overworld)
    overworld2["frame"] = {"hash": "frame_002"}
    metadata2 = {"frame_shape": (40, 120), "game_area": {"sprites": []}}
    naming_state2 = extractor._decode_naming_screen(overworld2, metadata2)
    assert naming_state2 is not None
    cursor2 = naming_state2.get("cursor", {})
    assert cursor2.get("letter") == "E"
    assert cursor2.get("source") == "history"


def test_decode_naming_screen_without_overlay_menu() -> None:
    extractor = OverworldExtractor()
    overworld = {
        "tiles": [
            {"map_id": "screen_local", "x": 0, "y": 0, "tile_id": 0x80, "screen": {"x": 2, "y": 8}, "passable": True},
            {"map_id": "screen_local", "x": 1, "y": 0, "tile_id": 0x81, "screen": {"x": 3, "y": 8}, "passable": True},
            {"map_id": "screen_local", "x": 2, "y": 0, "tile_id": 0x82, "screen": {"x": 4, "y": 8}, "passable": True},
            {"map_id": "screen_local", "x": 0, "y": 1, "tile_id": 0x83, "screen": {"x": 2, "y": 9}, "passable": True},
            {"map_id": "screen_local", "x": 1, "y": 1, "tile_id": 0x84, "screen": {"x": 3, "y": 9}, "passable": True},
            {"map_id": "screen_local", "x": 2, "y": 1, "tile_id": 0x85, "screen": {"x": 4, "y": 9}, "passable": True},
        ],
        "menus": [],
    }
    metadata = {"frame_shape": (40, 120)}
    naming_state = extractor._decode_naming_screen(overworld, metadata)  # type: ignore[attr-defined]
    assert naming_state is None  # too few entries without overlay

    # Add enough tiles to surpass threshold
    for col in range(20):
        for row in range(6):
            overworld["tiles"].append(
                {
                    "map_id": "screen_local",
                    "x": col,
                    "y": row,
                    "tile_id": 0x80 + (col % 26),
                    "screen": {"x": 2 + col, "y": 8 + row},
                    "passable": True,
                }
            )
    naming_state = extractor._decode_naming_screen(overworld, metadata)  # type: ignore[attr-defined]
    assert naming_state is not None
    assert isinstance(naming_state.get("grid_letters"), list)


def test_plan_coordinator_mission_plan_includes_naming_snapshot() -> None:
    naming_state = {
        "grid_letters": [["R", "E", "D"]],
        "cursor": {"row": 0, "col": 1, "letter": "E"},
        "cursor_history": [{"row": 0, "col": 1, "letter": "E"}],
        "presets": [{"label": "RED", "row": 0, "col": 0}],
        "dialog_lines": ["WHAT is your name?"],
    }
    last_payload = {
        "overworld": {
            "map": {"id": "screen_local", "name": "Bedroom"},
            "player": {"map_id": "screen_local", "tile": [5, 5], "facing": "north"},
            "tiles": [
                {"map_id": "screen_local", "x": 5, "y": 5, "passable": True, "terrain": "floor", "screen": {"x": 10, "y": 9}},
                {"map_id": "screen_local", "x": 6, "y": 5, "passable": True, "terrain": "floor", "screen": {"x": 11, "y": 9}},
            ],
            "menus": [{"id": "overlay:naming", "path": ["OVERLAY"], "open": True}],
            "naming_screen": naming_state,
            "tile_adjacency": {("screen_local", 5, 5): [("screen_local", 6, 5)]},
        }
    }
    dummy_executor = types.SimpleNamespace(
        extractor=types.SimpleNamespace(last_payload=last_payload),
        memory=None,
        state=types.SimpleNamespace(current_planlet=None, plan_queue=[]),
    )
    coordinator = PlanCoordinator(
        executor=dummy_executor,
        service=None,
        allow_search=False,
        nearby_limit=5,
        backend_label="test",
        logger=logging.getLogger("plan_coordinator_test"),
    )
    mission_plan = coordinator._mission_plan_for_prompt()
    snapshot = mission_plan["environment"]["overworld_snapshot"]
    assert "naming_screen" in snapshot
    naming_snapshot = snapshot["naming_screen"]
    assert naming_snapshot["cursor"]["letter"] == "E"
    assert naming_snapshot["presets"][0]["label"] == "RED"




