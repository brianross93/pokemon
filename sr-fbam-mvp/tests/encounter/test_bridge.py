from __future__ import annotations

import copy

from src.overworld.encounter import EncounterBridge, EncounterResult, OverworldSnapshot
from src.overworld.encounter.types import EncounterRequest
from src.overworld.graph.overworld_memory import OverworldMemory


def _make_observation(menu_open: bool = False):
    menus = []
    if menu_open:
        menus.append({"path": ["PKMN"], "open": True})
    return {
        "overworld": {
            "map": {"id": "01"},
            "player": {"map_id": "01", "tile": [0, 0], "facing": "south"},
            "menus": menus,
            "tiles": [
                {"map_id": "01", "x": 0, "y": 0, "passable": True, "terrain": "path", "special": ""}
            ],
        }
    }


def test_bridge_builds_request_and_entry_telemetry():
    memory = OverworldMemory()
    bridge = EncounterBridge()
    observation = _make_observation()
    request, entry = bridge.build_request(
        observation=observation,
        memory=memory,
        mode="escape_first",
        timeout_steps=120,
        rng_state=b"seed",
    )
    assert isinstance(request, EncounterRequest)
    assert entry["phase"] == "battle.entry"
    assert entry["overworld"]["menu_open"] is False


def test_bridge_detects_invariant_violation():
    memory = OverworldMemory()
    bridge = EncounterBridge()
    observation = _make_observation()
    request, _ = bridge.build_request(
        observation=observation,
        memory=memory,
        mode="escape_first",
        timeout_steps=120,
        rng_state=b"seed",
    )
    exit_snapshot = OverworldSnapshot(
        rng_state=request.snapshot_overworld.rng_state,
        map_id=request.snapshot_overworld.map_id,
        tile_xy=request.snapshot_overworld.tile_xy,
        facing=request.snapshot_overworld.facing,
        menu_open=True,
        party_summary=dict(request.snapshot_overworld.party_summary),
        graph_snapshot=request.snapshot_overworld.graph_snapshot,
    )
    result = EncounterResult(
        status="ESCAPED",
        turns=3,
        reason=None,
        snapshot_return=exit_snapshot,
    )
    invariants, telemetry = bridge.complete(
        result=result,
        observation=observation,
        memory=memory,
    )
    assert invariants["menu_closed"] is False
    assert telemetry["phase"] == "battle.exit"


def test_bridge_fail_generates_exit_event():
    bridge = EncounterBridge()
    event = bridge.fail(reason="timeout")
    assert event["phase"] == "battle.exit"
    assert event["result"]["status"] == "ERROR"
