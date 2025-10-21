#!/usr/bin/env python3
"""
Run a minimal SR-FBAM overworld demo that walks a straight corridor with no LLM.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

import sys
from pathlib import Path

# Ensure repository root on sys.path for src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.overworld.skills.base import SkillStatus
from src.overworld.skills.navigate import NavigateSkill
from src.pkmn_overworld.world_graph import WorldGraph


_DELTA = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def _coords_from_tile(tile_id: str) -> Tuple[int, int]:
    _, _, x_str, y_str = tile_id.split(":")
    return int(x_str), int(y_str)


def build_corridor_world(length: int = 4) -> Tuple[WorldGraph, str, str]:
    world = WorldGraph()
    start_tile, target_tile = world.ensure_corridor("corridor", length)
    start_x, start_y = _coords_from_tile(start_tile)
    world.set_player_position("corridor", start_x, start_y)
    return world, start_tile, target_tile


def run_corridor_demo(length: int = 4) -> Tuple[list[Dict[str, object]], SkillStatus]:
    world, start_tile, target_tile = build_corridor_world(length)
    target_coords = _coords_from_tile(target_tile)

    planlet = SimpleNamespace(args={"target": {"tile": target_coords}})
    navigate = NavigateSkill()
    navigate.on_enter(planlet, world.memory)
    navigate.update_context(memory=world.memory)

    actions: list[Dict[str, object]] = []
    for _ in range(32):
        action = navigate.select_action({}, world.memory)
        actions.append(action)

        progress = navigate.progress(world.memory)
        if progress.status == SkillStatus.SUCCEEDED:
            break

        if action.get("kind") == "button":
            label = action.get("label", "").upper()
            dx, dy = _DELTA.get(label, (0, 0))
            player = world.memory.assoc_player()
            if player is None:
                break
            x = int(player.attributes.get("x", 0)) + dx
            y = int(player.attributes.get("y", 0)) + dy
            map_id = player.attributes.get("map_id", "corridor")
            world.set_player_position(str(map_id), x, y)

    final_status = navigate.progress(world.memory).status
    return actions, final_status


def main() -> int:
    actions, status = run_corridor_demo()
    print("Corridor actions:")
    for idx, action in enumerate(actions):
        print(f"{idx:02d}: {action}")
    print(f"Final status: {status.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
