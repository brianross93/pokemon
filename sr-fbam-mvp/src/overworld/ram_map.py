"""
RAM layout for key overworld signals used by the extractor.
"""

from __future__ import annotations

from typing import Dict, Mapping

# Addresses sourced from documented Pokemon Red/Blue RAM maps.
DEFAULT_OVERWORLD_RAM_MAP: Mapping[str, int] = {
    "map_id": 0xD35E,
    "map_bank": 0xD35F,
    "player_facing": 0xD360,
    "player_x": 0xD361,
    "player_y": 0xD362,
    "menu_state": 0xCC3C,
    "warp_count": 0xD31E,
    "warp_table": 0xD31F,  # 4 bytes per entry: y, x, dest_warp_id, dest_map
    "npc_count": 0xD2F4,
    "npc_table": 0xD2F5,  # 4 bytes per entry: y, x, movement/sprite, text_id
}

FACING_LOOKUP: Dict[int, str] = {
    0x00: "south",
    0x04: "north",
    0x08: "west",
    0x0C: "east",
}


def decode_facing(raw_value: int) -> str:
    """Translate byte stored at ``player_facing`` into a human-readable orientation."""

    return FACING_LOOKUP.get(raw_value & 0x0C, "south")
