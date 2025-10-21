"""
Helpers for capturing and restoring overworld snapshots during encounters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from src.pkmn_battle.graph.memory import GraphMemory
from src.pkmn_battle.graph.snapshot import GraphSnapshot
from src.pkmn_battle.graph.schema import Edge, Node, WriteOp


@dataclass
class OverworldSnapshot:
    """
    Serializable snapshot of overworld state prior to a battle.
    """

    rng_state: bytes
    map_id: Optional[str]
    tile_xy: Optional[Tuple[int, int]]
    facing: Optional[str]
    menu_open: bool
    party_summary: Dict[str, object] = field(default_factory=dict)
    graph_snapshot: GraphSnapshot = field(default_factory=lambda: GraphSnapshot(nodes=[], edges=[]))


def build_snapshot(
    observation: Mapping[str, object],
    memory: GraphMemory,
    *,
    rng_state: bytes = b"",
    party_summary: Optional[Dict[str, object]] = None,
) -> OverworldSnapshot:
    data = observation.get("overworld") if isinstance(observation, Mapping) else None
    if not isinstance(data, Mapping):
        data = {}
    player = data.get("player")
    if not isinstance(player, Mapping):
        player = {}

    tile = _coerce_tile(player.get("tile"))
    facing = player.get("facing")
    map_info = data.get("map")
    map_id = map_info.get("id") if isinstance(map_info, Mapping) else player.get("map_id")
    menu_open = any(_menu_open(menu) for menu in data.get("menus", []))

    return OverworldSnapshot(
        rng_state=rng_state,
        map_id=str(map_id) if map_id is not None else None,
        tile_xy=tile,
        facing=str(facing) if facing is not None else None,
        menu_open=menu_open,
        party_summary=dict(party_summary or {}),
        graph_snapshot=memory.snapshot(),
    )


def apply_snapshot(memory: GraphMemory, snapshot: OverworldSnapshot) -> None:
    """
    Apply the provided graph snapshot onto the supplied memory.
    """

    for node in snapshot.graph_snapshot.nodes:
        memory.write(
            WriteOp(
                kind="node",
                payload=Node(
                    type=str(node.get("type", "")),
                    node_id=str(node.get("node_id", "")),
                    attributes=dict(node.get("attributes", {})),
                ),
            )
        )
    for edge in snapshot.graph_snapshot.edges:
        memory.write(
            WriteOp(
                kind="edge",
                payload=Edge(
                    relation=str(edge.get("relation", "")),
                    src=str(edge.get("src", "")),
                    dst=str(edge.get("dst", "")),
                    attributes=dict(edge.get("attributes", {})),
                ),
            )
        )


def _coerce_tile(tile: object) -> Optional[Tuple[int, int]]:
    if isinstance(tile, Sequence) and len(tile) >= 2:
        try:
            return int(tile[0]), int(tile[1])
        except (TypeError, ValueError):
            return None
    return None


def _menu_open(menu: object) -> bool:
    if not isinstance(menu, Mapping):
        return False
    open_flag = menu.get("open")
    if isinstance(open_flag, bool):
        return open_flag
    if isinstance(open_flag, (int, float)):
        return bool(open_flag)
    return False
