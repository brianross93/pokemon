from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from src.overworld.graph.overworld_memory import OverworldMemory
from src.pkmn_battle.graph.schema import Edge, Node, WriteOp


def _tile_node_id(map_id: str, x: int, y: int) -> str:
    return f"tile:{map_id}:{int(x)}:{int(y)}"


@dataclass(frozen=True)
class TileSpec:
    map_id: str
    x: int
    y: int
    terrain: Optional[str] = None


class WorldGraph:
    """
    Lightweight helper around :class:`OverworldMemory` for building symbolic maps.

    The graph stores tile nodes and higher-level entities (locations, NPCs, shops, etc.)
    using SR-FBAM's :class:`GraphMemory` primitives so they can be consumed directly by
    skills such as :class:`NavigateSkill`.
    """

    def __init__(self, memory: Optional[OverworldMemory] = None) -> None:
        self._memory = memory or OverworldMemory()
        self._tiles: Dict[str, TileSpec] = {}

    @property
    def memory(self) -> OverworldMemory:
        return self._memory

    # ------------------------------------------------------------------ #
    # Node helpers
    # ------------------------------------------------------------------ #

    def add_tile(self, map_id: str, x: int, y: int, *, terrain: Optional[str] = None) -> str:
        node_id = _tile_node_id(map_id, x, y)
        spec = TileSpec(map_id=map_id, x=x, y=y, terrain=terrain)
        self._tiles[node_id] = spec
        attributes = {"map_id": map_id, "x": int(x), "y": int(y)}
        if terrain:
            attributes["terrain"] = terrain
        self._write_node("Tile", node_id, attributes)
        return node_id

    def add_location(self, location_id: str, *, name: Optional[str] = None, tile: Optional[str] = None) -> str:
        attributes = {"name": name} if name else {}
        attributes["tile"] = tile
        node_id = f"location:{location_id}"
        self._write_node("Location", node_id, attributes)
        if tile:
            self._write_edge(node_id, "located_at", tile)
        return node_id

    def add_door(self, door_id: str, *, src_tile: str, dst_tile: str) -> str:
        node_id = f"door:{door_id}"
        self._write_node("Door", node_id, {"src": src_tile, "dst": dst_tile})
        self._write_edge(node_id, "door_from", src_tile)
        self._write_edge(node_id, "door_to", dst_tile)
        return node_id

    def add_npc(self, npc_id: str, *, name: Optional[str] = None, tile: Optional[str] = None) -> str:
        attributes = {"name": name} if name else {}
        if tile:
            attributes["tile"] = tile
        node_id = f"npc:{npc_id}"
        self._write_node("NPC", node_id, attributes)
        if tile:
            self._write_edge(node_id, "occupies", tile)
        return node_id

    def add_shop(self, shop_id: str, *, name: Optional[str] = None, tile: Optional[str] = None) -> str:
        attributes = {"name": name} if name else {}
        if tile:
            attributes["tile"] = tile
        node_id = f"shop:{shop_id}"
        self._write_node("Shop", node_id, attributes)
        if tile:
            self._write_edge(node_id, "located_at", tile)
        return node_id

    def add_center(self, center_id: str, *, tile: Optional[str] = None) -> str:
        node_id = f"center:{center_id}"
        attributes = {"tile": tile} if tile else {}
        self._write_node("Center", node_id, attributes)
        if tile:
            self._write_edge(node_id, "located_at", tile)
        return node_id

    def add_item(self, item_id: str, *, name: str, tile: Optional[str] = None) -> str:
        node_id = f"item:{item_id}"
        attributes = {"name": name}
        if tile:
            attributes["tile"] = tile
        self._write_node("Item", node_id, attributes)
        if tile:
            self._write_edge(node_id, "located_at", tile)
        return node_id

    def add_trigger(self, trigger_id: str, *, kind: str, tile: Optional[str] = None) -> str:
        node_id = f"trigger:{trigger_id}"
        attributes = {"kind": kind}
        if tile:
            attributes["tile"] = tile
        self._write_node("Trigger", node_id, attributes)
        if tile:
            self._write_edge(node_id, "located_at", tile)
        return node_id

    def add_hazard(self, hazard_id: str, *, kind: str, tile: Optional[str] = None) -> str:
        node_id = f"hazard:{hazard_id}"
        attributes = {"kind": kind}
        if tile:
            attributes["tile"] = tile
        self._write_node("Hazard", node_id, attributes)
        if tile:
            self._write_edge(node_id, "located_at", tile)
        return node_id

    def set_player_position(self, map_id: str, x: int, y: int, *, facing: str = "south") -> None:
        node = Node(
            type="Player",
            node_id="player",
            attributes={"map_id": map_id, "x": int(x), "y": int(y), "facing": facing},
        )
        self._memory.write(WriteOp(kind="node", payload=node))

    # ------------------------------------------------------------------ #
    # Edge helpers
    # ------------------------------------------------------------------ #

    def connect_adjacent(self, tile_a: str, tile_b: str) -> None:
        self._write_edge(tile_a, "adjacent", tile_b)
        self._write_edge(tile_b, "adjacent", tile_a)

    def add_warp(self, src_tile: str, dst_tile: str, *, door_id: Optional[str] = None) -> None:
        relation = "warp_to"
        self._write_edge(src_tile, relation, dst_tile)
        if door_id:
            door_node = f"door:{door_id}"
            self._write_edge(door_node, relation, dst_tile)

    def add_requirement(self, src_node: str, *, requirement: str, value: str) -> None:
        node_id = f"requirement:{requirement}:{value}"
        self._write_node("Requirement", node_id, {"requirement": requirement, "value": value})
        self._write_edge(src_node, "requires", node_id)

    def add_shop_inventory(self, shop_node: str, *, item: str, price: int) -> None:
        stock_id = f"stock:{shop_node}:{item}"
        self._write_node("InventoryItem", stock_id, {"item": item, "price": int(price)})
        self._write_edge(shop_node, "sells", stock_id)

    # ------------------------------------------------------------------ #
    # Corridor helpers
    # ------------------------------------------------------------------ #

    def ensure_corridor(self, map_id: str, length: int) -> Tuple[str, str]:
        """
        Build a simple straight corridor of `length` tiles on the given map.

        Returns `(start_tile_id, end_tile_id)`.
        """

        if length < 2:
            raise ValueError("Corridor length must be >= 2")

        tile_ids = [
            self.add_tile(map_id, x, 0, terrain="path")
            for x in range(length)
        ]
        for left, right in zip(tile_ids[:-1], tile_ids[1:]):
            self.connect_adjacent(left, right)
        return tile_ids[0], tile_ids[-1]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _write_node(self, type_: str, node_id: str, attributes: Optional[Dict[str, object]] = None) -> None:
        node = Node(type=type_, node_id=node_id, attributes=attributes or {})
        self._memory.write(WriteOp(kind="node", payload=node))

    def _write_edge(
        self,
        src: str,
        relation: str,
        dst: str,
        *,
        attributes: Optional[Dict[str, object]] = None,
    ) -> None:
        edge = Edge(relation=relation, src=src, dst=dst, attributes=attributes or {})
        self._memory.write(WriteOp(kind="edge", payload=edge))
