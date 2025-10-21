from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


DEFAULT_STATIC_PATH = Path(__file__).resolve().parents[2] / "data" / "overworld" / "static_entities.json"


class WorldGraph:
    """
    Lightweight helper around :class:`OverworldMemory` for building symbolic maps.

    The graph stores tile nodes and higher-level entities (locations, NPCs, shops, etc.)
    using SR-FBAM's :class:`GraphMemory` primitives so they can be consumed directly by
    skills such as :class:`NavigateSkill`.
    """

    _STATIC_CACHE: Optional[Dict[str, object]] = None

    def __init__(
        self,
        memory: Optional[OverworldMemory] = None,
        *,
        load_static: bool = True,
        static_path: Optional[Path] = None,
    ) -> None:
        self._memory = memory or OverworldMemory()
        self._tiles: Dict[str, TileSpec] = {}
        self._static_loaded = False
        self._static_maps: Dict[str, Dict[str, object]] = {}

        if load_static:
            self._load_static_data(static_path or DEFAULT_STATIC_PATH)

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
        if tile:
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

    def add_warp_hint(
        self,
        map_id: str,
        x: int,
        y: int,
        *,
        destination_map: str,
        destination_warp: int,
    ) -> str:
        node_id = f"warp_hint:{map_id}:{int(x)}:{int(y)}"
        attributes = {
            "map_id": map_id,
            "x": int(x),
            "y": int(y),
            "destination_map": destination_map,
            "destination_warp": int(destination_warp),
        }
        self._write_node("WarpHint", node_id, attributes)
        self._write_edge(node_id, "leads_to_map", f"location:{destination_map}")
        self._write_edge(f"location:{map_id}", "has_warp", node_id)
        tile_id = self.add_tile(map_id, x, y, terrain="warp")
        self._write_edge(tile_id, "warp_hint", node_id)
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

    # ------------------------------------------------------------------ #
    # Static data loading
    # ------------------------------------------------------------------ #

    def _load_static_data(self, path: Path) -> None:
        if self._static_loaded:
            return
        if not path.exists():
            return
        if WorldGraph._STATIC_CACHE is None:
            WorldGraph._STATIC_CACHE = json.loads(path.read_text(encoding="utf-8"))
        data = WorldGraph._STATIC_CACHE or {}
        maps = data.get("maps", {})
        if not isinstance(maps, dict):
            return
        self._ingest_static_maps(maps)
        self._static_maps = maps  # type: ignore[assignment]
        self._static_loaded = True

    def _ingest_static_maps(self, maps: Dict[str, Dict[str, object]]) -> None:
        # First pass: create location nodes for every map.
        for map_key, payload in maps.items():
            map_const = str(payload.get("constant") or map_key)
            display_name = payload.get("label") or map_const
            self.add_location(map_const, name=str(display_name))

        # Second pass: connections, warps, NPCs, and background events.
        for map_key, payload in maps.items():
            map_const = str(payload.get("constant") or map_key)
            location_node = f"location:{map_const}"

            # Connections
            connections = payload.get("connections") or []
            if isinstance(connections, list):
                for conn in connections:
                    if not isinstance(conn, dict):
                        continue
                    target_const = str(conn.get("target_const") or conn.get("target_label") or "")
                    if not target_const:
                        continue
                    self._write_edge(location_node, "connected_to", f"location:{target_const}")

            # Warps
            warps = payload.get("warps") or []
            if isinstance(warps, list):
                for warp in warps:
                    if not isinstance(warp, dict):
                        continue
                    try:
                        x = int(warp.get("x", 0))
                        y = int(warp.get("y", 0))
                        destination_map = str(warp.get("destination_map") or "")
                        destination_warp = int(warp.get("destination_warp", 0))
                    except (TypeError, ValueError):
                        continue
                    if not destination_map:
                        continue
                    self.add_warp_hint(map_const, x, y, destination_map=destination_map, destination_warp=destination_warp)

            # NPCs
            npcs = payload.get("npcs") or []
            if isinstance(npcs, list):
                for npc in npcs:
                    if not isinstance(npc, dict):
                        continue
                    try:
                        x = int(npc.get("x", 0))
                        y = int(npc.get("y", 0))
                    except (TypeError, ValueError):
                        continue
                    tile_id = self.add_tile(map_const, x, y)
                    sprite = npc.get("sprite")
                    script = npc.get("script") or npc.get("name") or f"{map_const}_{x}_{y}"
                    npc_id = f"{map_const}:{script}"
                    self.add_npc(npc_id, name=str(sprite) if sprite else None, tile=tile_id)

            # Background events (signs/triggers)
            bg_events = payload.get("background_events") or []
            if isinstance(bg_events, list):
                for bg in bg_events:
                    if not isinstance(bg, dict):
                        continue
                    try:
                        x = int(bg.get("x", 0))
                        y = int(bg.get("y", 0))
                    except (TypeError, ValueError):
                        continue
                    tile_id = self.add_tile(map_const, x, y)
                    script = str(bg.get("script") or f"{map_const}_bg_{x}_{y}")
                    trigger_id = f"{map_const}:{script}"
                    self.add_trigger(trigger_id, kind="background_event", tile=tile_id)
