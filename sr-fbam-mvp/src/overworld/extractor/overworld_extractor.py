"""
Prototype overworld extractor that emits nodes and edges consumable by SR-FBAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np

from src.overworld.env.overworld_adapter import OverworldObservation
from src.pkmn_battle.graph.schema import Edge, Node, WriteOp
from ..ram_map import DEFAULT_OVERWORLD_RAM_MAP, decode_facing


@dataclass(frozen=True)
class OverworldTile:
    node_id: str
    map_id: str
    x: int
    y: int
    passable: bool
    terrain: str
    special: str


class OverworldExtractor:
    """
    Converts structured overworld observations into graph write operations.

    The current implementation expects a partially parsed observation dictionary
    (e.g. as produced by a RAM decoder). It focuses on map, tile, warp, NPC,
    inventory, and flag entities so that downstream skills can rehearse queries.
    """

    def __init__(self, *, ram_map: Mapping[str, int] = DEFAULT_OVERWORLD_RAM_MAP) -> None:
        self._ram_map = dict(ram_map)
        self._last_payload: Optional[Mapping[str, object]] = None

    # ------------------------------------------------------------------ #
    # Observation normalisation
    # ------------------------------------------------------------------ #

    def _normalise_observation(
        self, observation: Union[OverworldObservation, Mapping[str, object]]
    ) -> Mapping[str, object]:
        if isinstance(observation, OverworldObservation):
            visual = self._decode_visual(observation)
            payload: Dict[str, object] = {
                "frame": {
                    "hash": observation.frame_hash(),
                    "shape": tuple(int(dim) for dim in observation.framebuffer.shape[:2]),
                    "metadata": dict(observation.metadata),
                },
                "overworld": visual,
                "ram": dict(observation.ram) if isinstance(observation.ram, Mapping) else observation.ram,
            }
            if observation.ram:
                decoded = self._decode_ram_snapshot(observation.ram)
                payload["overworld"] = self._merge_overworld(decoded, visual)
            return payload
        if isinstance(observation, Mapping):
            return observation
        raise TypeError(f"Unsupported observation type: {type(observation)!r}")

    def _decode_visual(self, observation: OverworldObservation) -> Dict[str, object]:
        metadata = observation.metadata or {}
        analysis = self._analyse_frame(observation)
        raw_state = analysis.get("frame_state")
        map_id = "unknown"
        map_name = f"Map_{map_id}"
        player_tile = [0, 0]
        facing = "south"
        menus = analysis.get("menus", [])
        tiles: List[Dict[str, object]] = []
        npcs: List[Dict[str, object]] = []
        warps: List[Dict[str, object]] = []
        entities: List[Dict[str, object]] = []
        overworld = {
            "map": {"id": map_id, "name": map_name},
            "player": {"map_id": map_id, "tile": list(player_tile), "facing": facing},
            "tiles": self._coerce_list_of_mappings(tiles),
            "warps": self._coerce_list_of_mappings(warps),
            "npcs": self._coerce_list_of_mappings(npcs),
            "menus": menus,
            "entities": self._coerce_list_of_mappings(entities),
            "frame": {
                "hash": observation.frame_hash(),
                "shape": tuple(int(dim) for dim in observation.framebuffer.shape[:2]),
                "state": raw_state,
            },
        }
        if analysis.get("dialog") is not None:
            overworld["dialog"] = analysis["dialog"]
        if analysis.get("highlights"):
            overworld["highlights"] = analysis["highlights"]
        return overworld

    @staticmethod
    def _coerce_overworld_mapping(hint: Mapping[str, object]) -> Dict[str, object]:
        return {
            key: (dict(value) if isinstance(value, Mapping) else value)
            if not isinstance(value, list)
            else [dict(item) if isinstance(item, Mapping) else item for item in value]
            for key, value in dict(hint).items()
        }

    @staticmethod
    def _coerce_list_of_mappings(items: object) -> List[Dict[str, object]]:
        result: List[Dict[str, object]] = []
        if not isinstance(items, Iterable):
            return result
        for item in items:
            if isinstance(item, Mapping):
                result.append(dict(item))
        return result

    def extract(self, observation: Union[OverworldObservation, Mapping[str, object]]) -> List[WriteOp]:
        payload = self._normalise_observation(observation)
        if not isinstance(payload, Mapping):
            return []
        data = payload.get("overworld")
        if not isinstance(data, Mapping):
            return []

        # Legacy payloads may still rely on explicit RAM decode.
        if not isinstance(observation, OverworldObservation):
            ram_snapshot = payload.get("ram")
            if ram_snapshot is not None:
                decoded = self._decode_ram_snapshot(ram_snapshot)
                data = self._merge_overworld(decoded, data)

        writes: List[WriteOp] = []
        tile_records: Dict[Tuple[str, int, int], OverworldTile] = {}

        map_info = data.get("map")
        map_node_id = None
        if isinstance(map_info, Mapping):
            map_id = str(map_info.get("id", "unknown"))
            map_name = map_info.get("name", map_id)
            map_node_id = f"map:{map_id}"
            map_node = Node(
                type="MapRegion",
                node_id=map_node_id,
                attributes={"map_id": map_id, "name": map_name},
            )
            writes.append(WriteOp(kind="node", payload=map_node))

        tiles = data.get("tiles", [])
        if isinstance(tiles, Iterable):
            for tile in tiles:
                if not isinstance(tile, Mapping):
                    continue
                tile_map = str(tile.get("map_id", map_info.get("id") if isinstance(map_info, Mapping) else "unknown"))
                x = int(tile.get("x", 0))
                y = int(tile.get("y", 0))
                passable = bool(tile.get("passable", True))
                terrain = str(tile.get("terrain", "unknown"))
                special = str(tile.get("special", ""))
                node_id = f"tile:{tile_map}:{x}:{y}"
                tile_records[(tile_map, x, y)] = OverworldTile(
                    node_id=node_id,
                    map_id=tile_map,
                    x=x,
                    y=y,
                    passable=passable,
                    terrain=terrain,
                    special=special,
                )
                tile_node = Node(
                    type="Tile",
                    node_id=node_id,
                    attributes={
                        "map_id": tile_map,
                        "x": x,
                        "y": y,
                        "passable": passable,
                        "terrain": terrain,
                        "special": special,
                    },
                )
                writes.append(WriteOp(kind="node", payload=tile_node))
                if map_node_id is not None:
                    writes.append(
                        WriteOp(
                            kind="edge",
                            payload=Edge(
                                relation="contains",
                                src=map_node_id,
                                dst=node_id,
                                attributes={},
                            ),
                        )
                    )

        self._emit_adjacent_edges(tile_records, writes)

        warps = data.get("warps", [])
        if isinstance(warps, Iterable):
            for warp in warps:
                if not isinstance(warp, Mapping):
                    continue
                warp_id = str(warp.get("id"))
                if not warp_id:
                    continue
                src_tile = warp.get("src_tile") or warp.get("tile")
                if not isinstance(src_tile, Iterable):
                    continue
                src_coords = tuple(int(v) for v in src_tile[:2])
                dst_map = str(warp.get("dst_map_id", "unknown"))
                dst_tile = warp.get("dst_tile")
                dst_coords = tuple(int(v) for v in (dst_tile[:2] if isinstance(dst_tile, Iterable) else (0, 0)))
                node_id = f"warp:{warp_id}"
                warp_node = Node(
                    type="Warp",
                    node_id=node_id,
                    attributes={
                        "src_map": warp.get("src_map_id", map_info.get("id") if isinstance(map_info, Mapping) else "unknown"),
                        "src_tile": list(src_coords),
                        "dst_map": dst_map,
                        "dst_tile": list(dst_coords),
                    },
                )
                writes.append(WriteOp(kind="node", payload=warp_node))
                src_tile_id = tile_records.get((warp_node.attributes["src_map"], src_coords[0], src_coords[1]))
                if src_tile_id:
                    writes.append(
                        WriteOp(
                            kind="edge",
                            payload=Edge(
                                relation="located_at",
                                src=node_id,
                                dst=src_tile_id.node_id,
                                attributes={},
                            ),
                        )
                    )
                dst_tile_id = tile_records.get((dst_map, dst_coords[0], dst_coords[1]))
                if dst_tile_id:
                    writes.append(
                        WriteOp(
                            kind="edge",
                            payload=Edge(
                                relation="warp_to",
                                src=node_id,
                                dst=dst_tile_id.node_id,
                                attributes={},
                            ),
                        )
                    )
                if src_tile_id and dst_tile_id:
                    writes.append(
                        WriteOp(
                            kind="edge",
                            payload=Edge(
                                relation="warp_exit",
                                src=src_tile_id.node_id,
                                dst=dst_tile_id.node_id,
                                attributes={"warp_id": warp_id},
                            ),
                        )
                    )

        player = data.get("player")
        if isinstance(player, Mapping):
            map_id = str(player.get("map_id", map_info.get("id") if isinstance(map_info, Mapping) else "unknown"))
            tile = player.get("tile")
            if isinstance(tile, Iterable):
                px, py = int(tile[0]), int(tile[1])
            else:
                px = int(player.get("x", 0))
                py = int(player.get("y", 0))
            player_node_id = "player:me"
            player_node = Node(
                type="Player",
                node_id=player_node_id,
                attributes={
                    "map_id": map_id,
                    "x": px,
                    "y": py,
                    "facing": str(player.get("facing", "south")),
                },
            )
            writes.append(WriteOp(kind="node", payload=player_node))
            tile_ref = tile_records.get((map_id, px, py))
            if tile_ref:
                writes.append(
                    WriteOp(
                        kind="edge",
                        payload=Edge(
                            relation="located_at",
                            src=player_node_id,
                            dst=tile_ref.node_id,
                            attributes={},
                        ),
                    )
                )

        npcs = data.get("npcs", [])
        if isinstance(npcs, Iterable):
            for npc in npcs:
                if not isinstance(npc, Mapping):
                    continue
                npc_id = str(npc.get("id"))
                if not npc_id:
                    continue
                node_id = f"npc:{npc_id}"
                map_id = str(npc.get("map_id", map_info.get("id") if isinstance(map_info, Mapping) else "unknown"))
                tile = npc.get("tile") or (npc.get("x"), npc.get("y"))
                tx, ty = 0, 0
                if isinstance(tile, Iterable):
                    coords = list(tile)
                    tx = int(coords[0])
                    ty = int(coords[1])
                npc_node = Node(
                    type="NPC",
                    node_id=node_id,
                    attributes={
                        "name": npc.get("name", npc_id),
                        "map_id": map_id,
                        "x": tx,
                        "y": ty,
                        "role": npc.get("role", "unknown"),
                        "facing": npc.get("facing", "south"),
                    },
                )
                writes.append(WriteOp(kind="node", payload=npc_node))
                tile_ref = tile_records.get((map_id, tx, ty))
                if tile_ref:
                    writes.append(
                        WriteOp(
                            kind="edge",
                            payload=Edge(
                                relation="located_at",
                                src=node_id,
                                dst=tile_ref.node_id,
                                attributes={},
                            ),
                        )
                    )

        inventory = data.get("inventory", [])
        if isinstance(inventory, Iterable):
            for item in inventory:
                if not isinstance(item, Mapping):
                    continue
                item_id = str(item.get("id"))
                if not item_id:
                    continue
                node_id = f"inventory:{item_id}"
                qty = int(item.get("qty", 0))
                item_node = Node(
                    type="InventoryItem",
                    node_id=node_id,
                    attributes={
                        "item_id": item_id,
                        "name": item.get("name", item_id),
                        "quantity": qty,
                    },
                )
                writes.append(WriteOp(kind="node", payload=item_node))
                writes.append(
                    WriteOp(
                        kind="edge",
                        payload=Edge(
                            relation="owns",
                            src="player:me",
                            dst=node_id,
                            attributes={"quantity": qty},
                        ),
                    )
                )

        flags = data.get("flags", [])
        if isinstance(flags, Iterable):
            for flag in flags:
                if isinstance(flag, Mapping):
                    flag_id = str(flag.get("id") or flag.get("name"))
                    if not flag_id:
                        continue
                    node_id = f"flag:{flag_id}"
                    flag_node = Node(
                        type="Flag",
                        node_id=node_id,
                        attributes={
                            "name": flag.get("name", flag_id),
                            "value": flag.get("value"),
                        },
                    )
                    writes.append(WriteOp(kind="node", payload=flag_node))

        menus = data.get("menus", [])
        if isinstance(menus, Iterable):
            for menu in menus:
                if not isinstance(menu, Mapping):
                    continue
                menu_id = str(menu.get("id") or "_".join(menu.get("path", [])))
                if not menu_id:
                    continue
                node_id = f"menu:{menu_id}"
                menu_node = Node(
                    type="MenuState",
                    node_id=node_id,
                    attributes={
                        "path": list(menu.get("path", [])),
                        "open": bool(menu.get("open", False)),
                    },
                )
                writes.append(WriteOp(kind="node", payload=menu_node))

        self._last_payload = payload
        return writes

    def _analyse_frame(self, observation: OverworldObservation) -> Dict[str, object]:
        framebuffer = observation.framebuffer.astype(np.float32)
        gray = framebuffer.mean(axis=2)
        h, w = gray.shape
        bottom = gray[int(h * 0.75) :, :]
        mid = gray[int(h * 0.45) : int(h * 0.7), :]
        left = gray[:, : int(w * 0.1)]
        right = gray[:, -int(w * 0.1) :]

        dialog_brightness = bottom.mean()
        dialog_variance = bottom.std()
        hud_brightness = np.concatenate([left, right], axis=1).mean()

        dialog_open = dialog_brightness < 60 and dialog_variance < 50
        menu_overlay = hud_brightness < 120 or mid.std() > 40

        raw_hash = observation.metadata.get("raw_frame_hash") or observation.frame_hash()
        menus: List[Dict[str, object]] = []

        if dialog_open:
            menus.append(
                {
                    "id": f"dialog:{raw_hash}",
                    "path": ["DIALOG"],
                    "open": True,
                    "state": raw_hash,
                }
            )
            dialog = {
                "id": f"dialog:{raw_hash}",
                "state": raw_hash,
                "hash": raw_hash,
            }
        elif menu_overlay:
            menus.append(
                {
                    "id": f"overlay:{raw_hash}",
                    "path": ["OVERLAY"],
                    "open": True,
                    "state": raw_hash,
                }
            )
            dialog = None
        else:
            menus.append(
                {
                    "id": f"screen:{raw_hash}",
                    "path": ["SCREEN"],
                    "open": False,
                    "state": raw_hash,
                }
            )
            dialog = None

        highlights: List[Dict[str, object]] = []
        if dialog_open:
            column_energy = bottom.mean(axis=0)
            cursor_col = int(np.argmax(column_energy))
            highlights.append(
                {
                    "kind": "dialog_cursor",
                    "column": cursor_col,
                    "state": raw_hash,
                }
            )

        return {
            "frame_state": raw_hash,
            "menus": menus,
            "dialog": dialog,
            "highlights": highlights or None,
        }

    @property
    def last_payload(self) -> Optional[Mapping[str, object]]:
        return self._last_payload

    @staticmethod
    def _emit_adjacent_edges(
        tiles: Mapping[Tuple[str, int, int], OverworldTile],
        writes: List[WriteOp],
    ) -> None:
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for tile in tiles.values():
            for dx, dy in offsets:
                neighbor = tiles.get((tile.map_id, tile.x + dx, tile.y + dy))
                if neighbor is None:
                    continue
                attributes = {"passable": tile.passable and neighbor.passable}
                writes.append(
                    WriteOp(
                        kind="edge",
                        payload=Edge(
                            relation="adjacent",
                            src=tile.node_id,
                            dst=neighbor.node_id,
                            attributes=attributes,
                        ),
                    )
                )

    def _decode_ram_snapshot(self, ram_snapshot: Union[Mapping[int, int], bytes, bytearray]) -> Mapping[str, object]:
        """
        Translate a RAM snapshot into a minimal overworld dictionary.

        Args:
            ram_snapshot: Either a mapping of address→value or a byte-like object.
        """

        def read_byte(address: int) -> int:
            if isinstance(ram_snapshot, Mapping):
                return int(ram_snapshot.get(address, 0))
            if isinstance(ram_snapshot, (bytes, bytearray)):
                if address < 0 or address >= len(ram_snapshot):
                    return 0
                return int(ram_snapshot[address])
            return 0

        map_id = read_byte(self._ram_map["map_id"])
        map_hex = f"{map_id:02X}"
        player_x = read_byte(self._ram_map["player_x"])
        player_y = read_byte(self._ram_map["player_y"])
        facing_raw = read_byte(self._ram_map["player_facing"])
        facing = decode_facing(facing_raw)
        menu_state = read_byte(self._ram_map["menu_state"])

        tiles_map: Dict[Tuple[str, int, int], Dict[str, object]] = {}

        def ensure_tile(map_key: str, x_coord: int, y_coord: int) -> None:
            key = (map_key, x_coord, y_coord)
            if key not in tiles_map:
                tiles_map[key] = {
                    "map_id": map_key,
                    "x": x_coord,
                    "y": y_coord,
                    "passable": True,
                    "terrain": "unknown",
                    "special": "",
                }

        ensure_tile(map_hex, player_x, player_y)

        # Decode warp table
        warps: List[Dict[str, object]] = []
        if "warp_count" in self._ram_map and "warp_table" in self._ram_map:
            warp_count = min(8, read_byte(self._ram_map["warp_count"]))
            warp_base = self._ram_map["warp_table"]
            for index in range(warp_count):
                base = warp_base + index * 4
                raw_y = read_byte(base)
                raw_x = read_byte(base + 1)
                dest_warp_id = read_byte(base + 2)
                dest_map = read_byte(base + 3)
                y = max(0, raw_y - 4)
                x = max(0, raw_x - 4)
                ensure_tile(map_hex, x, y)
                ensure_tile(map_hex, x, y)
                dest_map_hex = f"{dest_map:02X}"
                dest_tile_coords = [0, 0]
                ensure_tile(dest_map_hex, dest_tile_coords[0], dest_tile_coords[1])
                warps.append(
                    {
                        "id": f"{index}",
                        "src_tile": [x, y],
                        "dst_map_id": f"{dest_map:02X}",
                        "dst_tile": [0, 0],
                        "src_map_id": map_hex,
                        "dest_warp_id": dest_warp_id,
                    }
                )

        # Decode NPC table
        npcs: List[Dict[str, object]] = []
        if "npc_count" in self._ram_map and "npc_table" in self._ram_map:
            npc_count = min(16, read_byte(self._ram_map["npc_count"]))
            npc_base = self._ram_map["npc_table"]
            for index in range(npc_count):
                base = npc_base + index * 4
                raw_y = read_byte(base)
                raw_x = read_byte(base + 1)
                sprite = read_byte(base + 2)
                text_id = read_byte(base + 3)
                y = max(0, raw_y - 4)
                x = max(0, raw_x - 4)
                ensure_tile(map_hex, x, y)
                npcs.append(
                    {
                        "id": f"{index}",
                        "name": f"NPC_{index:02d}",
                        "map_id": map_hex,
                        "tile": [x, y],
                        "role": f"sprite_{sprite:02X}",
                        "facing": "south",
                        "text_id": text_id,
                    }
                )

        overworld: Dict[str, object] = {
            "map": {"id": map_hex, "name": f"Map_{map_hex}"},
            "player": {
                "map_id": map_hex,
                "tile": [player_x, player_y],
                "facing": facing,
            },
            "tiles": list(tiles_map.values()),
            "warps": warps,
            "npcs": npcs,
            "menus": [
                {
                    "id": "main_menu",
                    "path": ["MAIN"],
                    "open": bool(menu_state),
                }
            ],
        }
        return overworld

    @staticmethod
    def _merge_overworld(decoded: Mapping[str, object], existing: Mapping[str, object]) -> Mapping[str, object]:
        if not existing:
            return dict(decoded)
        merged = dict(decoded)
        for key, value in existing.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            elif key in merged and isinstance(merged[key], list) and isinstance(value, Iterable):
                merged_list = list(merged[key])
                merged_list.extend(item for item in value if item not in merged_list)
                merged[key] = merged_list
            else:
                merged[key] = value
        return merged
