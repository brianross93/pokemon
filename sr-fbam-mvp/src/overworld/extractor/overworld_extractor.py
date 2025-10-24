"""
Prototype overworld extractor that emits nodes and edges consumable by SR-FBAM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

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
        self._last_dialog_hash: Optional[str] = None
        self._naming_cursor_history: Deque[Dict[str, object]] = deque(maxlen=8)

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
                merged = self._merge_overworld(decoded, visual)
                meta = observation.metadata or {}
                screen_hint = None
                if isinstance(meta, Mapping):
                    screen_hint = meta.get("game_area") or meta.get("screen")
                if isinstance(screen_hint, Mapping):
                    self._augment_from_screen(merged, screen_hint)
                payload["overworld"] = merged
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
        screen_hint = None
        manual_hint = None
        if isinstance(metadata, Mapping):
            screen_hint = metadata.get("game_area") or metadata.get("screen")
            manual_hint = metadata.get("visual_overworld")
        if isinstance(manual_hint, Mapping):
            hint_payload = self._coerce_overworld_mapping(manual_hint)
            overworld = self._merge_overworld(overworld, hint_payload)
        dialog_lines: Optional[List[str]] = None
        if analysis.get("dialog") is not None:
            overworld["dialog"] = analysis["dialog"]
            if isinstance(screen_hint, Mapping):
                dialog_lines = self._decode_dialog_text(screen_hint)
                if dialog_lines:
                    overworld["dialog_lines"] = dialog_lines
        if analysis.get("highlights"):
            overworld["highlights"] = analysis["highlights"]
        if isinstance(screen_hint, Mapping):
            self._augment_from_screen(overworld, screen_hint)
        naming = self._decode_naming_screen(overworld, metadata)
        if naming is not None:
            overworld["naming_screen"] = naming
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

    def _detect_overlay_from_metadata(self, metadata: Mapping[str, object]) -> Optional[str]:
        screen = metadata.get("game_area") or metadata.get("screen")
        if not isinstance(screen, Mapping):
            return None
        tile_ids = screen.get("tile_ids")
        if tile_ids is None:
            return None
        tile_grid = np.asarray(tile_ids, dtype=int)
        if tile_grid.ndim != 2 or tile_grid.size == 0:
            return None

        bottom_rows = tile_grid[-4:, :]

        letter_tiles = np.isin(bottom_rows, np.concatenate([
            np.arange(0x80, 0xC0),
            np.arange(0xE0, 0x100),
            np.arange(0x200, 0x240),
        ]))
        letter_hits = int(letter_tiles.sum())

        if letter_hits >= 6:
            border_tiles = np.isin(bottom_rows, [377, 378, 379, 380, 381, 382])
            border_hits = int(border_tiles.sum())
            if border_hits >= 12:
                return "dialog"
            return None

        overlay_tiles = np.isin(bottom_rows, [377, 378, 379, 380, 381, 382])
        overlay_hits = int(overlay_tiles.sum())
        if overlay_hits >= 12:
            return "overlay"

        return None

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

        dialog_lines = data.get("dialog_lines")
        if isinstance(dialog_lines, list) and dialog_lines:
            frame_hash = payload.get("frame", {}).get("hash")
            if isinstance(frame_hash, str) and frame_hash and frame_hash != self._last_dialog_hash:
                dialog_node = Node(
                    type="DialogEvent",
                    node_id=f"dialog:{frame_hash}",
                    attributes={
                        "lines": dialog_lines,
                        "frame_hash": frame_hash,
                    },
                )
                writes.append(WriteOp(kind="node", payload=dialog_node))
                self._last_dialog_hash = frame_hash

        self._last_payload = payload
        return writes

    def _analyse_frame(self, observation: OverworldObservation) -> Dict[str, object]:
        overlay_state = self._detect_overlay_from_metadata(observation.metadata or {})

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

        if overlay_state == "dialog":
            dialog_open = True
            menu_overlay = False
        elif overlay_state == "overlay":
            dialog_open = False
            menu_overlay = True
        else:
            dialog_open = False
            menu_overlay = False

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

    def _merge_overworld(self, decoded: Mapping[str, object], existing: Mapping[str, object]) -> Mapping[str, object]:
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

    def _augment_from_screen(
        self,
        data: MutableMapping[str, object],
        screen: Mapping[str, object],
    ) -> None:
        tile_ids_raw = screen.get("tile_ids")
        collision_raw = screen.get("game_area_collision") or screen.get("collision")
        if tile_ids_raw is None:
            return
        tile_ids = np.asarray(tile_ids_raw, dtype=int)
        if tile_ids.ndim != 2 or tile_ids.size == 0:
            return
        collisions: Optional[np.ndarray]
        if collision_raw is not None:
            collisions = np.asarray(collision_raw)
            if collisions.shape != tile_ids.shape:
                collisions = None
        else:
            collisions = None

        map_info = data.get("map") or {}
        map_id = str(map_info.get("id", "unknown")) if isinstance(map_info, Mapping) else "unknown"

        player_info = data.get("player") or {}
        if isinstance(player_info, Mapping):
            player_tile = player_info.get("tile") or [0, 0]
        else:
            player_tile = [0, 0]
        try:
            player_x = int(player_tile[0])
            player_y = int(player_tile[1])
        except Exception:
            player_x = 0
            player_y = 0

        tile_grid = tile_ids
        height, width = tile_grid.shape
        half_w = width // 2
        half_h = height // 2

        has_absolute = map_id != "unknown" and (player_x != 0 or player_y != 0)
        if has_absolute:
            left = player_x - half_w
            top = player_y - half_h
        else:
            # Fallback to a deterministic local coordinate system.
            if map_id == "unknown":
                map_id = "screen_local"
            left = 0
            top = 0
            player_x = half_w
            player_y = half_h

        map_entry = data.setdefault("map", {"id": map_id, "name": f"Map_{map_id}"})
        if isinstance(map_entry, Mapping):
            map_entry["id"] = map_id
            map_entry["name"] = map_entry.get("name") or f"Map_{map_id}"
        player_info = data.setdefault("player", {"map_id": map_id, "tile": [player_x, player_y], "facing": "south"})
        if isinstance(player_info, Mapping):
            player_info["map_id"] = map_id
            player_info["tile"] = [player_x, player_y]
            player_info["facing"] = player_info.get("facing", "south")

        tiles_list = list(data.get("tiles", [])) if isinstance(data.get("tiles"), list) else []
        tile_lookup: Dict[Tuple[str, int, int], Dict[str, object]] = {}
        for tile in tiles_list:
            if isinstance(tile, Mapping):
                key = (
                    str(tile.get("map_id", map_id)),
                    int(tile.get("x", 0)),
                    int(tile.get("y", 0)),
                )
                tile_lookup[key] = dict(tile)
                if map_id == "unknown":
                    map_id = str(tile.get("map_id", map_id))

        for screen_y in range(height):
            for screen_x in range(width):
                world_x = left + screen_x
                world_y = top + screen_y
                key = (map_id, world_x, world_y)
                entry = tile_lookup.get(key)
                if entry is None:
                    entry = {
                        "map_id": map_id,
                        "x": world_x,
                        "y": world_y,
                        "terrain": "unknown",
                        "special": "",
                    }
                    tile_lookup[key] = entry
                entry["screen"] = {"x": screen_x, "y": screen_y}
                entry["tile_id"] = int(tile_grid[screen_y, screen_x])
                if collisions is not None:
                    value = collisions[screen_y, screen_x]
                    try:
                        passable = int(value) == 0
                    except Exception:
                        passable = not bool(value)
                else:
                    passable = True
                entry["passable"] = passable
                if entry.get("terrain") == "unknown":
                    entry["terrain"] = "floor" if passable else "wall"

        warps = data.get("warps") if isinstance(data.get("warps"), list) else []
        for warp in warps:
            if not isinstance(warp, Mapping):
                continue
            src_tile = warp.get("src_tile") or warp.get("tile")
            if not isinstance(src_tile, Iterable):
                continue
            try:
                sx = int(src_tile[0])
                sy = int(src_tile[1])
            except Exception:
                continue
            key = (map_id, sx, sy)
            entry = tile_lookup.get(key)
            if entry is not None:
                entry["terrain"] = "door"
                entry["special"] = "warp"
                entry["passable"] = True

        tiles = list(tile_lookup.values())
        data["tiles"] = tiles

        # Create adjacency edges for passable tiles
        passable_lookup = { (tile["map_id"], int(tile["x"]), int(tile["y"])): tile for tile in tiles if tile.get("passable", True) }
        adjacency: Dict[Tuple[str, int, int], List[Tuple[str, int, int]]] = {}
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for key, tile in passable_lookup.items():
            _, x, y = key
            neighbors: List[Tuple[str, int, int]] = []
            for dx, dy in offsets:
                neighbor_key = (tile["map_id"], x + dx, y + dy)
                if neighbor_key in passable_lookup:
                    neighbors.append(neighbor_key)
            if neighbors:
                adjacency[key] = neighbors
        data["tile_adjacency"] = adjacency

        data["warp_tiles"] = []
        for warp in warps:
            if not isinstance(warp, Mapping):
                continue
            src_tile = warp.get("src_tile") or warp.get("tile")
            if not isinstance(src_tile, Iterable):
                continue
            try:
                sx = int(src_tile[0])
                sy = int(src_tile[1])
            except Exception:
                continue
            data["warp_tiles"].append({
                "map_id": map_id,
                "tile": [sx, sy],
                "dest_map_id": warp.get("dst_map_id"),
                "dest_tile": warp.get("dst_tile"),
            })

        sprite_entries = []
        sprites = screen.get("sprites") if isinstance(screen.get("sprites"), list) else []
        entities = data.setdefault("entities", [])
        for sprite in sprites:
            if not isinstance(sprite, Mapping):
                continue
            try:
                idx = int(sprite.get("index", 0))
                tile_id = int(sprite.get("tile_id", 0))
                px = int(sprite.get("x", 0))
                py = int(sprite.get("y", 0))
            except Exception:
                continue
            screen_tx = max(0, (px - 8) // 8)
            screen_ty = max(0, (py - 16) // 8)
            world_x = left + screen_tx
            world_y = top + screen_ty
            sprite_entry = {
                "id": f"sprite:{idx}",
                "map_id": map_id,
                "tile": [world_x, world_y],
                "screen": {"x": screen_tx, "y": screen_ty, "px": px, "py": py},
                "tile_id": tile_id,
                "on_screen": bool(sprite.get("on_screen", True)),
                "attributes": dict(sprite.get("attributes", {})),
            }
            sprite_entries.append(sprite_entry)

        if sprite_entries:
            # Merge with any pre-existing entities without duplicating IDs.
            existing_ids = {entity.get("id") for entity in entities if isinstance(entity, Mapping)}
            for entry in sprite_entries:
                if entry["id"] not in existing_ids:
                    entities.append(entry)

    NAMING_TILESET: Dict[int, str] = {}
    for idx, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        NAMING_TILESET[0x80 + idx] = ch
        NAMING_TILESET[0xA0 + idx] = ch.lower()
    NAMING_TILESET.update(
        {
            0x7A: "END",
            0xF6: "0",
            0xF7: "1",
            0xF8: "2",
            0xF9: "3",
            0xFA: "4",
            0xFB: "5",
            0xFC: "6",
            0xFD: "7",
            0xFE: "8",
            0xFF: "9",
            0xE6: ".",
            0xE7: ",",
            0xE8: "'",
            0xE9: "…",
            0xEA: "“",
            0xEB: "”",
            0xEC: "‘",
            0xED: "’",
            0xEE: "☀",
            0xEF: "♪",
            0xF0: "%",
            0xF1: "×",
            0xF2: "÷",
            0xF3: "=",
            0xF4: "+",
            0xF5: "-",
        }
    )

    DIALOG_CHAR_RANGES: Tuple[Tuple[int, int, str], ...] = (
        (0x80, 0x99, "upper"),
        (0xA0, 0xB9, "lower"),
        (0xF6, 0xFF, "digit"),
    )
    DIALOG_PUNCTUATION: Dict[int, str] = {
        0xE6: ".",
        0xE7: ",",
        0xE8: "'",
        0xE9: "…",
        0xEA: "\"",
        0xEB: "\"",
        0xEC: "'",
        0xED: "'",
        0xEE: "♪",
        0xEF: "♪",
        0xF0: "%",
        0xF1: "!",
        0xF2: "?",
        0xF3: "=",
        0xF4: "+",
        0xF5: "-",
        0x72: "-",
        0x73: "?",
        0x74: "!",
        0x75: "♂",
        0x76: "♀",
        0x77: "/",
        0x78: ".",
        0x79: "→",
        0x7A: "END",
        0x232: "&",
        0x233: ":",
    }
    DIALOG_SPACE_TILES = {0, 0xFF, 0x383, 383, 0x380, 380}
    DIALOG_BORDER_TILES = {377, 378, 379, 380, 381, 382, 383}
    def _decode_naming_screen(
        self,
        overworld: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> Optional[Dict[str, object]]:
        menus = overworld.get("menus")
        overlay_present = False
        if isinstance(menus, list):
            for menu in menus:
                if not isinstance(menu, Mapping):
                    continue
                menu_id = str(menu.get("id") or "")
                path = menu.get("path") or []
                if "overlay" in menu_id.lower() or any("overlay" in str(part).lower() for part in path):
                    overlay_present = True
                    break
        else:
            menus = []

        tiles = overworld.get("tiles")
        if not isinstance(tiles, list) or not tiles:
            return None
        tile_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
        for tile in tiles:
            if not isinstance(tile, Mapping):
                continue
            try:
                sx = int(tile.get("screen", {}).get("x"))
                sy = int(tile.get("screen", {}).get("y"))
            except Exception:
                continue
            tile_lookup[(sx, sy)] = dict(tile)

        frame_meta = overworld.get("frame")
        frame_hash = None
        if isinstance(frame_meta, Mapping):
            frame_hash = frame_meta.get("hash")
        # Heuristic for naming grid region: rows ~8-18 (doubling screen coordinate), cols wide middle portion.
        naming_entries: List[Dict[str, object]] = []
        tiles_per_row = 20  # Game Boy tiles horizontally when doubled from 40 px
        tiles_per_col = 18  # vertical
        for sy in range(tiles_per_col):
            for sx in range(tiles_per_row):
                tile = tile_lookup.get((sx, sy))
                if tile is None:
                    continue
                tile_id = tile.get("tile_id")
                if not isinstance(tile_id, int):
                    continue
                glyph = self.NAMING_TILESET.get(tile_id)
                if glyph is None:
                    continue
                world_x = int(tile.get("x", 0))
                world_y = int(tile.get("y", 0))
                entry: Dict[str, object] = {
                    "letter": glyph,
                    "tile_id": tile_id,
                    "screen": {"x": sx, "y": sy},
                    "world": {"x": world_x, "y": world_y},
                }
                if glyph.upper() in {"ED", "END"}:
                    entry["role"] = "END"
                elif glyph.upper() in {"PK", "MN"}:
                    entry["role"] = glyph.upper()
                naming_entries.append(entry)

        if not naming_entries:
            if not overlay_present:
                self._naming_cursor_history.clear()
            return None

        if not overlay_present and len(naming_entries) < 40:
            self._naming_cursor_history.clear()
            return None

        # Normalize rows/columns.
        unique_rows = sorted({entry["screen"]["y"] for entry in naming_entries})
        grid_letters: List[List[str]] = []
        row_entries: List[List[Dict[str, object]]] = []
        for row_idx, sy in enumerate(unique_rows):
            row = [entry for entry in naming_entries if entry["screen"]["y"] == sy]
            row.sort(key=lambda item: item["screen"]["x"])
            for col_idx, entry in enumerate(row):
                entry["grid"] = {"row": row_idx, "col": col_idx}
            row_entries.append(row)
            grid_letters.append([entry["letter"] for entry in row])

        # Attempt to infer cursor by projecting sprite positions.
        cursor: Optional[Dict[str, object]] = None
        viewport_meta = None
        if isinstance(metadata, Mapping):
            viewport_meta = metadata.get("game_area") or metadata.get("screen")
        sprites = viewport_meta.get("sprites") if isinstance(viewport_meta, Mapping) else None
        if isinstance(sprites, list):
            best_entry: Optional[Dict[str, object]] = None
            best_dist = float("inf")
            for sprite in sprites:
                if not isinstance(sprite, Mapping):
                    continue
                if not bool(sprite.get("on_screen", True)):
                    continue
                sprite_screen = sprite.get("screen")
                if not isinstance(sprite_screen, Mapping):
                    continue
                sx = sprite_screen.get("x")
                sy = sprite_screen.get("y")
                if sx is None or sy is None:
                    continue
                try:
                    sx = int(sx)
                    sy = int(sy)
                except Exception:
                    continue
                for entry in naming_entries:
                    ex = entry["screen"]["x"]
                    ey = entry["screen"]["y"]
                    dist = abs(ex - sx) + abs(ey - sy)
                    if dist < best_dist:
                        best_dist = dist
                        best_entry = entry
            if best_entry is not None:
                cursor = {
                    "row": best_entry["grid"]["row"],
                    "col": best_entry["grid"]["col"],
                    "letter": best_entry["letter"],
                    "screen": dict(best_entry["screen"]),
                    "world": dict(best_entry["world"]),
                    "source": "sprite",
                }

        if cursor is None and self._naming_cursor_history:
            last = self._naming_cursor_history[-1]
            cursor = {
                "row": int(last.get("row", 0)),
                "col": int(last.get("col", 0)),
                "letter": str(last.get("letter", "")),
                "screen": dict(last.get("screen", {})),
                "world": dict(last.get("world", {})),
                "source": "history",
            }
        elif cursor is not None:
            history_entry = {
                "frame": frame_hash,
                "row": cursor["row"],
                "col": cursor["col"],
                "letter": cursor["letter"],
                "screen": dict(cursor.get("screen", {})),
                "world": dict(cursor.get("world", {})),
                "source": cursor.get("source", "sprite"),
            }
            self._naming_cursor_history.append(history_entry)

        presets: List[Dict[str, object]] = []

        def _flush_run(run: List[Dict[str, object]]) -> None:
            if not run:
                return
            word = "".join(entry.get("letter", "") for entry in run)
            if len(word) < 3 or len(word) > 5:
                return
            if not all(ch.isalpha() for ch in word):
                return
            start = run[0]
            presets.append(
                {
                    "label": word.upper(),
                    "row": start["grid"]["row"],
                    "col": start["grid"]["col"],
                    "length": len(word),
                    "screen": dict(start.get("screen", {})),
                }
            )

        for row in row_entries:
            current_run: List[Dict[str, object]] = []
            for entry in row:
                letter = entry.get("letter")
                if isinstance(letter, str) and len(letter) == 1 and letter.isalpha():
                    current_run.append(entry)
                else:
                    _flush_run(current_run)
                    current_run = []
            _flush_run(current_run)
        if presets:
            deduped_presets: List[Dict[str, object]] = []
            seen = set()
            for preset in presets:
                key = (preset["label"], preset["row"], preset["col"])
                if key in seen:
                    continue
                seen.add(key)
                deduped_presets.append(preset)
            presets = deduped_presets

        dialog_lines: Optional[List[str]] = None
        existing_dialog = overworld.get("dialog_lines")
        if isinstance(existing_dialog, list):
            cleaned = [str(line).strip() for line in existing_dialog if isinstance(line, str) and line.strip()]
            if cleaned:
                dialog_lines = cleaned[:4]

        naming_state: Dict[str, object] = {
            "entries": naming_entries,
            "grid_letters": grid_letters,
        }
        if cursor is not None:
            naming_state["cursor"] = cursor
        if self._naming_cursor_history:
            naming_state["cursor_history"] = list(self._naming_cursor_history)
        if presets:
            naming_state["presets"] = presets
        if dialog_lines:
            naming_state["dialog_lines"] = dialog_lines

        return naming_state

    def _tile_to_char(self, tile_id: int) -> str:
        for lower, upper, mode in self.DIALOG_CHAR_RANGES:
            if lower <= tile_id <= upper:
                offset = tile_id - lower
                if mode == "upper":
                    return chr(ord("A") + offset)
                if mode == "lower":
                    return chr(ord("a") + offset)
                if mode == "digit":
                    return chr(ord("0") + offset)
        if tile_id in self.DIALOG_PUNCTUATION:
            return self.DIALOG_PUNCTUATION[tile_id]
        if tile_id in self.DIALOG_SPACE_TILES:
            return " "
        return " "

    def _decode_dialog_text(self, screen: Mapping[str, object]) -> Optional[List[str]]:
        tile_ids = screen.get("tile_ids")
        if tile_ids is None:
            return None
        tile_grid = np.asarray(tile_ids, dtype=int)
        if tile_grid.ndim != 2:
            return None
        text_lines: List[str] = []
        for row in tile_grid:
            letters = sum(1 for val in row if 0x80 <= val <= 0xB9)
            if letters < 3:
                continue
            chars = [self._tile_to_char(int(val)) for val in row]
            text = "".join(chars).strip()
            if text:
                text_lines.append(text)
        if not text_lines:
            return None
        # Deduplicate contiguous identical lines
        deduped: List[str] = []
        for line in text_lines:
            if not deduped or deduped[-1] != line:
                deduped.append(line)
        return deduped[-3:]
