"""
Overworld-specific GraphMemory wrapper.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from src.pkmn_battle.graph.memory import GraphMemory
from src.pkmn_battle.graph.schema import Edge, Node, WriteOp


class OverworldMemory(GraphMemory):
    """
    Specialised memory store for overworld entities.
    """

    def __init__(
        self,
        *,
        max_nodes_per_type: int = 512,
        turn_horizon: int = 0,
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            max_nodes_per_type=max_nodes_per_type,
            turn_horizon=turn_horizon,
            confidence_threshold=confidence_threshold,
        )

    def latest_player_tile(self) -> Optional[str]:
        """Convenience accessor for the player's current tile node id."""

        matches = self.assoc(type_="Player")
        if not matches:
            return None
        player = matches[-1]
        map_id = player.attributes.get("map_id", "unknown")
        x = player.attributes.get("x")
        y = player.attributes.get("y")
        if x is None or y is None:
            return None
        return f"tile:{map_id}:{int(x)}:{int(y)}"

    def assoc_player(self) -> Optional[Node]:
        matches = self.assoc(type_="Player")
        return matches[-1] if matches else None

    def player_coords(self) -> Optional[Tuple[int, int]]:
        player = self.assoc_player()
        if player is None:
            return None
        try:
            x = int(player.attributes.get("x"))
            y = int(player.attributes.get("y"))
            return (x, y)
        except (TypeError, ValueError):
            return None

    def tile_node(self, map_id: str, x: int, y: int) -> Optional[Node]:
        node_id = f"tile:{map_id}:{x}:{y}"
        tiles = self.assoc(type_="Tile", key=node_id)
        return tiles[-1] if tiles else None

    def adjacent_tiles(self, tile_id: str) -> List[Node]:
        return list(self.follow(src=tile_id, relation="adjacent"))

    def warp_exit_tiles(self, tile_id: str) -> List[Node]:
        return list(self.follow(src=tile_id, relation="warp_exit"))

    def manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    # ------------------------------------------------------------------ #
    # Helper factories
    # ------------------------------------------------------------------ #

    @staticmethod
    def make_map_region(map_id: str, name: str) -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(type="MapRegion", node_id=f"map:{map_id}", attributes={"map_id": map_id, "name": name}),
        )

    @staticmethod
    def make_tile(map_id: str, x: int, y: int, *, passable: bool, terrain: str, special: str = "") -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(
                type="Tile",
                node_id=f"tile:{map_id}:{x}:{y}",
                attributes={
                    "map_id": map_id,
                    "x": x,
                    "y": y,
                    "passable": passable,
                    "terrain": terrain,
                    "special": special,
                },
            ),
        )

    @staticmethod
    def make_warp(
        warp_id: str,
        *,
        src_map: str,
        src_tile: Iterable[int],
        dst_map: str,
        dst_tile: Iterable[int],
    ) -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(
                type="Warp",
                node_id=f"warp:{warp_id}",
                attributes={
                    "src_map": src_map,
                    "src_tile": list(src_tile),
                    "dst_map": dst_map,
                    "dst_tile": list(dst_tile),
                },
            ),
        )

    @staticmethod
    def make_npc(
        npc_id: str,
        *,
        name: str,
        map_id: str,
        x: int,
        y: int,
        role: str,
        facing: str,
    ) -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(
                type="NPC",
                node_id=f"npc:{npc_id}",
                attributes={
                    "name": name,
                    "map_id": map_id,
                    "x": x,
                    "y": y,
                    "role": role,
                    "facing": facing,
                },
            ),
        )

    @staticmethod
    def make_inventory_item(item_id: str, *, name: str, quantity: int) -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(
                type="InventoryItem",
                node_id=f"inventory:{item_id}",
                attributes={"item_id": item_id, "name": name, "quantity": quantity},
            ),
        )

    @staticmethod
    def make_item(item_id: str, *, name: str) -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(
                type="Item",
                node_id=f"item:{item_id}",
                attributes={"item_id": item_id, "name": name},
            ),
        )

    @staticmethod
    def make_player(map_id: str, x: int, y: int, *, facing: str) -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(
                type="Player",
                node_id="player:me",
                attributes={"map_id": map_id, "x": x, "y": y, "facing": facing},
            ),
        )

    @staticmethod
    def make_flag(flag_id: str, *, name: str, value: object) -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(
                type="Flag",
                node_id=f"flag:{flag_id}",
                attributes={"name": name, "value": value},
            ),
        )

    @staticmethod
    def make_menu_state(menu_id: str, *, path: Iterable[str], open_: bool) -> WriteOp:
        return WriteOp(
            kind="node",
            payload=Node(
                type="MenuState",
                node_id=f"menu:{menu_id}",
                attributes={"path": list(path), "open": bool(open_)},
            ),
        )

    @staticmethod
    def make_adjacent_edge(src_tile_id: str, dst_tile_id: str, *, passable: bool) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(
                relation="adjacent",
                src=src_tile_id,
                dst=dst_tile_id,
                attributes={"passable": passable},
            ),
        )

    @staticmethod
    def make_contains_edge(map_node_id: str, tile_node_id: str) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(relation="contains", src=map_node_id, dst=tile_node_id, attributes={}),
        )

    @staticmethod
    def make_located_edge(src_id: str, tile_node_id: str) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(relation="located_at", src=src_id, dst=tile_node_id, attributes={}),
        )

    @staticmethod
    def make_warp_edge(warp_node_id: str, dst_tile_id: str) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(relation="warp_to", src=warp_node_id, dst=dst_tile_id, attributes={}),
        )

    @staticmethod
    def make_warp_exit_edge(src_tile_id: str, dst_tile_id: str, *, warp_id: str) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(
                relation="warp_exit",
                src=src_tile_id,
                dst=dst_tile_id,
                attributes={"warp_id": warp_id},
            ),
        )

    @staticmethod
    def make_owns_edge(src_id: str, item_node_id: str, *, quantity: int) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(
                relation="owns",
                src=src_id,
                dst=item_node_id,
                attributes={"quantity": quantity},
            ),
        )

    @staticmethod
    def make_offers_edge(npc_node_id: str, item_node_id: str) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(
                relation="offers",
                src=npc_node_id,
                dst=item_node_id,
                attributes={},
            ),
        )

    @staticmethod
    def make_requires_edge(planlet_node_id: str, flag_node_id: str) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(
                relation="requires",
                src=planlet_node_id,
                dst=flag_node_id,
                attributes={},
            ),
        )

    @staticmethod
    def make_last_path_edge(src_tile_id: str, dst_tile_id: str, *, planlet_id: str | None) -> WriteOp:
        return WriteOp(
            kind="edge",
            payload=Edge(
                relation="last_path",
                src=src_tile_id,
                dst=dst_tile_id,
                attributes={"planlet_id": planlet_id},
            ),
        )

    # ------------------------------------------------------------------ #
    # Telemetry helpers
    # ------------------------------------------------------------------ #

    def summarise_nodes(self) -> Dict[str, int]:
        """Return a histogram of nodes by type."""

        counts: Dict[str, int] = defaultdict(int)
        for node in self.assoc():
            counts[node.type] += 1
        return dict(counts)

    def make_step_telemetry(
        self,
        *,
        planlet_id: str,
        planlet_kind: str,
        gate: Dict[str, object],
        action: Dict[str, object],
        latency_ms: float,
        fallback_required: bool,
    ) -> Dict[str, object]:
        """
        Construct a telemetry payload for a single executor step, draining hop traces.
        """

        core = {
            "gate": dict(gate),
            "action": dict(action),
            "latency_ms": float(latency_ms),
            "fallback_required": bool(fallback_required),
            "hop_trace": self.drain_hops(),
        }
        return {"core": core, "overworld": {}}
