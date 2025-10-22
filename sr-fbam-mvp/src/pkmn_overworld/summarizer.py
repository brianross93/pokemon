from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.overworld.graph.overworld_memory import OverworldMemory


def _sorted_nodes(memory: OverworldMemory, type_: str, limit: int | None = None) -> List[Dict[str, Any]]:
    nodes = memory.assoc(type_=type_)
    items: List[Dict[str, Any]] = []
    for node in nodes:
        payload = {"node_id": node.node_id}
        payload.update(node.attributes)
        items.append(payload)
    items.sort(key=lambda item: item["node_id"])
    if limit is not None:
        items = items[:limit]
    return items


@dataclass
class WorldSummary:
    map_id: str
    side: str
    data: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        return {"map_id": self.map_id, **self.data}


def summarize_world_for_llm(memory: OverworldMemory, *, nearby_limit: int = 5) -> WorldSummary:
    """Produce a deterministic summary of the overworld memory."""

    player_nodes = memory.assoc(type_="Player")
    player_payload: Dict[str, Any] = {}
    map_id = "unknown"
    if player_nodes:
        player_node = player_nodes[-1]
        map_id = str(player_node.attributes.get("map_id", "unknown"))
        player_payload = dict(player_node.attributes)
        player_payload["node_id"] = player_node.node_id

    tiles = _sorted_nodes(memory, "Tile")
    hazards = _sorted_nodes(memory, "Hazard")
    npcs = _sorted_nodes(memory, "NPC")
    shops = _sorted_nodes(memory, "Shop")
    menus = _sorted_nodes(memory, "MenuState")

    nearby_tiles = tiles[:nearby_limit]
    nearby_npcs = npcs[:nearby_limit]
    nearby_shops = shops[:nearby_limit]
    nearby_hazards = hazards[:nearby_limit]
    nearby_menus = menus[:nearby_limit]

    snapshot = memory.snapshot()
    graph_stats = {
        "nodes": len(snapshot.nodes),
        "edges": len(snapshot.edges),
    }

    summary = {
        "player": player_payload,
        "tiles": nearby_tiles,
        "nearby": {
            "npcs": nearby_npcs,
            "shops": nearby_shops,
            "hazards": nearby_hazards,
            "menus": nearby_menus,
        },
        "menus": [menu for menu in menus if menu.get("open")],
        "graph": graph_stats,
    }

    return WorldSummary(map_id=map_id, side="p1", data=summary)

