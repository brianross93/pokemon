from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from src.pkmn_battle.graph.schema import WriteOp

from .world_graph import WorldGraph, _tile_node_id


def parse_screen_state(screen_state: Dict[str, object], graph: Optional[WorldGraph] = None) -> WorldGraph:
    """
    Minimal screen parser that maps a structured JSON payload into the WorldGraph.

    Parameters
    ----------
    screen_state:
        Dictionary with optional keys: ``map_id``, ``tiles``, ``warps``, ``hazards``.
        This function expects a simplified representation used for smoke tests and
        tooling prototypes. Example::

            {
                "map_id": "corridor",
                "tiles": [
                    {"x": 0, "y": 0, "terrain": "path", "adjacent": [[1, 0]]},
                    {"x": 1, "y": 0, "terrain": "path", "adjacent": [[0, 0], [2, 0]]}
                ],
                "hazards": [{"id": "grass-0", "kind": "grass", "tile": [2, 0]}]
            }

    graph:
        Optional :class:`WorldGraph` instance to populate. When omitted a new graph
        backed by :class:`OverworldMemory` is created.

    Returns
    -------
    WorldGraph
        The populated graph wrapper.
    """

    world_graph = graph or WorldGraph()
    map_id = str(screen_state.get("map_id", "unknown"))

    # Tiles and adjacency
    tiles: List[Dict[str, object]] = screen_state.get("tiles", []) or []
    for tile in tiles:
        x = int(tile.get("x", 0))
        y = int(tile.get("y", 0))
        terrain = tile.get("terrain")
        tile_id = world_graph.add_tile(map_id, x, y, terrain=terrain if isinstance(terrain, str) else None)
        for neighbor in tile.get("adjacent", []) or []:
            if isinstance(neighbor, (list, tuple)) and len(neighbor) >= 2:
                nx, ny = int(neighbor[0]), int(neighbor[1])
                neighbor_id = world_graph.add_tile(map_id, nx, ny)
                world_graph.connect_adjacent(tile_id, neighbor_id)

    # Hazards
    for hazard in screen_state.get("hazards", []) or []:
        hid = str(hazard.get("id", "hazard"))
        kind = str(hazard.get("kind", "unknown"))
        tile_ref = hazard.get("tile")
        tile_id = None
        if isinstance(tile_ref, (list, tuple)) and len(tile_ref) >= 2:
            tile_id = world_graph.add_tile(map_id, int(tile_ref[0]), int(tile_ref[1]))
        world_graph.add_hazard(hid, kind=kind, tile=tile_id)

    # Warps
    for warp in screen_state.get("warps", []) or []:
        src = warp.get("src")
        dst = warp.get("dst")
        if isinstance(src, (list, tuple)) and isinstance(dst, (list, tuple)) and len(src) >= 2 and len(dst) >= 2:
            src_id = world_graph.add_tile(map_id, int(src[0]), int(src[1]))
            dst_id = world_graph.add_tile(map_id, int(dst[0]), int(dst[1]))
            world_graph.add_warp(src_id, dst_id)

    return world_graph
