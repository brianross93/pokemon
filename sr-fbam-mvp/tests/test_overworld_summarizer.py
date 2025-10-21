from src.pkmn_overworld.world_graph import WorldGraph
from src.pkmn_overworld.summarizer import summarize_world_for_llm


def test_summarize_world_deterministic():
    world = WorldGraph()
    start, end = world.ensure_corridor("map", length=3)
    sx, sy = [int(part) for part in start.split(":")[2:]]
    world.set_player_position("map", sx, sy)
    summary = summarize_world_for_llm(world.memory)

    payload = summary.to_payload()
    assert payload["map_id"] == "map"
    assert payload["player"]["map_id"] == "map"
    tile_ids = [tile["node_id"] for tile in payload["tiles"]]
    assert tile_ids == sorted(tile_ids)
    # Ensure nearby hazards / shops lists exist even if empty
    assert "hazards" in payload["nearby"]
    assert "shops" in payload["nearby"]
