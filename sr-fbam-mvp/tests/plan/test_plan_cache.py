from __future__ import annotations

from dataclasses import dataclass

from src.plan.cache import PlanCache, make_battle_cache_key, make_overworld_cache_key
from pkmn_battle.summarizer import GraphSummary
from pkmn_overworld.summarizer import WorldSummary


@dataclass
class FakeClock:
    now: float = 0.0

    def advance(self, delta: float) -> None:
        self.now += delta

    def __call__(self) -> float:
        return self.now


def _battle_summary(turn: int = 1) -> GraphSummary:
    data = {
        "entities": {
            "Pokemon": [
                {"node_id": "poke:1", "name": "Bulbasaur"},
                {"node_id": "poke:2", "name": "Charmander"},
            ]
        },
        "relations": [
            {"src": "poke:1", "relation": "ally", "dst": "side-p1"},
            {"src": "poke:2", "relation": "ally", "dst": "side-p2"},
        ],
        "actives": {"ours": "poke:1", "opponent": "poke:2"},
        "hazards": {"ours": [], "opponent": []},
    }
    return GraphSummary(turn=turn, side="p1", format="gen9ou", data=data)


def _overworld_summary(map_id: str = "map") -> WorldSummary:
    data = {
        "player": {"node_id": "player:0", "x": 12, "y": 10},
        "tiles": [{"node_id": "tile:map:12:10", "terrain": "grass"}],
        "nearby": {
            "npcs": [{"node_id": "npc:oak"}],
            "shops": [],
            "hazards": [],
        },
        "graph": {"nodes": 10, "edges": 12},
    }
    return WorldSummary(map_id=map_id, side="p1", data=data)


def test_battle_cache_key_changes_with_turn() -> None:
    key_a = make_battle_cache_key(_battle_summary(turn=1))
    key_b = make_battle_cache_key(_battle_summary(turn=2))
    assert key_a != key_b


def test_overworld_cache_key_changes_with_map() -> None:
    key_a = make_overworld_cache_key(_overworld_summary("map_a"))
    key_b = make_overworld_cache_key(_overworld_summary("map_b"))
    assert key_a != key_b


def test_plan_cache_ttl_eviction() -> None:
    clock = FakeClock()
    cache = PlanCache(max_size=2, ttl_seconds=10.0, clock=clock)
    cache.store("k1", {"planlet_id": "p1"})
    assert cache.lookup("k1") is not None
    clock.advance(11.0)
    cache.prune()
    assert cache.lookup("k1") is None


def test_plan_cache_success_weighted_eviction() -> None:
    clock = FakeClock()
    cache = PlanCache(max_size=2, ttl_seconds=100.0, clock=clock)
    cache.store("k1", {"planlet_id": "pa"})
    cache.store("k2", {"planlet_id": "pb"})
    cache.record_feedback("k1", success=True, weight=3.0)
    cache.record_feedback("k2", success=False, weight=2.0)
    cache.store("k3", {"planlet_id": "pc"})

    stats = cache.stats()
    assert "k3" in stats["keys"]
    assert "pa" in stats["planlets"].values()
    # Low success entry should have been evicted.
    assert "pb" not in stats["planlets"].values()
