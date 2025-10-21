import unittest

from src.pkmn_battle.extractor.blue_extractor import BlueExtractor
from src.pkmn_battle.graph import GraphMemory
from src.pkmn_battle.graph.schema import WriteOp


def _build_obs() -> dict:
    return {
        "turn": 3,
        "field": {"weather_id": 1, "player_screens": 0b11, "player_side_effects": 0b101},
        "my_active": {
            "slot": 0,
            "species": 25,
            "level": 50,
            "hp": 120,
            "hp_max": 135,
            "status": 5,
            "types": [12],
            "moves": [
                {"slot": 0, "move_id": 85, "pp": 10, "max_pp": 15, "disabled": False},
                {"slot": 1, "move_id": 98, "pp": 15, "max_pp": 20, "disabled": False},
            ],
            "is_active": True,
            "fainted": False,
        },
        "opp_active": {
            "slot": 0,
            "species": 6,
            "level": 50,
            "hp": 80,
            "hp_max": 150,
            "status": 0,
            "types": [2, 9],
            "moves": [
                {"slot": 0, "move_id": 52, "pp": 10, "max_pp": 15, "disabled": False},
            ],
            "is_active": True,
            "fainted": False,
        },
        "my_party": [
            {
                "slot": 1,
                "species": 133,
                "level": 40,
                "hp": 90,
                "hp_max": 120,
                "status": 0,
                "moves": [{"slot": 0, "move_id": 97, "pp": 20, "max_pp": 20, "disabled": False}],
                "is_active": False,
                "fainted": False,
            }
        ],
        "opp_party": [],
    }


class TestBlueExtractor(unittest.TestCase):
    def test_emits_nodes_and_edges(self) -> None:
        extractor = BlueExtractor()
        obs = _build_obs()
        writes = extractor.extract(obs)
        self.assertTrue(writes, "extractor should emit write operations")

        memory = GraphMemory()
        for op in writes:
            memory.write(op)

        my_pokemon = memory.assoc(type_="Pokemon", filters={"side": "my"})
        self.assertTrue(any(node.attributes.get("species") == 25 for node in my_pokemon))
        self.assertTrue(all("revealed" in node.attributes for node in my_pokemon))

        moves = memory.follow(src="pokemon:my:0:25", relation="has_move")
        self.assertTrue(any(move.attributes.get("move_id") == 85 for move in moves))

        status_nodes = memory.follow(src="pokemon:my:0:25", relation="has_status")
        self.assertTrue(status_nodes)

        screens = memory.follow(src="side:my", relation="active_screen")
        self.assertTrue(screens)

    def test_low_confidence_fallback(self) -> None:
        extractor = BlueExtractor()
        obs = _build_obs()
        obs["my_active"]["species"] = 0  # unknown species
        writes = extractor.extract(obs)
        self.assertTrue(any(op.fallback_required for op in writes))
        pokemon_write = next(op for op in writes if op.kind == "node" and op.payload.type == "Pokemon")
        self.assertFalse(pokemon_write.payload.attributes.get("revealed", True))


if __name__ == "__main__":
    unittest.main()
