import unittest

from src.pkmn_battle.graph import GraphMemory
from src.pkmn_battle.graph.schema import Edge, Node, WriteOp


class TestGraphMemory(unittest.TestCase):
    def test_idempotent_write_and_follow(self) -> None:
        memory = GraphMemory()

        pika = Node("Pokemon", "pokemon:my:0:25", {"side": "my", "slot": 0, "species": 25})
        move = Node("Move", "move:85", {"move_id": 85, "max_pp": 24})
        edge = Edge("has_move", pika.node_id, move.node_id, {"slot": 0, "pp": 10})

        memory.write(WriteOp(kind="node", payload=pika, turn_hint=1))
        memory.write(WriteOp(kind="node", payload=pika, turn_hint=1))  # idempotent
        memory.write(WriteOp(kind="node", payload=move, turn_hint=1))
        memory.write(WriteOp(kind="edge", payload=edge, turn_hint=1))

        pokes = memory.assoc(type_="Pokemon")
        self.assertEqual(len(pokes), 1)

        moves = memory.follow(src=pika.node_id, relation="has_move")
        self.assertEqual(len(moves), 1)
        hop_trace = memory.drain_hops()
        self.assertEqual(len(hop_trace), 1)

    def test_follow_returns_empty_if_target_missing(self) -> None:
        memory = GraphMemory()
        edge = Edge("has_move", "pokemon:my:0:25", "move:85", {})
        memory.write(WriteOp(kind="edge", payload=edge, turn_hint=0))
        self.assertEqual(memory.follow(src="pokemon:my:0:25", relation="has_move"), [])

    def test_lru_and_turn_horizon(self) -> None:
        memory = GraphMemory(max_nodes_per_type=2, turn_horizon=1)

        a = Node("Pokemon", "pokemon:my:0:1", {"side": "my", "species": 1})
        b = Node("Pokemon", "pokemon:my:0:2", {"side": "my", "species": 2})
        c = Node("Pokemon", "pokemon:my:0:3", {"side": "my", "species": 3})

        memory.write(WriteOp(kind="node", payload=a, turn_hint=0))
        memory.write(WriteOp(kind="node", payload=b, turn_hint=0))
        memory.write(WriteOp(kind="node", payload=c, turn_hint=0))
        pokes = memory.assoc(type_="Pokemon")
        self.assertEqual(len(pokes), 2)
        self.assertFalse(any(p.node_id == "pokemon:my:0:1" for p in pokes))

        turn_old = Node("Turn", "turn:0", {"turn": 0})
        turn_new = Node("Turn", "turn:5", {"turn": 5})
        stale = Node("Pokemon", "pokemon:opp:0:99", {"side": "opp", "species": 99})
        memory.write(WriteOp(kind="node", payload=turn_old, turn_hint=0))
        memory.write(WriteOp(kind="node", payload=stale, turn_hint=0))
        memory.write(WriteOp(kind="node", payload=turn_new, turn_hint=5))

        recent = Node("Pokemon", "pokemon:opp:0:100", {"side": "opp", "species": 100})
        memory.write(WriteOp(kind="node", payload=recent, turn_hint=5))
        opp_nodes = memory.assoc(type_="Pokemon", filters={"side": "opp"})
        self.assertTrue(any(p.node_id == "pokemon:opp:0:100" for p in opp_nodes))
        self.assertFalse(any(p.node_id == "pokemon:opp:0:99" for p in opp_nodes))


if __name__ == "__main__":
    unittest.main()
