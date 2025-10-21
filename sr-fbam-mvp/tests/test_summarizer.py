from pkmn_battle.graph.memory import GraphMemory
from pkmn_battle.graph.schema import Edge, Node, WriteOp
from pkmn_battle.summarizer import summarize_for_llm


def test_summarize_for_llm_orders_nodes_and_edges():
    memory = GraphMemory()
    memory.write(WriteOp(kind="node", payload=Node(type="turn", node_id="turn-1", attributes={"turn": 3})))
    memory.write(WriteOp(kind="node", payload=Node(type="pokemon", node_id="p1", attributes={"name": "Greninja"})))
    memory.write(WriteOp(kind="node", payload=Node(type="pokemon", node_id="p2", attributes={"name": "Dragonite"})))
    memory.write(WriteOp(kind="edge", payload=Edge(relation="active", src="side-p1", dst="p1", attributes={})))
    memory.write(WriteOp(kind="edge", payload=Edge(relation="active", src="side-p2", dst="p2", attributes={})))

    summary = summarize_for_llm(memory, side_view="p1")
    assert summary.turn == 3
    assert summary.side == "p1"
    assert summary.format == "unknown"

    entities = summary.data["entities"]
    assert set(entities.keys()) == {"turn", "pokemon"}
    pokemon_ids = [entry["node_id"] for entry in entities["pokemon"]]
    assert pokemon_ids == sorted(pokemon_ids)

    relations = summary.data["relations"]
    assert relations == sorted(
        relations,
        key=lambda e: (e["src"], e["relation"], e["dst"], tuple(sorted(e.items()))),
    )

    actives = summary.data["actives"]
    assert actives["ours"] == "p1"
    assert actives["opponent"] == "p2"
    assert summary.data["hazards"] == {"ours": [], "opponent": []}


def test_summarize_for_llm_defaults_to_zero_turn():
    memory = GraphMemory()
    summary = summarize_for_llm(memory, side_view="p2")
    assert summary.turn == 0
    assert summary.side == "p2"
    assert summary.format == "unknown"
    assert summary.data["entities"] == {}
    assert summary.data["relations"] == []
    assert summary.data["actives"] == {"ours": None, "opponent": None}
    assert summary.data["hazards"] == {"ours": [], "opponent": []}
