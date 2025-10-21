from llm.planlets import FakeOverworldLLM, PlanletProposer
from pkmn_overworld.world_graph import WorldGraph
from pkmn_overworld.summarizer import summarize_world_for_llm


def test_fake_overworld_llm_generates_planlet():
    world = WorldGraph()
    start, _ = world.ensure_corridor("map", length=2)
    sx, sy = [int(part) for part in start.split(":")[2:]]
    world.set_player_position("map", sx, sy)
    summary = summarize_world_for_llm(world.memory)

    llm = FakeOverworldLLM()
    proposer = PlanletProposer()
    proposal = proposer.generate_planlet(summary, llm, allow_search=False)

    assert proposal.planlet["kind"] == "OVERWORLD"
    assert proposal.planlet["script"]
    assert proposal.search_calls == 0
