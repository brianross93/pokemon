import json, os
from pathlib import Path

from src.llm.llm_client import LLMConfig, create_llm_client
from src.llm.planlets.proposer import PlanletProposer
from pkmn_battle.summarizer import GraphSummary

# --- load the menu snapshot we want the planner to react to ---
snapshot_path = Path("tests/fixtures/overworld/snapshot_menu.json")
snapshot = json.loads(snapshot_path.read_text())

# --- set up the same summary object the controller would use ---
summary = GraphSummary(
    turn=0,
    side="p1",
    format=snapshot.get("map", {}).get("id", "menu_boot"),
    data={"overworld": snapshot},
)

# --- build a GPT‑5 client ---
config = LLMConfig(
    model="gpt-5",
    api_key=os.environ["OPENAI_API_KEY"],
    response_format={"type": "json_object"},
)
client = create_llm_client(config)

# --- ask the planner to generate a planlet for this snapshot ---
proposer = PlanletProposer()
proposal = proposer.generate_planlet(summary, client, allow_search=False)

print("RAW GPT-5 RESPONSE:")
print(proposal.raw_response or "<empty>")

print("\nPARSED PLANLET:")
print(json.dumps(proposal.planlet, indent=2))

