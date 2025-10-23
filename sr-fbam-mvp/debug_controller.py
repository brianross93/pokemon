import json, os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.llm_client import LLMConfig, create_llm_client
from llm.planlets.proposer import PlanletProposer

def main():
    print("Testing controller-like GPT-5 flow...")
    
    # Build a GPT-5 client (same as controller)
    config = LLMConfig(
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
        response_format={"type": "json_object"},
    )
    client = create_llm_client(config)
    
    # Create proposer (same as controller)
    proposer = PlanletProposer()
    
    # Create a simple mock summary
    class MockSummary:
        def __init__(self):
            self.turn = 0
            self.format = "overworld"
            self.side = "player"
            self.nodes = []
            self.edges = []
            self.nearby_nodes = []
    
    summary = MockSummary()
    
    print("Generating planlet...")
    try:
        proposal = proposer.generate_planlet(summary, client, allow_search=False)
        print("SUCCESS!")
        print(f"Raw response: '{proposal.raw_response}'")
        print(f"Planlet: {json.dumps(proposal.planlet, indent=2)}")
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Debug the raw response
        try:
            messages = [
                {"role": "system", "content": "You are a Pokemon game assistant."},
                {"role": "user", "content": "Generate a menu sequence. Return JSON with buttons array."}
            ]
            raw = client.generate_response(messages)
            print(f"Direct client response: '{raw}'")
            print(f"Response type: {type(raw)}")
            print(f"Response length: {len(raw)}")
            
            # Test normalise_llm_output
            content, usage = proposer._normalise_llm_output(raw)
            print(f"After normalise: '{content}'")
            print(f"Usage: {usage}")
            
        except Exception as debug_e:
            print(f"Debug error: {debug_e}")

if __name__ == "__main__":
    main()

