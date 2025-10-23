import json, os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.llm_client import LLMConfig, create_llm_client

def main():
    print("Testing GPT-5 LLM client directly...")
    print(f"API Key set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    
    # Build a GPT-5 client
    config = LLMConfig(
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
        response_format={"type": "json_object"},
    )
    client = create_llm_client(config)
    
    # Simple test message
    messages = [
        {"role": "system", "content": "You are a Pokemon game assistant. Return JSON only."},
        {"role": "user", "content": "Generate a menu sequence to start a new game. Return JSON with a 'buttons' array like [\"START\", \"A\", \"A\"]."}
    ]
    
    print("Sending request to GPT-5...")
    try:
        response = client.generate_response(messages)
        print(f"Response type: {type(response)}")
        print(f"Response length: {len(response) if response else 0}")
        print(f"Raw response: '{response}'")
        
        if response:
            try:
                parsed = json.loads(response)
                print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
        else:
            print("ERROR: Empty response from GPT-5")
            
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    main()
