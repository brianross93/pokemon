#!/usr/bin/env python3
"""
Debug script to understand what's happening during the Pokemon Blue intro.
This will help us figure out why we can't move after the intro sequence.
"""

import sys
from pathlib import Path
import time

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from middleware.pyboy_adapter import PyBoyPokemonAdapter, PyBoyConfig, PokemonAction

def main():
    print("Pokemon Blue Intro Debug")
    print("=" * 50)
    print("This script will help us understand what's happening during the intro.")
    print("We'll press A and watch the telemetry to see when we can actually move.")
    print()
    
    # Get ROM path
    rom_path = input("Enter path to Pokemon Blue ROM: ").strip()
    if not rom_path:
        rom_path = "../pokemonblue/Pokemon Blue.gb"

    config = PyBoyConfig(
        rom_path=rom_path,
        window_type="SDL2",  # Always show window for debugging
        speed=1,  # Normal speed for debugging
        # Default addresses, will be updated by inspect_pyboy_memory.py
        addresses={
            "map_id": 0xD35E,
            "player_x": 0xD361,
            "player_y": 0xD362,
            "in_grass": 0xD5A5,
            "in_battle": 0xD057,
            "menu_state": 0xCC3B, # Placeholder, needs to be found
            "menu_cursor": 0xCC3C, # Placeholder, needs to be found
            "joy_ignore": 0xD730,
        }
    )
    
    adapter = PyBoyPokemonAdapter(config)
    telemetry = adapter.reset()

    print("Game started. Watch the PyBoy window and press Enter to continue...")
    input()
    
    print("Starting intro sequence...")
    print("Press Enter after each A press to see what happens...")
    
    for i in range(50):  # Try up to 50 A presses
        print(f"\n--- A Press {i+1} ---")
        print(f"Before: area={telemetry.area_id}, pos=({telemetry.x},{telemetry.y}), joy_ignore={telemetry.extra.get('joy_ignore', 'N/A')}")
        
        # Press A
        telemetry = adapter.step(PokemonAction("A", {"frames": 6}))
        
        print(f"After:  area={telemetry.area_id}, pos=({telemetry.x},{telemetry.y}), joy_ignore={telemetry.extra.get('joy_ignore', 'N/A')}")
        
        # Check if we can move now
        if telemetry.area_id > 0 and telemetry.x > 0 and telemetry.y > 0:
            print("🎉 We're in the overworld! Testing movement...")
            
            # Try moving
            original_pos = (telemetry.x, telemetry.y)
            telemetry = adapter.step(PokemonAction("DOWN", {"frames": 12}))
            new_pos = (telemetry.x, telemetry.y)
            
            if new_pos != original_pos:
                print(f"✅ Movement worked! {original_pos} -> {new_pos}")
                print("SUCCESS! We can move now!")
                break
            else:
                print(f"❌ Movement didn't work, still at {original_pos}")
        
        # Ask user to continue
        user_input = input("Press Enter to continue, or 'q' to quit: ").strip()
        if user_input.lower() == 'q':
            break
    
    print("\nDebug session finished.")
    adapter.pyboy.stop()

if __name__ == "__main__":
    main()





