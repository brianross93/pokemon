#!/usr/bin/env python3
"""
Script to automatically start a new Pokemon Blue game and get to the overworld.
This handles the menu navigation and intro sequence.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from middleware.pyboy_adapter import PyBoyPokemonAdapter, PyBoyConfig
except ImportError:
    print("Error: Could not import middleware. Make sure you're in the sr-fbam-mvp directory.")
    sys.exit(1)

def start_new_game(rom_path: str, steps: int = 2000) -> None:
    """Start a new Pokemon Blue game and navigate to the overworld."""
    
    # Configure PyBoy
    config = PyBoyConfig(
        rom_path=rom_path,
        window_type="null",  # Headless
        speed=0,  # Normal speed
        auto_save_slot=None,
        addresses={
            "map_id": 0xD35E,
            "player_x": 0xD361, 
            "player_y": 0xD362,
            "in_battle": 0xD057,
            "species_id": 0xD058,
            "grass_flag": 0xD5A5,
        }
    )
    
    print("Starting Pokemon Blue...")
    adapter = PyBoyPokemonAdapter(config)
    
    # Get initial state
    telemetry = adapter.reset()
    print(f"Initial state: area={telemetry.area_id}, x={telemetry.x}, y={telemetry.y}")
    
    # Step 1: Press START to open menu
    print("Pressing START to open menu...")
    for _ in range(60):  # Wait for menu to appear
        telemetry = adapter.step("START")
        if telemetry.area_id != 0:  # Menu opened
            break
        time.sleep(0.1)
    
    # Step 2: Press A to select "NEW GAME"
    print("Selecting NEW GAME...")
    for _ in range(60):
        telemetry = adapter.step("A")
        if telemetry.area_id != 0:  # Game started
            break
        time.sleep(0.1)
    
    # Step 3: Navigate through intro (name selection, etc.)
    print("Navigating intro sequence...")
    for i in range(steps):
        # Use A to advance through text
        telemetry = adapter.step("A")
        
        # Check if we're in the actual game world
        if telemetry.area_id > 0 and telemetry.x > 0:
            print(f"Game started! Area: {telemetry.area_id}, Position: ({telemetry.x}, {telemetry.y})")
            break
            
        if i % 100 == 0:
            print(f"  Step {i}: area={telemetry.area_id}, x={telemetry.x}, y={telemetry.y}")
    
    # Final state
    print(f"Final state: area={telemetry.area_id}, x={telemetry.x}, y={telemetry.y}")
    print("Game ready for learning system!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start a new Pokemon Blue game")
    parser.add_argument("--rom", required=True, help="Path to Pokemon Blue ROM")
    parser.add_argument("--steps", type=int, default=2000, help="Max steps for intro navigation")
    
    args = parser.parse_args()
    start_new_game(args.rom, args.steps)
