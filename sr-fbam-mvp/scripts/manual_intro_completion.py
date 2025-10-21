#!/usr/bin/env python3
"""
Manual intro completion script for Pokemon Blue.
This script will help you manually complete the intro sequence and create a save state.
"""

import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from middleware.pyboy_adapter import PyBoyPokemonAdapter, PyBoyConfig, PokemonAction

def main():
    print("Manual Pokemon Blue Intro Completion")
    print("=" * 50)
    print("This script will help you complete the intro sequence manually.")
    print("Follow the instructions on screen to navigate through the intro.")
    print("Once you reach the overworld with control, we'll save the state.")
    print()
    
    # Get ROM path
    rom_path = input("Enter path to Pokemon Blue ROM: ").strip()
    if not rom_path:
        rom_path = "../pokemonblue/Pokemon Blue.gb"
    
    print(f"Using ROM: {rom_path}")
    print()
    
    # Create adapter
    config = PyBoyConfig(
        rom_path=rom_path,
        window_type="SDL2",  # Show window for manual control
        addresses={
            "map_id": 0xD35E,
            "player_x": 0xD361,
            "player_y": 0xD362,
            "in_grass": 0xD5A5,
            "in_battle": 0xD057,
            "menu_state": 0xD730,
            "menu_cursor": 0xD730,
            "joy_ignore": 0xD730,
        }
    )
    
    adapter = PyBoyPokemonAdapter(config)
    
    try:
        print("🚀 Starting Pokemon Blue...")
        telemetry = adapter.reset()
        
        print(f"Initial state: Area {telemetry.area_id}, Position ({telemetry.x}, {telemetry.y})")
        print()
        
        print("📋 MANUAL INSTRUCTIONS:")
        print("1. You should see the Pokemon Blue game window")
        print("2. Navigate through the intro sequence manually:")
        print("   - Press A to advance through dialogue")
        print("   - When you reach name selection, navigate to 'END' and press A")
        print("   - Continue pressing A through all dialogue")
        print("   - You should eventually reach the overworld where you can move")
        print()
        print("3. Once you can move around freely, press ENTER here to save the state")
        print()
        
        input("Press ENTER when you have control and can move around...")
        
        # Check final state
        telemetry = adapter._read_telemetry()
        print(f"Final state: Area {telemetry.area_id}, Position ({telemetry.x}, {telemetry.y})")
        
        # Save the state
        print("💾 Saving state...")
        adapter.save_state(0)
        print("✅ State saved successfully!")
        print()
        print("🎉 You can now use this save state in the main controller!")
        print("The controller will automatically load this state and skip the intro.")
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        try:
            adapter.close()
        except:
            pass

if __name__ == "__main__":
    main()
