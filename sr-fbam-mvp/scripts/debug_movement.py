"""
Quick debug script to inspect movement telemetry.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter
from src.middleware.pokemon_adapter import PokemonAction

cfg = PyBoyConfig(
    rom_path="../pokemonblue/Pokemon Blue.gb",
    window_type="null",
    addresses={
        "map_id": 0xD35E,
        "player_x": 0xD361,
        "player_y": 0xD362,
        "in_grass": 0xD5A5,
        "in_battle": 0xD057,
        "species_id": 0xD058,
    },
)

adapter = PyBoyPokemonAdapter(cfg)

telemetry = adapter.reset()
print("reset", telemetry.area_id, telemetry.x, telemetry.y)

for _ in range(60):
    telemetry = adapter.step(PokemonAction("START"))
for _ in range(60):
    telemetry = adapter.step(PokemonAction("A"))
for _ in range(2000):
    telemetry = adapter.step(PokemonAction("A"))

print("after intro", telemetry.area_id, telemetry.x, telemetry.y)
mem = adapter.pyboy.memory
print("memory slice before", [mem[0xD35C + i] for i in range(0x20)])

telemetry = adapter.step(PokemonAction("DOWN", {"frames": 120}))
print("after down", telemetry.area_id, telemetry.x, telemetry.y)
print("memory slice after down", [mem[0xD35C + i] for i in range(0x20)])

telemetry = adapter.step(PokemonAction("LEFT", {"frames": 120}))
print("after left", telemetry.area_id, telemetry.x, telemetry.y)
print("memory slice after left", [mem[0xD35C + i] for i in range(0x20)])

adapter.close()
