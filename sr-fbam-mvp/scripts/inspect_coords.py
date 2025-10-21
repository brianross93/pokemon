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
sequence = [
    ("START", 60, 10),
    ("A", 60, 10),
    ("A", 600, 8),
    ("DOWN", 1, 30),
    ("A", 1, 12),
    ("A", 400, 8),
    ("DOWN", 1, 30),
    ("A", 1, 12),
    ("A", 1200, 8),
]
for name, times, frames in sequence:
    for _ in range(times):
        telemetry = adapter.step(PokemonAction(name, {"frames": frames}))

for _ in range(500):
    telemetry = adapter.step(PokemonAction("A", {"frames": 8}))
    if telemetry.extra.get("joy_ignore", 1) == 0:
        break

mem = adapter.pyboy.memory
print("initial", telemetry.area_id, telemetry.x, telemetry.y, telemetry.extra.get("joy_ignore"))
print("coords D132/D133", mem[0xD132], mem[0xD133])
print("coords D361/D362", mem[0xD361], mem[0xD362])

for step in range(5):
    telemetry = adapter.step(PokemonAction("DOWN", {"frames": 24}))
    mem = adapter.pyboy.memory
    print(
        "step",
        step,
        "area",
        telemetry.area_id,
        "pos",
        (telemetry.x, telemetry.y),
        "joy",
        telemetry.extra.get("joy_ignore"),
    )
    print("  D132/D133", mem[0xD132], mem[0xD133], "D361/D362", mem[0xD361], mem[0xD362])

adapter.close()
