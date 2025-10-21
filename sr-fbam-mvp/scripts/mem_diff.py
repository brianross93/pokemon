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
for _ in range(60):
    telemetry = adapter.step(PokemonAction("START"))
for _ in range(60):
    telemetry = adapter.step(PokemonAction("A"))
for _ in range(2000):
    telemetry = adapter.step(PokemonAction("A"))

mem = adapter.pyboy.memory

ranges = [
    (0xD000, 0x100),
    (0xD100, 0x100),
    (0xD200, 0x100),
    (0xD300, 0x100),
    (0xD400, 0x100),
    (0xD500, 0x100),
    (0xD600, 0x100),
    (0xD700, 0x100),
]

before = {}
for base, length in ranges:
    before[base] = [mem[base + i] for i in range(length)]

telemetry = adapter.step(PokemonAction("DOWN", {"frames": 60}))

changes = []
for base, length in ranges:
    after = [mem[base + i] for i in range(length)]
    prev = before[base]
    for i in range(length):
        if prev[i] != after[i]:
            changes.append((base + i, prev[i], after[i]))

print("total changes", len(changes))
print(changes)
adapter.close()
