import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter
from src.middleware.pokemon_adapter import PokemonAction

ROM_PATH = ROOT / "pokemonblue" / "Pokemon Blue.gb"
if not ROM_PATH.exists():
    pytest.skip("pokemon ROM not available; skipping quick_move_test", allow_module_level=True)

cfg = PyBoyConfig(
    rom_path=str(ROM_PATH),
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

for _ in range(1000):
    telemetry = adapter.step(PokemonAction("A", {"frames": 8}))
    if telemetry.extra.get("joy_ignore", 1) == 0:
        break

print("control", telemetry.area_id, telemetry.x, telemetry.y, telemetry.extra.get("joy_ignore"))
telemetry = adapter.step(PokemonAction("DOWN", {"frames": 240}))
print("after hold-down", telemetry.area_id, telemetry.x, telemetry.y, telemetry.extra.get("joy_ignore"))
telemetry = adapter.step(PokemonAction("RIGHT", {"frames": 240}))
print("after hold-right", telemetry.area_id, telemetry.x, telemetry.y, telemetry.extra.get("joy_ignore"))

adapter.close()
