"""
PyBoy-based implementation of the PokemonBlueAdapter interface.

This module provides a thin bridge between the PyBoy emulator and the symbolic
middleware. Memory addresses are left as placeholders that must be populated
with the offsets corresponding to the ROM in use. See the documentation in
`KNOWLEDGE_MIDDLEWARE_PLAN.md` for workflow guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, TYPE_CHECKING

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    PyBoy = None  # type: ignore[assignment]
    WindowEvent = None  # type: ignore[assignment]

from src.middleware.pokemon_adapter import PokemonAction, PokemonBlueAdapter, PokemonTelemetry

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.pkmn_battle.env.blue_ram_adapter import BlueBattleRAMMap, BlueRAMAdapter


DEFAULT_ADDRS: Dict[str, int] = {
    "map_id": 0x0000,
    "player_x": 0x0000,
    "player_y": 0x0000,
    "in_grass": 0x0000,
    "in_battle": 0x0000,
    "species_id": 0x0000,
    "step_counter": 0x0000,
}

if WindowEvent is not None:
    # Button mapping uses explicit press/release events for deterministic input pulses.
    PYBOY_BUTTONS: Dict[str, tuple] = {
        "UP": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
        "DOWN": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
        "LEFT": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
        "RIGHT": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
        "A": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        "B": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
        "START": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
        "SELECT": (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
    }
else:  # pragma: no cover - only reachable without PyBoy installed
    PYBOY_BUTTONS = {}


@dataclass
class PyBoyConfig:
    """Runtime configuration for the PyBoyPokemonAdapter."""

    rom_path: str
    window_type: str = "null"
    speed: int = 0  # 0 = unlimited
    max_frame_skip: int = 5
    auto_save_slot: Optional[int] = None
    addresses: Dict[str, int] = field(default_factory=lambda: DEFAULT_ADDRS.copy())


class PyBoyPokemonAdapter(PokemonBlueAdapter):
    """
    Concrete adapter that uses PyBoy to run Pokemon Blue.

    Example:
        cfg = PyBoyConfig(rom_path="PokemonBlue.gb")
        adapter = PyBoyPokemonAdapter(cfg)
        telemetry = adapter.reset()
        telemetry = adapter.step(PokemonAction("WAIT", {"frames": 60}))
    """

    def __init__(self, config: PyBoyConfig) -> None:
        if PyBoy is None:  # pragma: no cover - defensive guard
            raise RuntimeError("pyboy package not found. Install pyboy to use this adapter.")
        self.config = config
        self.pyboy = PyBoy(
            config.rom_path,
            window=config.window_type,
            cgb=False,
        )
        self.pyboy.set_emulation_speed(config.speed)

    # --------------------------------------------------------------------- #
    # PokemonBlueAdapter API
    # --------------------------------------------------------------------- #

    def reset(self) -> PokemonTelemetry:
        # PyBoy doesn't have a reset method, just start fresh
        if self.config.auto_save_slot is not None:
            self.save_state(self.config.auto_save_slot)
        self._tick(120)
        return self._read_telemetry()

    def step(self, action: PokemonAction) -> PokemonTelemetry:
        self._apply_action(action)
        return self._read_telemetry()

    def save_state(self, slot: int = 0) -> None:
        # PyBoy save/load methods
        pass

    def load_state(self, slot: int = 0) -> PokemonTelemetry:
        # PyBoy save/load methods
        self._tick(60)
        return self._read_telemetry()

    # ------------------------------------------------------------------ #
    # Battle adapter bridge
    # ------------------------------------------------------------------ #

    def read_u8(self, address: int) -> int:
        """Return an unsigned byte from emulator memory."""
        return int(self.pyboy.memory[address]) & 0xFF

    def battle_environment(
        self,
        ram_map: Optional["BlueBattleRAMMap"] = None,
    ) -> "BlueRAMAdapter":
        """
        Construct a :class:`BlueRAMAdapter` bound to the underlying PyBoy emulator.
        """

        from src.pkmn_battle.env.blue_ram_adapter import BlueRAMAdapter
        from src.pkmn_battle.env.blue_ram_map import DEFAULT_BLUE_BATTLE_RAM_MAP

        return BlueRAMAdapter(
            read_u8=self.read_u8,
            ram_map=ram_map or DEFAULT_BLUE_BATTLE_RAM_MAP,
            snapshot_extra=lambda: {"frame": int(self.pyboy.frame_count)},
            action_executor=self._execute_battle_action,
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _apply_action(self, action: PokemonAction) -> None:
        name = action.name.upper()
        if name == "SCRIPT":
            script: Iterable[str] = action.payload.get("inputs", [])
            frames_per_input = int(action.payload.get("frames", 4))
            for button in script:
                self._press_button(button, frames_per_input)
        elif name == "WAIT":
            frames = int(action.payload.get("frames", 60))
            self._tick(frames)
        elif name == "BATTLE_MOVE":
            slot = int(action.payload.get("slot", 0))
            self._navigate_battle_menu(slot, is_switch=False)
        elif name == "BATTLE_SWITCH":
            slot = int(action.payload.get("slot", 0))
            self._navigate_battle_menu(slot, is_switch=True)
        else:
            self._press_button(name, int(action.payload.get("frames", 4)))

    def _press_button(self, button_name: str, frames: int) -> None:
        events = PYBOY_BUTTONS.get(button_name.upper())
        if events is None:
            raise ValueError(f"Unknown button '{button_name}'")
        press, release = events
        if press is not None:
            self.pyboy.send_input(press)
        self._tick(frames)
        if release is not None:
            self.pyboy.send_input(release)
        self._tick(2)

    def _tick(self, frames: int) -> None:
        for _ in range(frames):
            self.pyboy.tick()

    def _read_telemetry(self) -> PokemonTelemetry:
        """Read current emulator state and convert to PokemonTelemetry."""
        mem = self.pyboy.memory
        addr = self.config.addresses

        def _get(name: str, required: bool = True) -> Optional[int]:
            offset = addr.get(name)
            if offset is None or offset == 0:
                if required:
                    raise ValueError(
                        f"Memory address for '{name}' is not set. Update PyBoyConfig.addresses with the correct value."
                    )
                return None
            return mem[offset]

        area_id = _get("map_id") or 0
        player_x = _get("player_x") or 0
        player_y = _get("player_y") or 0
        in_grass = bool(_get("in_grass", required=False) or 0)
        in_battle = bool(_get("in_battle", required=False) or 0)
        species_raw = _get("species_id", required=False)
        species_id = species_raw if in_battle else None
        step_val = _get("step_counter", required=False)
        if step_val is None:
            step_counter = int(self.pyboy.frame_count)
        else:
            step_counter = step_val
        elapsed_ms = self.pyboy.frame_count * (1000.0 / 60.0)
        method = "grass" if in_grass else "overworld"

        telemetry = PokemonTelemetry(
            area_id=area_id,
            x=player_x,
            y=player_y,
            in_grass=in_grass,
            in_battle=in_battle,
            encounter_species_id=species_id,
            step_counter=step_counter,
            elapsed_ms=elapsed_ms,
            method=method,
            extra={
                "joy_ignore": int(mem[0xD730]),
                "menu_state": int(mem[0xD122]),
                "menu_cursor": int(mem[0xD13A]),
            },
        )
        return telemetry

    def _execute_battle_action(self, action: Dict[str, object]) -> None:
        kind = action.get("kind")
        if kind == "move":
            slot = int(action.get("index", 0))
            self._navigate_battle_menu(slot, is_switch=False)
        elif kind == "switch":
            slot = int(action.get("index", 0))
            self._navigate_battle_menu(slot, is_switch=True)
        else:
            wait_frames = int(action.get("payload", {}).get("frames", 45))
            self._tick(wait_frames)

    def _is_input_locked(self) -> bool:
        try:
            return bool(self.pyboy.memory[0xD730])
        except Exception:
            return False

    def _wait_until_input_ready(self, timeout_frames: int = 360) -> None:
        for _ in range(max(1, timeout_frames)):
            if not self._is_input_locked():
                return
            self._tick(1)

    def _mash_button(self, button: str, repeats: int = 4, frames: int = 4) -> None:
        for _ in range(max(1, repeats)):
            self._press_button(button, frames)
            self._tick(2)

    def _navigate_battle_menu(self, slot: int, *, is_switch: bool) -> None:
        slot = max(0, slot)
        self._wait_until_input_ready()

        # Enter fight or switch menu
        if is_switch:
            self._press_button("DOWN", 4)
            self._press_button("A", 4)
        else:
            self._press_button("A", 4)

        for _ in range(slot):
            self._press_button("DOWN", 4)

        self._press_button("A", 6)
        self._wait_until_input_ready(90)

        # Dismiss dialog/textboxes by mashing A; fall back to B if needed
        self._mash_button("A", repeats=6, frames=4)
        if self._is_input_locked():
            self._press_button("B", 4)
            self._mash_button("A", repeats=3, frames=4)

    def close(self) -> None:
        try:
            self.pyboy.stop()
        except Exception:
            pass
