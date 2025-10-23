"""
PyBoy-based implementation of the PokemonBlueAdapter interface.

This module provides a thin bridge between the PyBoy emulator and the symbolic
middleware. Memory addresses are left as placeholders that must be populated
with the offsets corresponding to the ROM in use. See the documentation in
`KNOWLEDGE_MIDDLEWARE_PLAN.md` for workflow guidance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    PyBoy = None  # type: ignore[assignment]
    WindowEvent = None  # type: ignore[assignment]

from src.middleware.pokemon_adapter import ObservationBundle, PokemonAction, PokemonBlueAdapter
from src.overworld.ram_map import DEFAULT_OVERWORLD_RAM_MAP

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.pkmn_battle.env.blue_ram_adapter import BlueBattleRAMMap, BlueRAMAdapter


DEFAULT_ADDRS: Dict[str, int] = {
    # Updated based on RAM-diff analysis - addresses that change during movement
    "map_id": 0xD35E,  # wCurMap (keep original, test if it changes between areas)
    "player_x": 0xD13A,  # Found via RAM-diff - shows clear movement pattern
    "player_y": 0xD122,  # Found via RAM-diff - changes with movement
    "in_grass": 0xD5B5,  # wGrassCollision bitfield (non-zero when standing in grass)
    "in_battle": 0xD057,  # wIsInBattle
    "species_id": 0xD0B5,  # wEnemyMonSpecies
    "step_counter": 0xD31C,  # wStepCounter
    "rng_seed_high": 0x0000,  # Populated once verified via debug_overworld_addresses.py
    "rng_seed_low": 0x0000,
}

DEFAULT_FRAME_SHAPE: Tuple[int, int] = (40, 120)

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
    speed: float = 0.0  # 0 = unlimited
    max_frame_skip: int = 5
    auto_save_slot: Optional[int] = None
    addresses: Dict[str, int] = field(default_factory=lambda: DEFAULT_ADDRS.copy())
    debug_addresses: bool = False
    frame_shape: Tuple[int, int] = DEFAULT_FRAME_SHAPE
    capture_ram: bool = True


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
        self._bootstrapped = False
        self._frame_shape = self._normalise_frame_shape(config.frame_shape)
        self._screen_api = self._initialise_screen_api()

    def _normalise_frame_shape(self, shape: Tuple[int, int] | Iterable[int]) -> Tuple[int, int]:
        if not shape:
            return DEFAULT_FRAME_SHAPE
        try:
            height, width = shape  # type: ignore[misc]
        except Exception:  # pragma: no cover - defensive
            return DEFAULT_FRAME_SHAPE
        try:
            h = max(1, int(height))
            w = max(1, int(width))
        except Exception:  # pragma: no cover - defensive
            return DEFAULT_FRAME_SHAPE
        return h, w

    def _initialise_screen_api(self) -> Optional[Any]:
        manager_factory = getattr(self.pyboy, "botsupport_manager", None)
        if manager_factory is None:
            return None
        try:
            manager = manager_factory()
            screen = getattr(manager, "screen", lambda: None)()
        except Exception:  # pragma: no cover - defensive
            return None
        return screen

    # --------------------------------------------------------------------- #
    # PokemonBlueAdapter API
    # --------------------------------------------------------------------- #

    def reset(self) -> ObservationBundle:
        # PyBoy doesn't have a reset method, just start fresh
        if self.config.auto_save_slot is not None:
            self.save_state(self.config.auto_save_slot)
        self._tick(120)
        self._ensure_bootstrapped()
        return self._capture_observation()

    def step(self, action: PokemonAction) -> ObservationBundle:
        self._apply_action(action)
        return self._capture_observation()

    def save_state(self, slot: int = 0) -> None:
        try:
            self.pyboy.save_state(slot)
        except AttributeError as exc:  # pragma: no cover - depends on PyBoy build
            raise RuntimeError("PyBoy build does not support save_state.") from exc

    def load_state(self, slot: int = 0) -> ObservationBundle:
        try:
            self.pyboy.load_state(slot)
        except AttributeError as exc:  # pragma: no cover - depends on PyBoy build
            raise RuntimeError("PyBoy build does not support load_state.") from exc
        self._tick(60)
        self._ensure_bootstrapped()
        return self._capture_observation()

    def observe(self) -> ObservationBundle:
        """Capture the current observation without mutating emulator state."""
        return self._capture_observation()

    def snapshot_overworld_ram(self) -> Dict[int, int]:
        """Return a mapping of key overworld RAM addresses -> values."""

        if not self.config.capture_ram:
            return {}

        mapping: Dict[int, int] = {}
        mem = self.pyboy.memory

        def read(addr: int) -> int:
            try:
                return int(mem[addr]) & 0xFF
            except (IndexError, ValueError):  # pragma: no cover - defensive
                return 0

        for name, address in DEFAULT_OVERWORLD_RAM_MAP.items():
            value = read(address)
            mapping[address] = value
            if name == "warp_table":
                count_addr = DEFAULT_OVERWORLD_RAM_MAP.get("warp_count")
                warp_count = read(count_addr) if count_addr is not None else 0
                warp_count = max(0, min(8, warp_count))
                for index in range(warp_count):
                    base = address + index * 4
                    for offset in range(4):
                        mapping[base + offset] = read(base + offset)
            elif name == "npc_table":
                count_addr = DEFAULT_OVERWORLD_RAM_MAP.get("npc_count")
                npc_count = read(count_addr) if count_addr is not None else 0
                npc_count = max(0, min(16, npc_count))
                for index in range(npc_count):
                    base = address + index * 4
                    for offset in range(4):
                        mapping[base + offset] = read(base + offset)

        return mapping

    def set_rng_seed(self, seed: int) -> None:
        """
        Attempt to write a deterministic RNG seed into emulator memory.

        The default addresses are placeholders; override PyBoyConfig.addresses with the
        correct offsets exposed by debug_overworld_addresses.py.
        """
        seed = int(seed) & 0xFFFF
        high = (seed >> 8) & 0xFF
        low = seed & 0xFF
        addr = self.config.addresses
        memory = self.pyboy.memory
        high_addr = addr.get("rng_seed_high", 0)
        low_addr = addr.get("rng_seed_low", 0)
        if high_addr:
            memory[high_addr] = high
        if low_addr:
            memory[low_addr] = low

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

    # ------------------------------------------------------------------ #
    # Frame capture helpers
    # ------------------------------------------------------------------ #

    def _capture_observation(self) -> ObservationBundle:
        raw_frame = self._grab_raw_frame()
        framebuffer = self._downsample_frame(raw_frame)
        ram_snapshot = self.snapshot_overworld_ram()
        metadata = self._build_metadata(framebuffer, raw_frame)
        observation_ram: Optional[Dict[int, int]] = ram_snapshot or None
        return ObservationBundle(
            framebuffer=framebuffer,
            raw_framebuffer=raw_frame,
            ram=observation_ram,
            metadata=metadata,
        )

    def _grab_raw_frame(self) -> Optional[np.ndarray]:
        raw_frame: Optional[np.ndarray] = None
        try:
            image = getattr(self.pyboy, "screen").image
            if image is not None:
                raw_frame = np.array(image)
        except Exception:  # pragma: no cover - defensive
            raw_frame = None

        if raw_frame is None:
            return None

        frame_array = np.asarray(raw_frame)
        if frame_array.ndim == 2:
            frame_array = np.repeat(frame_array[:, :, None], 3, axis=2)
        elif frame_array.ndim == 3 and frame_array.shape[2] == 4:
            frame_array = frame_array[:, :, :3]
        return frame_array.astype(np.uint8, copy=False)

    def _downsample_frame(self, frame: Optional[np.ndarray]) -> np.ndarray:
        target_h, target_w = self._frame_shape
        if frame is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if height == target_h and width == target_w:
            return frame.copy()
        y_idx = np.linspace(0, height - 1, target_h).astype(int)
        x_idx = np.linspace(0, width - 1, target_w).astype(int)
        sampled = np.take(frame, y_idx, axis=0)
        sampled = np.take(sampled, x_idx, axis=1)
        return sampled.astype(np.uint8, copy=True)

    def _build_metadata(self, framebuffer: np.ndarray, raw_frame: Optional[np.ndarray]) -> Dict[str, Any]:
        frame_index = int(self.pyboy.frame_count)
        timestamp_ms = frame_index * (1000.0 / 60.0)
        metadata = {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "frame_hash": self._hash_frame(framebuffer),
            "frame_shape": tuple(int(dim) for dim in framebuffer.shape[:2]),
            "source": "pyboy",
        }
        if raw_frame is not None:
            metadata["raw_frame_shape"] = tuple(int(dim) for dim in raw_frame.shape[:2])
            metadata["raw_frame_hash"] = self._hash_frame(raw_frame)
        return metadata

    @staticmethod
    def _hash_frame(framebuffer: np.ndarray) -> str:
        if framebuffer.size == 0:
            return "0" * 40
        digest = hashlib.sha1(framebuffer.tobytes())  # nosec: non-cryptographic usage
        return digest.hexdigest()

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

    def _ensure_bootstrapped(self) -> None:
        """
        Make sure the emulator has advanced past the title screen so telemetry reflects overworld state.

        We only need to run this once per emulator lifetime; subsequent resets would otherwise open the pause menu.
        """
        if self._bootstrapped:
            return
        # Send START and A pulses to clear intro/title, allowing overworld control.
        self._press_button("START", 6)
        self._tick(90)
        self._press_button("A", 6)
        self._tick(90)
        self._bootstrapped = True

    def _tick(self, frames: int) -> None:
        for _ in range(frames):
            self.pyboy.tick()

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
