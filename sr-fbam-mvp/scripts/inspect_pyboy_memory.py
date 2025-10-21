"""
Utility script to inspect PyBoy memory while running Pokemon Blue.

Usage:
    python scripts/inspect_pyboy_memory.py \
        --rom pokemonblue/"Pokemon Blue.gb" \
        --address map_id=0xD35E --address player_x=0xD361

The script loads the ROM, advances a configurable number of frames, and prints
the requested addresses on every tick. This helps discover the offsets needed
by ``PyBoyPokemonAdapter``.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

try:
    from pyboy import PyBoy
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("pyboy package is required to run this script.") from exc


def parse_address(arg: str) -> tuple[str, int]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"Invalid address '{arg}'. Expected name=0xVALUE.")
    name, value = arg.split("=", 1)
    try:
        offset = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer '{value}' in address spec.") from exc
    return name, offset


def parse_button(arg: str) -> tuple[str, int]:
    if ":" not in arg:
        raise argparse.ArgumentTypeError(f"Invalid button script '{arg}'. Expected BUTTON:FRAMES.")
    button, frames = arg.split(":", 1)
    try:
        frames_int = int(frames)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer '{frames}' in button script.") from exc
    return button.lower(), frames_int


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PyBoy memory locations.")
    parser.add_argument("--rom", required=True, help="Path to Pokemon Blue ROM.")
    parser.add_argument("--frames", type=int, default=120, help="Frames to run before sampling.")
    parser.add_argument(
        "--address",
        action="append",
        type=parse_address,
        default=[],
        help="Address specifier in the form name=0xOFFSET (can be repeated).",
    )
    parser.add_argument(
        "--pre-script",
        action="append",
        type=parse_button,
        default=[],
        help="Button sequence BEFORE sampling (BUTTON:FRAMES). E.g. start:60,A:40",
    )
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between samples.")
    parser.add_argument("--iterations", type=int, default=20, help="How many samples to record.")
    args = parser.parse_args()

    addresses: Dict[str, int] = dict(args.address)
    if not addresses:
        raise SystemExit("At least one --address name=0xOFFSET pair is required.")

    pyboy = PyBoy(args.rom, window="null", cgb=False)
    pyboy.set_emulation_speed(0)  # unlimited

    def press(button: str, frames: int) -> None:
        if button == "wait":
            for _ in range(frames):
                pyboy.tick()
            return
        pyboy.button(button, True)
        for _ in range(frames):
            pyboy.tick()
        pyboy.button(button, False)
        for _ in range(4):
            pyboy.tick()

    # Warm up frames to let the game reach a consistent state.
    for _ in range(args.frames):
        pyboy.tick()

    for button, frames in args.pre_script:
        press(button, frames)

    memory = pyboy.memory
    print("Inspecting addresses:", ", ".join(f"{name}=0x{offset:04X}" for name, offset in addresses.items()))
    print("Press Ctrl+C to stop.")

    try:
        for iteration in range(args.iterations):
            values = {name: memory[offset] for name, offset in addresses.items()}
            formatted = " ".join(f"{name}={value:03d}(0x{value:02X})" for name, value in values.items())
            print(f"[{iteration:02d}] {formatted}")
            for _ in range(60):
                pyboy.tick()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        pyboy.stop()


if __name__ == "__main__":
    main()
