"""
Utility for discovering the correct overworld RAM addresses for a given Pokemon Blue ROM.

The tool captures multiple memory snapshots from the PyBoy emulator and highlights bytes
that change between captures. Use it to identify where the current map ID, player X/Y,
grass/battle flags, etc. live for your specific ROM revision.

Examples:
    # Manual mode (you control the character between captures)
    python scripts/debug_overworld_addresses.py --rom "Pokemon\\ Blue.gb" --window SDL2 --captures 4

    # Auto mode (the script pulses directional inputs)
    python scripts/debug_overworld_addresses.py --rom "Pokemon\\ Blue.gb" --window SDL2 --auto
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import threading
import time

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter
    from src.middleware.pokemon_adapter import PokemonAction
except RuntimeError as exc:  # pragma: no cover - PyBoy missing at import time
    raise SystemExit(f"PyBoy is required for this script: {exc}") from exc


def parse_int(value: str) -> int:
    """Parse an integer argument accepting decimal or hex (0x-prefixed)."""
    return int(value, 0)


def parse_sequence(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff PyBoy RAM snapshots to locate overworld state addresses.")
    parser.add_argument("--rom", type=str, required=True, help="Path to the Pokemon Blue ROM.")
    parser.add_argument("--window", type=str, default="SDL2", help="PyBoy window type (use SDL2 for visible window).")
    parser.add_argument(
        "--start-addr",
        type=parse_int,
        default=0xD000,
        help="Start address (inclusive) for the memory scan (default: 0xD000).",
    )
    parser.add_argument(
        "--end-addr",
        type=parse_int,
        default=0xD3FF,
        help="End address (inclusive) for the memory scan (default: 0xD3FF).",
    )
    parser.add_argument(
        "--captures",
        type=int,
        default=4,
        help="Number of snapshots to capture in manual mode (baseline + additional positions).",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=60,
        help="Frames to advance after each capture/action to let memory stabilise.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable scripted directional inputs instead of prompting the user.",
    )
    parser.add_argument(
        "--sequence",
        type=parse_sequence,
        default=["RIGHT", "DOWN", "LEFT", "UP"],
        help="Comma-separated directional sequence for auto mode (e.g., RIGHT,RIGHT,DOWN).",
    )
    parser.add_argument(
        "--frames-per-move",
        type=int,
        default=12,
        help="Frames to hold each directional input in auto mode (default: 12).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many candidate addresses to display (sorted by change frequency).",
    )
    return parser.parse_args()


@dataclass
class Snapshot:
    label: str
    frame: int
    memory: bytes


def _tick(pyboy: "PyBoyPokemonAdapter", frames: int) -> None:
    for _ in range(max(0, frames)):
        pyboy.pyboy.tick()


def _read_memory(adapter: PyBoyPokemonAdapter, start: int, end: int) -> bytes:
    mem = adapter.pyboy.memory
    return bytes(mem[start : end + 1])


class _TickThread:
    def __init__(self, adapter: PyBoyPokemonAdapter) -> None:
        self._adapter = adapter
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._adapter.pyboy.tick()
            time.sleep(0.0)


def capture_manual(
    adapter: PyBoyPokemonAdapter,
    *,
    captures: int,
    start: int,
    end: int,
    settle_frames: int,
) -> List[Snapshot]:
    print("[manual] Resetting emulator…")
    adapter.reset()
    snapshots: List[Snapshot] = []
    for idx in range(captures):
        label = "baseline" if idx == 0 else f"capture_{idx}"
        prompt = (
            f"[manual] Position your character for '{label}'. "
            "Use the PyBoy window (if enabled), then press Enter to record."
        )
        ticker = _TickThread(adapter)
        ticker.start()
        input(prompt)
        ticker.stop()
        _tick(adapter, settle_frames)
        snapshots.append(
            Snapshot(
                label=label,
                frame=int(adapter.pyboy.frame_count),
                memory=_read_memory(adapter, start, end),
            )
        )
        print(f"[manual] Recorded snapshot '{label}' at frame {snapshots[-1].frame}.")
    return snapshots


def capture_auto(
    adapter: PyBoyPokemonAdapter,
    *,
    start: int,
    end: int,
    settle_frames: int,
    sequence: Sequence[str],
    frames_per_move: int,
) -> List[Snapshot]:
    print("[auto] Resetting emulator…")
    adapter.reset()
    snapshots: List[Snapshot] = [
        Snapshot("baseline", int(adapter.pyboy.frame_count), _read_memory(adapter, start, end))
    ]

    for step_idx, direction in enumerate(sequence, start=1):
        label = f"after_{direction.lower()}_{step_idx}"
        if direction == "WAIT":
            adapter.step(PokemonAction("WAIT", {"frames": frames_per_move}))
        else:
            action = PokemonAction("SCRIPT", {"inputs": [direction], "frames": frames_per_move})
            adapter.step(action)
        _tick(adapter, settle_frames)
        snapshots.append(
            Snapshot(
                label=label,
                frame=int(adapter.pyboy.frame_count),
                memory=_read_memory(adapter, start, end),
            )
        )
        print(f"[auto] Recorded snapshot '{label}' at frame {snapshots[-1].frame}.")
    return snapshots


def analyse_snapshots(
    snapshots: Sequence[Snapshot],
    *,
    start: int,
    top_k: int,
) -> List[Tuple[int, List[int]]]:
    if not snapshots:
        return []
    width = len(snapshots[0].memory)
    addresses = [start + offset for offset in range(width)]
    value_matrix = {addr: [] for addr in addresses}
    for snap in snapshots:
        if len(snap.memory) != width:
            raise ValueError("Snapshot memory lengths do not match; ensure start/end addresses remain constant.")
        for offset, byte in enumerate(snap.memory):
            addr = addresses[offset]
            value_matrix[addr].append(byte)

    candidates: List[Tuple[int, List[int]]] = []
    for addr, series in value_matrix.items():
        if len(set(series)) <= 1:
            continue
        candidates.append((addr, series))

    def sort_key(item: Tuple[int, List[int]]) -> Tuple[int, int]:
        _, series = item
        unique_count = len(set(series))
        total_delta = sum(abs(series[idx] - series[idx - 1]) for idx in range(1, len(series)))
        return (unique_count, total_delta)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[:top_k]


def print_report(
    snapshots: Sequence[Snapshot],
    candidates: Sequence[Tuple[int, List[int]]],
) -> None:
    labels = [snap.label for snap in snapshots]
    header = "Addr  | " + " | ".join(f"{label:>14}" for label in labels)
    print("\n=== Candidate Addresses ===")
    print(header)
    print("-" * len(header))
    for addr, series in candidates:
        series_str = " | ".join(f"{value:>14}" for value in series)
        print(f"0x{addr:04X} | {series_str}")
    print("\nTip: Look for addresses that track tile-level movement or change when entering grass/battles.")


def main() -> int:
    args = parse_args()
    if args.start_addr > args.end_addr:
        raise SystemExit("start-addr must be <= end-addr")

    cfg = PyBoyConfig(
        rom_path=args.rom,
        window_type=args.window,
        speed=0,
        debug_addresses=False,
    )
    adapter = PyBoyPokemonAdapter(cfg)

    try:
        if args.auto:
            snapshots = capture_auto(
                adapter,
                start=args.start_addr,
                end=args.end_addr,
                settle_frames=args.settle_frames,
                sequence=args.sequence,
                frames_per_move=args.frames_per_move,
            )
        else:
            snapshots = capture_manual(
                adapter,
                captures=args.captures,
                start=args.start_addr,
                end=args.end_addr,
                settle_frames=args.settle_frames,
            )

        candidates = analyse_snapshots(snapshots, start=args.start_addr, top_k=args.top)
        if not candidates:
            print("No varying addresses detected in the selected range. Try expanding the scan window or moving further.")
            return 0
        print_report(snapshots, candidates)
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
