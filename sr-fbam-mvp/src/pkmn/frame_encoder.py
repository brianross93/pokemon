"""
Frame encoding utilities for projecting PyBoy telemetry into SR-FBAM inputs.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from src.data.frame_dataset import frame_text_to_tensor
from src.middleware.pokemon_adapter import PokemonTelemetry
from srfbam.core import EncodedFrame

GRID_HEIGHT = 40
GRID_WIDTH = 120


def build_ascii_grid(telemetry: PokemonTelemetry) -> str:
    lines = []
    lines.append(f"Area: {telemetry.area_id:03d} Pos: ({telemetry.x:03d},{telemetry.y:03d})")
    lines.append(
        "InGrass: {ig}  InBattle: {ib}  JoyIgnore: {ji}".format(
            ig=int(telemetry.in_grass),
            ib=int(telemetry.in_battle),
            ji=int(telemetry.extra.get('joy_ignore', 0)),
        )
    )
    lines.append(
        "MenuState: {ms}  MenuCursor: {mc}".format(
            ms=int(telemetry.extra.get('menu_state', 0)),
            mc=int(telemetry.extra.get('menu_cursor', 0)),
        )
    )
    lines.append(f"Species: {telemetry.encounter_species_id or -1}")
    lines.append(f"StepCounter: {telemetry.step_counter}  ElapsedMS: {int(telemetry.elapsed_ms)}")
    lines.append("")
    body = "\n".join(lines)
    body = body[: GRID_HEIGHT * (GRID_WIDTH + 1)]
    return body



def build_numeric_features(telemetry: PokemonTelemetry) -> Tensor:
    x_norm = telemetry.x / 256.0
    y_norm = telemetry.y / 256.0
    area_norm = telemetry.area_id / 512.0
    in_grass = float(telemetry.in_grass)
    in_battle = float(telemetry.in_battle)
    joy_ignore = float(telemetry.extra.get("joy_ignore", 0))
    menu_state = float(telemetry.extra.get("menu_state", 0) / 256.0)
    menu_cursor = float(telemetry.extra.get("menu_cursor", 0) / 256.0)
    features = torch.tensor([[x_norm, y_norm, area_norm, in_grass, in_battle, joy_ignore, menu_state, menu_cursor]], dtype=torch.float32)
    return features


def encode_frame(telemetry: PokemonTelemetry) -> EncodedFrame:
    ascii_text = build_ascii_grid(telemetry)
    grid = frame_text_to_tensor(ascii_text, grid_height=GRID_HEIGHT, grid_width=GRID_WIDTH).to(torch.long)
    features = build_numeric_features(telemetry)
    mode = "battle" if telemetry.in_battle else "overworld"
    context_key = f"area:{telemetry.area_id}:mode:{mode}"
    return EncodedFrame(
        grid=grid,
        features=features,
        context_key=context_key,
        extra={
            "joy_ignore": int(telemetry.extra.get("joy_ignore", 0)),
            "menu_state": int(telemetry.extra.get("menu_state", 0)),
            "menu_cursor": int(telemetry.extra.get("menu_cursor", 0)),
        },
    )
