"""
Frame encoding utilities for projecting PyBoy framebuffer observations into SR-FBAM inputs.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from src.overworld.env.overworld_adapter import OverworldObservation
from src.srfbam.core import EncodedFrame

GRID_HEIGHT = 40
GRID_WIDTH = 120


def _ensure_frame_dimensions(framebuffer: np.ndarray) -> np.ndarray:
    if framebuffer.shape[0] != GRID_HEIGHT or framebuffer.shape[1] != GRID_WIDTH:
        raise ValueError(
            f"Expected framebuffer shape ({GRID_HEIGHT}, {GRID_WIDTH}, 3); got {framebuffer.shape!r}"
        )
    if framebuffer.ndim != 3 or framebuffer.shape[2] != 3:
        raise ValueError("Framebuffer must be an RGB array with shape (H, W, 3).")
    return framebuffer


def build_intensity_grid(framebuffer: np.ndarray) -> Tensor:
    """Convert RGB framebuffer into a 40x120 intensity grid."""
    frame = _ensure_frame_dimensions(framebuffer)
    grayscale = np.dot(frame[..., :3].astype(np.float32), np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
    normalized = grayscale / 255.0
    tensor = torch.from_numpy(normalized).to(torch.float32)
    grid = (tensor * 255.0).round().clamp(0, 255).to(torch.long)
    return grid


def build_numeric_features(framebuffer: np.ndarray, metadata: Mapping[str, object]) -> Tensor:
    """Compute lightweight numeric features from the framebuffer and metadata."""
    frame = _ensure_frame_dimensions(framebuffer)
    flat = frame.reshape(-1, 3).astype(np.float32) / 255.0
    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0)
    map_id = str(metadata.get("map_id", "unknown"))
    menu_flag = 1.0 if metadata.get("is_menu") or metadata.get("menu_open") else 0.0
    cursor_norm = float(metadata.get("menu_cursor", 0)) / 8.0
    map_token = (hash(map_id) % 1024) / 1023.0
    feature_vector = np.array(
        [
            mean_rgb[0],
            mean_rgb[1],
            mean_rgb[2],
            std_rgb[0],
            std_rgb[1],
            std_rgb[2],
            map_token,
            np.clip(menu_flag + cursor_norm, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(feature_vector).unsqueeze(0)


def encode_observation(observation: OverworldObservation) -> EncodedFrame:
    """Project an overworld observation into SR-FBAM's EncodedFrame structure."""
    framebuffer = observation.framebuffer
    metadata = observation.metadata or {}
    grid = build_intensity_grid(framebuffer)
    features = build_numeric_features(framebuffer, metadata)
    map_id = str(metadata.get("map_id", "unknown"))
    context_key = metadata.get("context_key") or f"overworld:map:{map_id}"
    extra = {
        "frame_hash": metadata.get("frame_hash"),
        "timestamp_ms": metadata.get("timestamp_ms"),
    }
    return EncodedFrame(grid=grid, features=features, context_key=context_key, extra=extra)
