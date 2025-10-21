"""Helpers for building the SR-FBAM imitation-learning datasets."""

from .dataset import BattleDecisionDataset
from .pipeline import ConversionConfig, convert_shard, discover_shards

__all__ = [
    "BattleDecisionDataset",
    "ConversionConfig",
    "convert_shard",
    "discover_shards",
]
