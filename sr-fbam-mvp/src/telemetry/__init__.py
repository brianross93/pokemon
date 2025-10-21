"""
Shared telemetry utilities and schema helpers.
"""

from .parsing import iter_entries, load_entries, normalize_entry
from .schema import DEFAULT_SCHEMA_PATH, load_schema, make_validator

__all__ = [
    "DEFAULT_SCHEMA_PATH",
    "iter_entries",
    "load_entries",
    "load_schema",
    "make_validator",
    "normalize_entry",
]

