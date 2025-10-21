"""
Helpers for loading the shared SR-FBAM telemetry JSON schema.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft7Validator

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "telemetry_entry.json"


@lru_cache(maxsize=1)
def load_schema(path: Path | None = None) -> Dict[str, Any]:
    """
    Load the consolidated telemetry schema as a Python dictionary.
    """

    schema_path = path or DEFAULT_SCHEMA_PATH
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_validator(path: Path | None = None) -> Draft7Validator:
    """
    Construct a Draft7 validator for the consolidated telemetry schema.
    """

    schema = load_schema(path)
    return Draft7Validator(schema)


__all__ = ["DEFAULT_SCHEMA_PATH", "load_schema", "make_validator"]
