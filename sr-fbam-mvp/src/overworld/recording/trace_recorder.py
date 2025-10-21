"""
JSONL trace recorder for overworld executor telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Mapping, Optional

from jsonschema import Draft7Validator, ValidationError

from src.telemetry import DEFAULT_SCHEMA_PATH as TELEMETRY_SCHEMA_PATH, make_validator


class TraceValidationError(RuntimeError):
    """Raised when a trace payload fails schema validation."""

    def __init__(self, message: str, *, errors: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.errors = list(errors or [])


DEFAULT_SCHEMA_PATH = TELEMETRY_SCHEMA_PATH


class OverworldTraceRecorder:
    """
    Writes per-step overworld traces with optional JSON schema validation.

    Parameters
    ----------
    path:
        Output file path. Parent directories are created on demand.
    schema_path:
        Optional override for the JSON schema file.
    validate:
        When ``True`` (default) each payload is validated before writing.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema_path: Optional[Path] = None,
        validate: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = self.path.open("w", encoding="utf-8")
        self._validator: Optional[Draft7Validator] = None
        if validate:
            schema_file = schema_path or DEFAULT_SCHEMA_PATH
            self._validator = make_validator(schema_file)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record(self, payload: Mapping[str, object]) -> None:
        """
        Append a single trace payload to disk.
        """

        if self._validator is not None:
            try:
                self._validator.validate(payload)
            except ValidationError as exc:  # pragma: no cover - defensive
                raise TraceValidationError(str(exc), errors=[exc.message]) from exc

        json.dump(payload, self._handle, ensure_ascii=False)
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file handle."""

        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "OverworldTraceRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["OverworldTraceRecorder", "TraceValidationError", "DEFAULT_SCHEMA_PATH"]
