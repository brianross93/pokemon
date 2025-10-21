"""
Utilities for validating LLM-generated planlets against the schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

from jsonschema import Draft7Validator, ValidationError

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "planlet.json"


class PlanValidationError(RuntimeError):
    """Raised when the LLM output does not satisfy the planlet schema."""

    def __init__(self, message: str, *, errors: Optional[Sequence[str]] = None) -> None:
        super().__init__(message)
        self.errors = list(errors or [])


@dataclass(frozen=True)
class PlanletSpec:
    """Typed representation of a single planlet."""

    id: str
    kind: str
    args: Dict[str, Any] = field(default_factory=dict)
    pre: Sequence[Dict[str, Any]] = field(default_factory=list)
    post: Sequence[Dict[str, Any]] = field(default_factory=list)
    hints: Dict[str, Any] = field(default_factory=dict)
    timeout_steps: int = 600
    recovery: Sequence[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PlanBundle:
    """Collection of planlets that fulfils a single high-level goal."""

    plan_id: str
    goal: Optional[str]
    planlets: List[PlanletSpec]
    raw: Dict[str, Any] = field(default_factory=dict)


def _load_schema() -> Draft7Validator:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    return Draft7Validator(schema)


_PLANLET_VALIDATOR = _load_schema()


def validate_plan_bundle(data: Mapping[str, Any]) -> PlanBundle:
    """
    Validate a mapping produced by the planner LLM.

    Args:
        data: Mapping containing plan metadata and planlets (typically parsed JSON).

    Returns:
        A ``PlanBundle`` instance if validation succeeds.

    Raises:
        PlanValidationError: When the payload fails schema or structural validation.
    """

    if not isinstance(data, Mapping):
        raise PlanValidationError("Plan bundle must be a mapping.")

    planlets = data.get("planlets")
    if not isinstance(planlets, Sequence) or isinstance(planlets, (str, bytes)):
        raise PlanValidationError("Plan bundle must contain a list of planlets under 'planlets'.")
    if len(planlets) == 0:
        raise PlanValidationError("Plan bundle contained no planlets.")

    plan_id = data.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        raise PlanValidationError("Plan bundle requires non-empty string 'plan_id'.")

    goal = data.get("goal")
    if goal is not None and not isinstance(goal, str):
        raise PlanValidationError("'goal' must be a string when provided.")

    seen_ids: set[str] = set()
    typed_planlets: List[PlanletSpec] = []
    validation_errors: List[str] = []

    for index, planlet in enumerate(planlets):
        if not isinstance(planlet, MutableMapping):
            validation_errors.append(f"Planlet {index} is not an object.")
            continue

        try:
            _PLANLET_VALIDATOR.validate(planlet)
        except ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path)
            pointer = f"planlets[{index}]" + (f".{path}" if path else "")
            validation_errors.append(f"{pointer}: {exc.message}")
            continue

        planlet_id = str(planlet["id"])
        if planlet_id in seen_ids:
            validation_errors.append(f"Duplicate planlet id '{planlet_id}'.")
            continue
        seen_ids.add(planlet_id)

        typed_planlets.append(
            PlanletSpec(
                id=planlet_id,
                kind=str(planlet["kind"]),
                args=dict(planlet.get("args") or {}),
                pre=list(planlet.get("pre") or []),
                post=list(planlet.get("post") or []),
                hints=dict(planlet.get("hints") or {}),
                timeout_steps=int(planlet.get("timeout_steps", 600)),
                recovery=list(planlet.get("recovery") or []),
            )
        )

    if validation_errors:
        raise PlanValidationError("Plan bundle failed validation.", errors=validation_errors)

    return PlanBundle(plan_id=plan_id, goal=goal, planlets=typed_planlets, raw=dict(data))


def validate_plan_json(raw: Union[str, bytes]) -> PlanBundle:
    """
    Parse JSON text produced by an LLM and validate it against the schema.
    """

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise PlanValidationError(f"Unable to parse plan JSON: {exc}") from exc
    return validate_plan_bundle(payload)


def load_plan_bundle(path: Union[str, Path]) -> PlanBundle:
    """
    Load and validate a plan bundle from disk.
    """

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_plan_bundle(payload)


def format_validation_errors(errors: Iterable[str]) -> str:
    """
    Helper that turns validation errors into a compact bullet list for logging or LLM feedback.
    """

    lines = [f"- {error}" for error in errors]
    return "\n".join(lines)
