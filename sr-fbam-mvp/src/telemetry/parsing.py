"""
Normalization helpers for SR-FBAM telemetry logs.

The project now emits telemetry records with the shape::

    {
        "source": "...",
        "context": {"domain": "...", ...},
        "observation": {...} | null,
        "telemetry": {
            "core": {...},
            "overworld": {...},
            "battle": {...},
        },
    }

Older logs stored battle and overworld payloads using flat schemas.  This
module provides utilities that lift both legacy structures into the new
namespaces so downstream analytics can consume a single stable format.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

JsonDict = Dict[str, object]


def iter_entries(path: Path | str) -> Iterator[JsonDict]:
    """
    Yield normalised telemetry entries from a JSONL file.

    Parameters
    ----------
    path:
        Path to a JSON Lines file containing telemetry records (legacy or
        consolidated schema).
    """

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield normalize_entry(raw)


def load_entries(path: Path | str) -> list[JsonDict]:
    """Return a list of normalised telemetry entries from ``path``."""

    return list(iter_entries(path))


def normalize_entry(entry: Mapping[str, object]) -> JsonDict:
    """
    Upgrade a telemetry payload to the consolidated schema.

    The function accepts three shapes:

    * New schema (already namespaced) - returned with minimal touch-ups.
    * Legacy overworld executor entries written by ``OverworldTraceRecorder``.
    * Legacy battle agent entries recorded by ``scripts/run_battle_agent.py``.
    """

    if _looks_like_new_schema(entry):
        return _normalize_new_schema(entry)
    if _looks_like_legacy_battle(entry):
        return _normalize_legacy_battle(entry)
    if _looks_like_legacy_overworld(entry):
        return _normalize_legacy_overworld(entry)
    raise ValueError("Unrecognised telemetry payload shape")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _looks_like_new_schema(entry: Mapping[str, object]) -> bool:
    telemetry = entry.get("telemetry")
    return isinstance(telemetry, Mapping) and isinstance(telemetry.get("core"), Mapping)


def _looks_like_legacy_battle(entry: Mapping[str, object]) -> bool:
    return (
        "gate" in entry
        and "fractions" in entry
        and "speedup" in entry
        and "turn" in entry
    )


def _looks_like_legacy_overworld(entry: Mapping[str, object]) -> bool:
    telemetry = entry.get("telemetry")
    return (
        isinstance(telemetry, Mapping)
        and "gate" in telemetry
        and "planlet_id" in entry
        and "planlet_kind" in entry
    )


def _normalise_gate_payload(gate: Mapping[str, object] | None) -> Dict[str, object]:
    gate = gate or {}
    mode = gate.get("mode", gate.get("op"))
    normalised = {
        "mode": mode,
        "encode_flag": gate.get("encode_flag"),
        "view": gate.get("view"),
        "confidence": gate.get("confidence"),
        "reason": gate.get("reason"),
        "raw": gate.get("raw", mode),
    }
    stats = gate.get("stats")
    if isinstance(stats, Mapping):
        normalised["stats"] = dict(stats)
    return normalised


def _ensure_core_defaults(core: MutableMapping[str, object]) -> None:
    core.setdefault("legal_actions", [])
    core.setdefault("action", None)
    core.setdefault("action_mask", None)
    core.setdefault("fractions", {})
    core.setdefault("speedup", {})
    core.setdefault("hop_trace", [])
    core.setdefault("fallback_required", False)


def _normalize_new_schema(entry: Mapping[str, object]) -> JsonDict:
    normalised = deepcopy(entry)
    normalised.setdefault("source", "unknown.telemetry")

    telemetry = normalised.setdefault("telemetry", {})
    if not isinstance(telemetry, MutableMapping):
        raise ValueError("Invalid telemetry payload")

    core = telemetry.setdefault("core", {})
    if not isinstance(core, MutableMapping):
        raise ValueError("Invalid telemetry core payload")
    core["gate"] = _normalise_gate_payload(core.get("gate"))
    _ensure_core_defaults(core)

    context = normalised.setdefault("context", {})
    if not isinstance(context, MutableMapping):
        raise ValueError("Invalid telemetry context payload")
    context.setdefault("domain", "unknown")

    return normalised  # type: ignore[return-value]


def _normalize_legacy_battle(entry: Mapping[str, object]) -> JsonDict:
    gate = _normalise_gate_payload(entry.get("gate"))  # type: ignore[arg-type]
    fractions = entry.get("fractions") if isinstance(entry.get("fractions"), Mapping) else {}
    speedup = entry.get("speedup") if isinstance(entry.get("speedup"), Mapping) else {}
    battle_payload: Dict[str, object] = {
        "writes": list(entry.get("writes", [])),
    }
    if "index_map" in entry:
        battle_payload["index_map"] = entry["index_map"]

    core = {
        "legal_actions": list(entry.get("legal_actions", [])),
        "action": entry.get("action"),
        "action_mask": entry.get("action_mask"),
        "gate": gate,
        "fractions": dict(fractions),
        "speedup": dict(speedup),
        "latency_ms": entry.get("latency_ms"),
        "fallback_required": bool(entry.get("fallback_required", False)),
        "hop_trace": list(entry.get("hop_trace", [])),
    }
    _ensure_core_defaults(core)

    context = {
        "domain": "battle",
        "battle": {
            "turn": int(entry.get("turn", 0) or 0),
            "step": int(entry.get("step", 0) or 0),
        },
    }

    return {
        "source": "legacy.battle",
        "context": context,
        "observation": entry.get("observation"),
        "telemetry": {
            "core": core,
            "battle": battle_payload,
        },
    }


def _normalize_legacy_overworld(entry: Mapping[str, object]) -> JsonDict:
    telemetry = entry.get("telemetry", {})
    if not isinstance(telemetry, Mapping):
        raise ValueError("Legacy overworld telemetry was not a mapping")

    gate = _normalise_gate_payload(telemetry.get("gate"))  # type: ignore[arg-type]
    fractions = telemetry.get("fractions") if isinstance(telemetry.get("fractions"), Mapping) else {}
    speedup = telemetry.get("speedup") if isinstance(telemetry.get("speedup"), Mapping) else {}

    core = {
        "legal_actions": [dict(action) for action in telemetry.get("legal_actions", [])],
        "action": telemetry.get("action"),
        "action_mask": telemetry.get("action_mask"),
        "gate": gate,
        "fractions": dict(fractions),
        "speedup": dict(speedup),
        "latency_ms": telemetry.get("latency_ms"),
        "fallback_required": bool(telemetry.get("fallback_required", False)),
        "hop_trace": list(telemetry.get("hop_trace", [])),
    }
    _ensure_core_defaults(core)

    overworld_payload: Dict[str, object] = {}
    for key in (
        "action_index",
        "frame_features",
        "memory",
        "view_usage",
        "hybrid",
        "navigate",
        "encounter",
        "recovery",
        "skill",
    ):
        if key in telemetry:
            overworld_payload[key] = telemetry[key]

    context = {
        "domain": "overworld",
        "plan": {
            "id": entry.get("plan_id"),
            "planlet_id": entry.get("planlet_id"),
            "planlet_kind": entry.get("planlet_kind"),
        },
        "status": entry.get("status"),
        "step_index": int(entry.get("step_index", 0) or 0),
    }

    return {
        "source": "legacy.overworld",
        "context": context,
        "observation": entry.get("observation"),
        "telemetry": {
            "core": core,
            "overworld": overworld_payload,
        },
    }


__all__ = ["iter_entries", "load_entries", "normalize_entry"]

