from __future__ import annotations

import hashlib
import math
import struct
from typing import Iterable, Mapping, Optional, Sequence


PLAN_STRUCTURED_DIM = 20
PLAN_EMBED_DIM = 16
PLAN_FEATURE_DIM = PLAN_STRUCTURED_DIM + PLAN_EMBED_DIM

_SUCCESS_STATES = {"complete", "completed", "done", "success", "resolved"}
_ACTIVE_STATES = {"active", "executing", "in_progress", "running"}
_FAILED_STATES = {"aborted", "failed", "expired", "blocked", "cancelled", "halted"}
_POSITIVE_ADHERENCE = {
    "adhered",
    "on_plan",
    "on_track",
    "match",
    "aligned",
    "perfect",
}
_NEGATIVE_ADHERENCE = {
    "deviated",
    "broken",
    "off_plan",
    "mismatch",
    "failed",
    "aborted",
}


def build_plan_feature_vector(
    plan: Mapping[str, object] | None,
    gate: Mapping[str, object] | None,
    plan_metrics: Mapping[str, object] | None = None,
    plan_adherence: Mapping[str, object] | float | int | str | None = None,
) -> list[float]:
    """
    Construct a fixed-length feature vector summarising the active plan state.

    The vector concatenates structured scalars (progress, gate metadata, adherence)
    with a deterministic text embedding derived from goal/script content so
    downstream models can condition on both plan state and semantics.
    """

    plan = plan or {}
    plan_metrics = plan_metrics or {}
    gate = gate or {}

    plan_status = str(
        plan.get("status")
        or plan_metrics.get("status")
        or ""
    ).lower()
    step_index = _safe_int(
        plan.get("step_index")
        or plan_metrics.get("step_index")
        or plan_metrics.get("plan_step")
        or 0
    )
    steps_total = _safe_int(
        plan.get("steps_total")
        or plan_metrics.get("steps_total")
        or len(_ensure_sequence(plan.get("script")))
        or len(_ensure_sequence(plan_metrics.get("script")))
        or plan.get("horizon_steps")
        or 0
    )

    plan_progress = 0.0
    if steps_total > 0:
        plan_progress = step_index / max(steps_total, 1)
    elif step_index > 0:
        plan_progress = float(step_index)
    plan_progress = max(0.0, min(plan_progress, 1.0))
    plan_remaining = max(0.0, 1.0 - plan_progress)

    steps_total_norm = _normalize_count(steps_total, cap=16)
    step_index_norm = _normalize_count(step_index, cap=max(steps_total, 1))

    horizon_steps = _safe_int(
        plan.get("horizon_steps")
        or plan_metrics.get("horizon_steps")
        or plan.get("timeout_steps")
        or plan_metrics.get("timeout_steps")
        or 0
    )
    horizon_norm = _normalize_count(horizon_steps, cap=64)

    plan_confidence = _safe_float(
        plan.get("plan_confidence")
        or plan.get("confidence")
        or plan_metrics.get("confidence")
        or gate.get("confidence")
        or 0.0
    )
    plan_confidence = max(0.0, min(plan_confidence, 1.0))

    plan_cache_hit = 1.0 if _coerce_bool(plan.get("cache_hit") or plan_metrics.get("cache_hit")) else 0.0
    plan_source = str(plan.get("source") or plan_metrics.get("source") or "").lower()
    plan_source_llm = 1.0 if plan_source == "llm" else 0.0

    gate_mode = str(gate.get("mode") or gate.get("decision") or "").upper()
    gate_confidence = max(0.0, min(_safe_float(gate.get("confidence") or 0.0), 1.0))
    gate_encode_flag = 1.0 if _coerce_bool(gate.get("encode_flag")) else 0.0

    adherence_score, adherence_ok, adherence_bad = _extract_adherence_metrics(
        plan_metrics=plan_metrics,
        plan_adherence=plan_adherence,
    )

    structured = [
        1.0 if plan_status in _ACTIVE_STATES else 0.0,
        1.0 if plan_status in _SUCCESS_STATES else 0.0,
        1.0 if plan_status in _FAILED_STATES else 0.0,
        plan_progress,
        plan_remaining,
        steps_total_norm,
        step_index_norm,
        horizon_norm,
        plan_cache_hit,
        plan_source_llm,
        plan_confidence,
        gate_confidence,
        1.0 if gate_mode == "PLAN_LOOKUP" else 0.0,
        1.0 if gate_mode == "PLAN_STEP" else 0.0,
        1.0 if gate_mode == "ASSOC" else 0.0,
        1.0 if gate_mode in {"WRITE", "ENCODE"} else 0.0,
        gate_encode_flag,
        adherence_score,
        adherence_ok,
        adherence_bad,
    ]

    embedding_tokens = _collect_plan_tokens(plan, plan_metrics, plan_adherence)
    embedding = _hashed_embedding(embedding_tokens, dims=PLAN_EMBED_DIM)

    features = structured + embedding
    if len(features) != PLAN_FEATURE_DIM:
        raise ValueError(
            f"Plan feature construction returned {len(features)} values (expected {PLAN_FEATURE_DIM})."
        )
    return features


def _collect_plan_tokens(
    plan: Mapping[str, object],
    plan_metrics: Mapping[str, object],
    plan_adherence: Mapping[str, object] | float | int | str | None,
) -> list[str]:
    tokens: list[str] = []
    for key in ("plan_id", "id", "planlet_id", "kind", "planlet_kind"):
        value = plan.get(key) or plan_metrics.get(key)
        if isinstance(value, str) and value:
            tokens.append(value.lower())

    for key in ("goal", "rationale", "risk_notes"):
        value = plan.get(key) or plan_metrics.get(key)
        if isinstance(value, str) and value:
            tokens.append(value)

    for constraint in _ensure_sequence(plan.get("constraints")):
        if isinstance(constraint, str) and constraint:
            tokens.append(constraint)

    script = plan.get("script") or plan_metrics.get("script") or []
    if isinstance(script, Mapping):
        script = script.get("steps", [])
    if isinstance(script, Sequence):
        for step in script:
            if not isinstance(step, Mapping):
                continue
            op = str(step.get("op") or step.get("operation") or "").upper()
            if op:
                tokens.append(f"op:{op}")
            actor = step.get("actor")
            if isinstance(actor, str) and actor:
                tokens.append(f"actor:{actor.lower()}")
            move = step.get("move")
            if isinstance(move, str) and move:
                tokens.append(f"move:{move.lower()}")
            trigger = step.get("trigger")
            if isinstance(trigger, str) and trigger:
                tokens.append(trigger)

    if isinstance(plan_adherence, Mapping):
        for key in ("status", "code", "label"):
            value = plan_adherence.get(key)
            if isinstance(value, str) and value:
                tokens.append(value.lower())
    elif isinstance(plan_adherence, str):
        tokens.append(plan_adherence.lower())

    return tokens


def _hashed_embedding(tokens: Sequence[str], *, dims: int) -> list[float]:
    if not tokens:
        return [0.0] * dims

    text = "\n".join(tokens)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=dims * 4).digest()
    embedding = []
    for idx in range(dims):
        chunk = digest[idx * 4 : (idx + 1) * 4]
        value = struct.unpack("<I", chunk)[0]
        scaled = (value / 0xFFFFFFFF) * 2.0 - 1.0
        embedding.append(float(scaled))
    return embedding


def _extract_adherence_metrics(
    *,
    plan_metrics: Mapping[str, object],
    plan_adherence: Mapping[str, object] | float | int | str | None,
) -> tuple[float, float, float]:
    numeric = _first_numeric(
        plan_metrics.get("adherence_score"),
        plan_metrics.get("adherence"),
        plan_metrics.get("plan_adherence"),
        plan_metrics.get("adherence_fraction"),
    )
    if numeric is None and isinstance(plan_adherence, Mapping):
        numeric = _first_numeric(
            plan_adherence.get("score"),
            plan_adherence.get("fraction"),
            plan_adherence.get("value"),
        )
    if numeric is None and isinstance(plan_adherence, (int, float)):
        numeric = float(plan_adherence)

    status = None
    if isinstance(plan_metrics.get("adherence_status"), str):
        status = plan_metrics.get("adherence_status")
    elif isinstance(plan_metrics.get("plan_adherence"), Mapping):
        status = plan_metrics.get("plan_adherence", {}).get("status")
    elif isinstance(plan_adherence, Mapping):
        status = plan_adherence.get("status") or plan_adherence.get("code")
    elif isinstance(plan_adherence, str):
        status = plan_adherence

    if numeric is None:
        if status:
            status_lower = status.lower()
            if status_lower in _POSITIVE_ADHERENCE:
                return 1.0, 1.0, 0.0
            if status_lower in _NEGATIVE_ADHERENCE:
                return 0.0, 0.0, 1.0
        return 0.0, 0.0, 0.0

    score = max(0.0, min(float(numeric), 1.0))
    ok_flag = 1.0 if score >= 0.6 else 0.0
    bad_flag = 1.0 if score <= 0.25 else 0.0
    return score, ok_flag, bad_flag


def _normalize_count(value: int, *, cap: int) -> float:
    if value <= 0:
        return 0.0
    return min(float(value), float(cap)) / float(cap)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return False


def _first_numeric(*values: object) -> Optional[float]:
    for value in values:
        converted = _coerce_float(value)
        if converted is not None:
            return converted
    return None


def _coerce_float(value: object | None) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object, default: float = 0.0) -> float:
    converted = _coerce_float(value)
    if converted is None or math.isnan(converted):
        return default
    return converted


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_sequence(value: object | None) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


__all__ = ["PLAN_FEATURE_DIM", "PLAN_STRUCTURED_DIM", "PLAN_EMBED_DIM", "build_plan_feature_vector"]
