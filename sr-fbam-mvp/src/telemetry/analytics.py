"""
Telemetry analytics helpers for dashboards and parity checks.

The helpers aggregate over normalised telemetry entries (see
``src.telemetry.parsing``) and surface summary metrics that previously
relied on the legacy flat schemas.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

JsonDict = Dict[str, object]


def summarize_entries(
    entries: Iterable[Mapping[str, object]],
    *,
    domain: Optional[str] = None,
) -> JsonDict:
    """
    Aggregate core metrics across telemetry entries.

    Parameters
    ----------
    entries:
        Iterable of telemetry payloads produced by
        :func:`src.telemetry.parsing.iter_entries` or compatible objects.
    domain:
        Optional filter (``"battle"`` or ``"overworld"``). When omitted,
        all entries contribute to the summary.
    """

    total_steps = 0
    latency_sum = 0.0
    fallback_count = 0
    fraction_totals = Counter[str]()  # encode/query/skip
    speedup_totals = Counter[str]()  # predicted/observed
    gate_view_counts = Counter[str]()

    domain_counts = Counter[str]()

    battle_steps = 0
    battle_writes_total = 0
    battle_legal_total = 0

    overworld_steps = 0
    overworld_action_sum = 0.0
    overworld_hybrid_projected = 0.0
    overworld_hybrid_ingested = 0.0
    overworld_recovery_count = 0
    overworld_encounter_events = 0
    overworld_status_counts = Counter[str]()

    plan_steps = 0
    plan_confidence_sum = 0.0
    plan_step_idx_sum = 0.0
    plan_steps_total_sum = 0.0
    plan_llm_latency_sum = 0.0
    plan_search_latency_sum = 0.0
    plan_cache_latency_sum = 0.0
    plan_gate_counts = Counter[str]()
    plan_kind_counts = Counter[str]()

    for entry in entries:
        ctx = entry.get("context") if isinstance(entry.get("context"), Mapping) else {}
        entry_domain = str(ctx.get("domain", "unknown"))
        if domain is not None and entry_domain != domain:
            continue

        telemetry = entry.get("telemetry") if isinstance(entry.get("telemetry"), Mapping) else {}
        core = telemetry.get("core") if isinstance(telemetry, Mapping) else {}
        if not isinstance(core, Mapping):
            continue

        total_steps += 1
        domain_counts[entry_domain] += 1

        latency = _safe_float(core.get("latency_ms"))
        latency_sum += latency
        if core.get("fallback_required"):
            fallback_count += 1

        fractions = core.get("fractions")
        if isinstance(fractions, Mapping):
            for key in ("encode", "query", "skip"):
                fraction_totals[key] += _safe_float(fractions.get(key))

        speedup = core.get("speedup")
        if isinstance(speedup, Mapping):
            for key in ("predicted", "observed"):
                speedup_totals[key] += _safe_float(speedup.get(key))

        gate = core.get("gate")
        if isinstance(gate, Mapping):
            view = gate.get("view")
            if isinstance(view, str) and view:
                gate_view_counts[view] += 1

        plan = core.get("plan")
        if isinstance(plan, Mapping):
            plan_steps += 1
            plan_confidence_sum += _safe_float(plan.get("plan_confidence"))
            plan_step_idx_sum += _safe_float(plan.get("step_idx"))
            plan_steps_total_sum += _safe_float(plan.get("steps_total"))
            plan_llm_latency_sum += _safe_float(plan.get("llm_latency_ms"))
            plan_search_latency_sum += _safe_float(plan.get("search_latency_ms"))
            plan_cache_latency_sum += _safe_float(plan.get("cache_latency_ms"))
            gate_dist = plan.get("gate_distribution")
            if isinstance(gate_dist, Mapping):
                for key, value in gate_dist.items():
                    plan_gate_counts[key] += _safe_float(value)

        if entry_domain == "battle":
            battle_steps += 1
            battle_payload = telemetry.get("battle") if isinstance(telemetry.get("battle"), Mapping) else {}
            writes = battle_payload.get("writes")
            if isinstance(writes, list):
                battle_writes_total += len(writes)
            legal_actions = core.get("legal_actions")
            if isinstance(legal_actions, list):
                battle_legal_total += len(legal_actions)

        if entry_domain == "overworld":
            overworld_steps += 1
            overworld_payload = telemetry.get("overworld") if isinstance(telemetry.get("overworld"), Mapping) else {}

            action_index = overworld_payload.get("action_index")
            overworld_action_sum += _safe_float(action_index)

            hybrid = overworld_payload.get("hybrid")
            if isinstance(hybrid, Mapping):
                overworld_hybrid_projected += _safe_float(hybrid.get("projected"))
                overworld_hybrid_ingested += _safe_float(hybrid.get("ingested"))

            if overworld_payload.get("recovery"):
                overworld_recovery_count += 1

            encounter = overworld_payload.get("encounter")
            if isinstance(encounter, list):
                overworld_encounter_events += len(encounter)

            status = ctx.get("status")
            if isinstance(status, str) and status:
                overworld_status_counts[status] += 1

    if total_steps == 0:
        return {
            "total_steps": 0,
            "core": {},
            "domains": {},
        }

    summary: Dict[str, object] = {
        "total_steps": total_steps,
        "domains": dict(domain_counts),
        "core": {
            "avg_latency_ms": latency_sum / total_steps,
            "fallback_rate": fallback_count / total_steps,
            "fractions": _average_counter(fraction_totals, total_steps),
            "speedup": _average_counter(speedup_totals, total_steps),
            "gate_view_counts": dict(gate_view_counts),
        },
    }

    if battle_steps:
        summary["battle"] = {
            "steps": battle_steps,
            "avg_write_ops": battle_writes_total / battle_steps,
            "avg_legal_actions": battle_legal_total / battle_steps if battle_steps else 0.0,
        }

    if overworld_steps:
        summary["overworld"] = {
            "steps": overworld_steps,
            "avg_action_index": overworld_action_sum / overworld_steps,
            "avg_projected_slots": overworld_hybrid_projected / overworld_steps,
            "avg_ingested_slots": overworld_hybrid_ingested / overworld_steps,
            "recovery_rate": overworld_recovery_count / overworld_steps,
            "avg_encounter_events": overworld_encounter_events / overworld_steps,
            "status_counts": dict(overworld_status_counts),
        }

    if plan_steps:
        summary["plan"] = {
            "usage_rate": plan_steps / total_steps,
            "avg_confidence": plan_confidence_sum / plan_steps,
            "avg_step_idx": plan_step_idx_sum / plan_steps,
            "avg_steps_total": plan_steps_total_sum / plan_steps if plan_steps else 0.0,
            "avg_llm_latency_ms": plan_llm_latency_sum / plan_steps,
            "avg_search_latency_ms": plan_search_latency_sum / plan_steps,
            "avg_cache_latency_ms": plan_cache_latency_sum / plan_steps,
            "gate_distribution": _average_counter(plan_gate_counts, plan_steps),
            "kind_counts": dict(plan_kind_counts),
        }

    return summary


def _average_counter(counter: Counter[str], denominator: int) -> Dict[str, float]:
    if denominator <= 0:
        return {}
    return {key: value / denominator for key, value in counter.items()}


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


__all__ = ["summarize_entries"]
