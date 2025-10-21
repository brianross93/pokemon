from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pytest

from src.telemetry import iter_entries
from src.telemetry.analytics import summarize_entries


def _load_legacy_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_summarize_battle_matches_legacy_manual() -> None:
    path = Path("runs/test_cpu.jsonl")
    if not path.exists():
        pytest.skip("legacy battle telemetry fixture not available")

    legacy_records = _load_legacy_jsonl(path)
    manual_latency = mean(rec["latency_ms"] for rec in legacy_records)
    manual_fallback = sum(1 for rec in legacy_records if rec.get("fallback_required")) / len(legacy_records)
    manual_encode = mean(rec["fractions"]["encode"] for rec in legacy_records)
    manual_pred = mean(rec["speedup"]["predicted"] for rec in legacy_records)
    manual_obs = mean(rec["speedup"]["observed"] for rec in legacy_records)
    manual_writes = mean(len(rec.get("writes", [])) for rec in legacy_records)

    summary = summarize_entries(iter_entries(path), domain="battle")
    core = summary["core"]
    battle = summary["battle"]

    assert core["avg_latency_ms"] == pytest.approx(manual_latency)
    assert core["fallback_rate"] == pytest.approx(manual_fallback)
    assert core["fractions"]["encode"] == pytest.approx(manual_encode)
    assert core["speedup"]["predicted"] == pytest.approx(manual_pred)
    assert core["speedup"]["observed"] == pytest.approx(manual_obs)
    assert battle["avg_write_ops"] == pytest.approx(manual_writes)


def test_summarize_overworld_matches_legacy_manual() -> None:
    path = Path("trace.jsonl")
    if not path.exists():
        pytest.skip("legacy overworld telemetry fixture not available")

    legacy_records = _load_legacy_jsonl(path)
    manual_latency = mean(rec["telemetry"]["latency_ms"] for rec in legacy_records)
    manual_fallback = sum(1 for rec in legacy_records if rec["telemetry"].get("fallback_required")) / len(legacy_records)
    manual_encode = mean(rec["telemetry"]["fractions"]["encode"] for rec in legacy_records)
    manual_pred = mean(rec["telemetry"]["speedup"]["predicted"] for rec in legacy_records)
    manual_obs = mean(rec["telemetry"]["speedup"]["observed"] for rec in legacy_records)
    manual_action_index = mean(rec["telemetry"].get("action_index", 0.0) for rec in legacy_records)

    summary = summarize_entries(iter_entries(path), domain="overworld")
    core = summary["core"]
    overworld = summary["overworld"]

    assert core["avg_latency_ms"] == pytest.approx(manual_latency)
    assert core["fallback_rate"] == pytest.approx(manual_fallback)
    assert core["fractions"]["encode"] == pytest.approx(manual_encode)
    assert core["speedup"]["predicted"] == pytest.approx(manual_pred)
    assert core["speedup"]["observed"] == pytest.approx(manual_obs)
    assert overworld["avg_action_index"] == pytest.approx(manual_action_index)

