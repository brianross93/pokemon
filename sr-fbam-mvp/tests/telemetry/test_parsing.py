from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.telemetry import normalize_entry


def _load_first_line(path: Path) -> dict:
    line = path.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


def test_normalize_legacy_battle(tmp_path: Path) -> None:
    legacy_path = Path("runs/test_cpu.jsonl")
    if not legacy_path.exists():
        pytest.skip("legacy battle telemetry fixture not available")
    legacy_entry = _load_first_line(legacy_path)

    normalized = normalize_entry(legacy_entry)
    assert normalized["context"]["domain"] == "battle"

    telemetry = normalized["telemetry"]
    core = telemetry["core"]
    gate = core["gate"]

    assert gate["mode"] == legacy_entry["gate"]["op"]
    assert core["fractions"]["encode"] == pytest.approx(legacy_entry["fractions"]["encode"])
    assert core["speedup"]["observed"] == pytest.approx(legacy_entry["speedup"]["observed"])
    assert telemetry["battle"]["writes"] == legacy_entry["writes"]


def test_normalize_legacy_overworld() -> None:
    legacy_path = Path("trace.jsonl")
    if not legacy_path.exists():
        pytest.skip("legacy overworld telemetry fixture not available")

    legacy_entry = _load_first_line(legacy_path)
    normalized = normalize_entry(legacy_entry)

    context = normalized["context"]
    assert context["domain"] == "overworld"
    assert context["plan"]["planlet_id"] == legacy_entry["planlet_id"]
    assert context["status"] == legacy_entry["status"]

    telemetry = normalized["telemetry"]
    core = telemetry["core"]
    overworld = telemetry["overworld"]

    assert core["gate"]["mode"] == legacy_entry["telemetry"]["gate"]["mode"]
    assert overworld["memory"] == legacy_entry["telemetry"]["memory"]
    assert overworld["hybrid"] == legacy_entry["telemetry"]["hybrid"]

