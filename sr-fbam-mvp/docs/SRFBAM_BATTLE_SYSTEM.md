# SR-FBAM Battle System Overview

This document summarises the current implementation of the SR-FBAM-powered Pokemon battle agent. It covers the full pipeline, module responsibilities, telemetry surface, and remaining limitations.

---

## 1. Architecture Summary

```
PyBoy Emulator
    | read_u8 / button scripts
    v
PyBoyPokemonAdapter (src/middleware/pyboy_adapter.py)
    | RAM snapshot -> BattleObs
    v
BlueRAMAdapter (src/pkmn_battle/env/blue_ram_adapter.py)
    | deterministic parsing -> WRITE ops
    v
BlueExtractor (src/pkmn_battle/extractor/blue_extractor.py)
    | node/edge writes (Pokemon, Move, Status, Field effects)
    v
GraphMemory (src/pkmn_battle/graph/memory.py)
    | assoc/follow + hop trace capture
    v
SRFBAMBattleAgent (src/srfbam/tasks/battle.py)
    |-- SRFBAMCore (shared gating/encoder)
    `-- BattleControllerPolicy + ActionSpace
        | logits + legal mask
        v
BattleActionHead (controller policy)
    | chosen action -> PyBoy button scripts
    v
PyBoyPokemonAdapter._navigate_battle_menu()
```

- **SRFBAMCore** reuses the code-editing SR-FBAM implementation (transformer symbol extractor, symbolic memory, learned gate).
- **BattleControllerPolicy** encodes lightweight battle features, invokes the core, and maps the output embeddings to logits while respecting legal-action masks.
- **GraphMemory** stores typed nodes/edges from the extractor and records hop traces whenever `follow()` is called.

---

## 2. Module Reference

| Module | Responsibility |
| ------ | -------------- |
| `src/middleware/pyboy_adapter.py` | Emulator bridge (RAM reads, button scripts, battle environment factory). |
| `src/pkmn_battle/env/blue_ram_adapter.py` | Converts RAM offsets into `BattleObs` (active Pokemon, parties, field effects). |
| `src/pkmn_battle/env/blue_ram_map.py` | Canonical Gen-1 RAM offsets with helper loaders for overrides. |
| `src/pkmn_battle/extractor/blue_extractor.py` | Emits typed nodes/edges (`Pokemon`, `Move`, `Status`, `Turn`, `Field`, screens, side-effects). |
| `src/pkmn_battle/graph/memory.py` | In-memory ASSOC/FOLLOW/WRITE store plus hop-trace collection. |
| `src/pkmn_battle/policy/controller_policy.py` | SR-FBAM controller-backed policy (embedding -> logits). |
| `src/srfbam/tasks/battle.py` | Task wrapper orchestrating extractor, graph, SR-FBAM core, and policy integration. |
| `scripts/run_battle_agent.py` | CLI harness with optional JSONL telemetry output. |

Unit tests for the symbolic stack live under `tests/pkmn_battle/`.

---

## 3. Telemetry & Logging

`SRFBAMBattleAgent.telemetry()` exposes:

- `last_summary` â€“ SR-FBAM gate statistics (cache hit rate, reuse rate, extract rate, etc.).
- `last_writes` â€“ WRITE operations applied during the latest observation phase.
- `legal_actions` / `action_mask` â€“ action-space inputs passed to the policy head.
- `gate_decision` â€“ discrete gate mode (`WRITE`, `ASSOC`, `FOLLOW`, `HALT`) together with the reason metadata provided by the controller policy.
- `hop_trace` â€“ list of `{src, relation, dst, edge_attributes, node_attributes}` entries emitted from recent `FOLLOW` calls.
- `fractions` â€“ running `(encode/query/skip)` proportions accumulated since the last reset.
- `speedup` â€“ predicted vs. observed latency speedup derived from the SR-FBAM three-mode cost model.
- `latency_ms` â€“ per-step wall-clock latency of the control loop.

HALT decisions surface an `encode_flag` to indicate whether the core fell back to dense frame encoding on that step.

Running `scripts/run_battle_agent.py --log-file battle.jsonl` appends one JSONL record per step using the shared schema (`source`, `context`, `observation`, `telemetry`). The `telemetry.core` namespace captures gate decisions, fractions, speed metrics, and hop traces, while `telemetry.battle` carries battle-specific data such as write operations and the legal-action index map.
See `docs/telemetry_schema.json` for the complete specification.

---

## 4. Known Limitations / TODO

- **Menu navigation**: the executor clears the standard fight/Pokemon dialogs and mashes through textbox sequences, but it does not yet reason about exceptional prompts (e.g., trying to send a fainted Pokemon or other yes/no confirmations).
- **Metadata coverage**: the RAM map currently exposes active Pokemon, party members, status ailments, Light Screen / Reflect, and a small selection of side-effect flags. Additional overlays (hazards, item reveals, multi-turn effects) can be mapped as needed.
- **Training**: the battle policy runs the SR-FBAM core in inference mode. Offline training/evaluation pipelines for battle datasets are planned but not yet implemented.
- **Type-chart reasoning**: deterministic damage calculators and type-matchup heuristics are not yet integrated; the controller presently relies on learned embeddings plus symbolic lookups.

---

## 5. Getting Started Checklist

1. Install dependencies and confirm PyBoy can boot the target ROM.
2. Review/adjust `data/blue_ram_map.yaml` or build an override via `ram_map_from_dict()` if offsets differ.
3. Run `python scripts/run_battle_agent.py --rom <path> --steps 200 --log-file runs/battle.jsonl`.
4. Inspect console/log output to verify gate decisions, `(e/q/s)` fractions, hop traces, and button-script execution.
5. Extend the extractor/graph (and accompanying unit tests) if new entitiesâ€”items, hazards, weather, etc.â€”are required.

For additional architectural context, see `SR_FBAM_ARCHITECTURE_OVERVIEW.md` and the inline docstrings referenced throughout the codebase.


