## Integrated Planlet Migration Checklist (Battle + Overworld)

Legend: ✅ done, 🆕 new/overworld, ◻︎ pending  
DoD = Definition of Done (unblocks the next phase)

---

### Phase 0 – Foundations (✅ battle assets; add overworld stubs)

- ✅ Mechanics data mirrored (`data/mechanics/`)
- ✅ Battle graph + SR‑FBAM memory operators
- ✅ Frame renderer → 40×120 grid
- ✅ Metamon trajectory ingestion + JSONL schema
- 🆕 ◻︎ WorldGraph skeleton (`pkmn_overworld/world_graph.py`):  
  nodes {Location, Door/Warp, NPC, Shop, Center, Item, Trigger, Hazard},  
  edges {adjacent, warp_to, requires(item|flag), sells(item)}
- 🆕 ◻︎ Screen parsers for overworld (`pkmn_overworld/screen_parse.py`): tile clusters, dialog text, UI icons → WorldGraph updates

**DoD:** One replay runs SR‑FBAM-only (no LLM) in both modes: battle baseline + overworld baseline that walks a short corridor using adjacent edges.

---

### Phase 1 – Schema & Telemetry Contracts (extend, don’t refactor)

- ✅ Publish planlet schema (`docs/PLANLET_SCHEMA.md`)
- ✅ Note planlet telemetry fields (planlet_id, confidence, gate distribution, LLM latency)
- ◻︎ Add JSONSchema validation to telemetry ingestion (`telemetry_schema.json`)
- ◻︎ Update dashboards to surface plan metrics
- 🆕 ◻︎ Unify planlets with `kind: "BATTLE" | "OVERWORLD"` and op enums
- 🆕 ◻︎ Telemetry additions (overworld): `mode:"OVERWORLD"`, skill, path_len_planned, path_len_executed, encounters, abort_code, adhere_code

**DoD:** CI fails on invalid planlet/telemetry JSON; dashboard panels show per-turn gate mix and plan adherence for both modes (overworld + battle).

_Planlet JSON (tight core, unified)_  
```json
{
  "planlet_id": "pl_2025-10-21_0004",
  "kind": "OVERWORLD",
  "seed_frame_id": 9121,
  "horizon_steps": 5,
  "goal": "Heal team and buy 3 Potions, then reach Route 3 gate.",
  "rationale": "Low HP; nearest Center and Mart are reachable.",
  "preconditions": ["Cash >= 900", "Center reachable"],
  "script": [
    {"op":"NAVIGATE","to":"PokeCenter_Pewter"},
    {"op":"INTERACT","target":"Nurse"},
    {"op":"NAVIGATE","to":"Mart_Pewter"},
    {"op":"BUY","item":"Potion","qty":3},
    {"op":"NAVIGATE","to":"Route3_Gate"}
  ],
  "constraints": ["Avoid_Grass"],
  "retrieved_docs": [],
  "confidence": 0.68
}
```

---

### Phase 2 – Graph Summary & LLM Harness (add overworld summaries)

- ✅ Deterministic battle summarizer (`pkmn_battle/summarizer.py`) + smoke tests
- ✅ Planlet proposer stub + schema validation (`llm/planlets/`)
- ◻︎ Integrate GPT‑5 tool prompts + search stubs (`llm/planlets/proposer.py`)
- ◻︎ End-to-end harness to request planlets from live state
- 🆕 ◻︎ Overworld summarizer (`pkmn_overworld/summarizer.py`): player {pos,map,cash}, party, inventory(top‑k), nearby {locations,npcs,shops,hazards}, objectives(flags), graph_stats
- 🆕 ◻︎ Fake‑LLM deterministic backend for CI (returns seeded overworld planlets)

**DoD:** `scripts/request_planlet.py --mode overworld` emits a valid overworld planlet for a saved frame; golden snapshot tests verify stable ordering/size of summaries.

---

### Phase 3 – Plan Logging & Storage (one store, two kinds)

- ◻︎ Extend battle runner to attach planlet metadata per turn
- ◻︎ Persist planlets (Parquet/JSONL) with retrieved docs & tokens
- ◻︎ Manifest support for plan-conditioned JSONL (train/val/test)
- 🆕 ◻︎ Plan cache keyed by graph neighborhood hash:  
  • Battle: (our_active, opp_active, hazards_hash)  
  • Overworld: (map_id, zone_hash(W.pos), top_flags[:N])
- 🆕 ◻︎ Success-weighted eviction for plan cache; retention policy/TTL

**DoD:** After a mixed (overworld↔battle) run, `data/planlets/planlets.parquet` contains both kinds, linked to frames; cache hits reduce LLM calls on repeated zones.

---

### Phase 4 – Executor Updates (shared gates; new overworld skills)

- ◻︎ Add SR‑FBAM gates `PLAN_LOOKUP` / `PLAN_STEP`
- ◻︎ Map battle plan ops → legal actions + fallbacks
- ◻︎ Runtime precondition checks & graceful abort path
- ◻︎ Telemetry: log gate probabilities, plan adherence, fallbacks
- 🆕 ◻︎ Overworld skills (`pkmn_overworld/skills/*.py`):  
  `SKILL_NAVIGATE`, `SKILL_INTERACT`, `SKILL_TALK`, `SKILL_BUY`,  
  `SKILL_PICKUP`, `SKILL_USE_ITEM`, `SKILL_MENU`, `SKILL_WAIT`
- 🆕 ◻︎ Op→Skill registry (`op_registry.py`) with unit tests + illegal-state fuzzing
- 🆕 ◻︎ Context switch logic: battle interrupts overwrite plan; post-battle resume/revise
- 🆕 ◻︎ Rule-based confidence gate v0 (pre-learned head)

**DoD:** With fake planlets, runner executes ≥2 overworld script steps via PLAN_STEP, aborts cleanly on precondition failure, logs `adhere_code=OK|DEVIATE|ABORT`. Mixed run handles battle interruptions and resumes.

_Op→Skill mapping (excerpt)_  
```python
OW_OP_TO_SKILL = {
    "NAVIGATE": "SKILL_NAVIGATE",
    "INTERACT": "SKILL_INTERACT",
    "TALK":     "SKILL_TALK",
    "BUY":      "SKILL_BUY",
    "PICKUP":   "SKILL_PICKUP",
    "USE":      "SKILL_USE_ITEM",
    "WAIT":     "SKILL_WAIT",
}
```

---

### Phase 5 – Training & Distillation (unified controller; mode bit)

- ◻︎ Extend battle dataset with plan features (goal embedding, script op)
- ◻︎ Implement plan-conditioned controller training loop (confidence head)
- ◻︎ Set up self-distillation (LLM-on → retrain → LLM-off)
- 🆕 ◻︎ OverworldDecisionDataset merges with battle dataset; add `mode_bit` + overworld plan features (goal_cls, next_op, horizon_bucket, 64‑d text embedding of goal+rationale)
- 🆕 ◻︎ Ablations: no-plan vs plan features; text masked; gate frozen vs trainable (both modes)

**DoD:** On held-out mixed tasks, maintain or improve baseline success rates with ≥40% fewer LLM calls after first distill; gate mix shows ENCODE↓, PLAN_*↑ and/or ASSOC/FOLLOW↑ in both modes (mirrors SR‑FBAM amortization).

---

### Phase 6 – Evaluation & QA (add overworld tasks & metrics)

- ◻︎ Define metrics: plan success rate, LLM reliance, latency, win-rate
- ◻︎ Regression tests for planlet schema + prompt compliance
- ◻︎ Benchmarks for latency with and without planlets
- 🆕 ◻︎ Overworld tasks: Reach location, Heal+Shop, Fetch+Deliver, Multi-hop route (constraints: Avoid_Grass, No_Ledge)
- 🆕 ◻︎ Overworld metrics: success rate, path stretch (executed/optimal), encounters per 100 tiles, p95 per-turn latency, LLM calls per task, tokens/km walked, gate mix
- 🆕 ◻︎ Value-of-computation: delta success/winprob on turns with vs without LLM calls (both modes)

**DoD:** Dashboard shows LLM calls/battle and LLM calls/overworld-task, tokens, adherence, p95 latency; report includes gate-mix trends and speedups grounded in discrete entity queries.

---

### Phase 7 – Cleanup & Decommission (keep a kill-switch)

- ◻︎ Retire legacy code-editing dataset + trainers (optional archive)
- ◻︎ Prune unused imitation scripts once plan flow is primary
- ◻︎ Document operating handbook for planlet mode (README + telemetry guide)
- 🆕 ◻︎ Flags: `--disable-planlets` (SR‑FBAM-only), `--disable-overworld` (battle-only)

**DoD:** System runs under all flags; README includes Overworld Runbook (failure modes, budgets, resume rules, FAQs).

---

### Minimal interfaces (reference)

_World summarizer (deterministic, bounded)_  
```python
def summarize_world_for_llm(world) -> dict:
    return {
        "player": {"pos": world.pos, "map": world.map_id, "cash": world.cash},
        "party": brief_party(world.party),
        "inventory": top_items(world.inventory, k=10),
        "nearby": {
            "locations": k_nearest_locations(world, k=5),
            "npcs": seen_npcs(world, radius=R),
            "shops": seen_shops(world),
            "hazards": visible_hazards(world),
        },
        "objectives": active_objectives(world),
        "graph": {"nodes": world.stats.nodes, "edges": world.stats.edges},
    }
```

_Zone hash (plan cache key)_  
```python
def zone_hash(world) -> int:
    return hash((world.map_id, bucket(world.pos), tuple(sorted(world.active_flags))[:8]))
```

_Precondition check (shared battle/overworld)_  
```python
def preconds_ok(planlet, graph_or_world) -> bool:
    for cond in planlet["preconditions"]:
        if not eval_precond(cond, graph_or_world):
            return False
    return True
```

_Directory layout (new files in **bold**)_  
```
src/
  pkmn_battle/
    summarizer.py
    graph/...
  pkmn_overworld/
    **world_graph.py**
    **screen_parse.py**
    **summarizer.py**
    **skills/**
      **navigate.py, interact.py, shop.py, inventory.py**
  llm/planlets/
    proposer.py
    **service.py**
  schemas/
    planlet.schema.json
    telemetry.schema.json
scripts/
  request_planlet.py
  run_battle_agent.py
  **run_overworld_agent.py**
```

**Why this stays faithful to SR‑FBAM**

We keep a single controller with discrete gates `{ENCODE, ASSOC, FOLLOW, WRITE, PLAN_LOOKUP, PLAN_STEP, HALT}` controlling two graphs (BattleGraph & WorldGraph). Overworld skills are deterministic programs that consume entity queries from WorldGraph; the controller learns when to query versus encode, delivering the same amortized speed and generalization benefits SR‑FBAM demonstrated for battle.
