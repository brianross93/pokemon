# SR-FBAM MVP: Symbolic-Recurrence Frame-Based Action Model

**Minimum Viable Product** for testing the SR-FBAM architecture (Phase 1)

Aligned with: `Overview.md:101-174` (Problem -> Solution brief)

## Status: Data Generation Complete

All synthetic data has been generated, validated, and is ready for model implementation.

## Quick Links

- [DATA_DELIVERY_SUMMARY.md](DATA_DELIVERY_SUMMARY.md) - Complete data overview and quick start
- [DATA_FILES_MANIFEST.txt](DATA_FILES_MANIFEST.txt) - List of all generated files
- [data/README.md](data/README.md) - Detailed dataset documentation
- [Overview.md](../Overview.md) - Whitepaper outline and specs

## Project Structure

```
sr-fbam-mvp/
|-- config/
|   |-- metrics_spec.yaml          # Baseline specs
|-- data/                          # Generated synthetic KG + queries
|   |-- nodes.csv                  # 99 entities
|   |-- edges.csv                  # 332 relations
|   |-- queries_train.jsonl        # 171 queries
|   |-- queries_eval.jsonl         # 43 queries
|   |-- stress_variants/           # 5 test scenarios
|-- notebooks/
|   |-- 01_explore_data.ipynb      # Data exploration
|-- plots/                         # Figures (empty)
|-- results/                       # Experiment outputs (empty)
|-- scripts/
|   |-- generate_data.py           # Data generation
|   |-- validate_data.py           # Integrity checks
|-- src/
    |-- data/
    |   |-- kg_loader.py           # KG loading API (DONE)
    |-- evaluation/
    |   |-- logger_schema.py       # Logging classes (DONE)
    |   |-- stress_tests.py        # Test specs (DONE)
    |-- models/                    # SR-FBAM + baselines (TODO)
    |-- training/                  # Training loop (TODO)
    |-- utils/                     # Visualization (TODO)
```

## Dataset Summary

Baseline knowledge graph:
- 99 nodes (Actor, Director, Movie, Location, Genre, Award)
- 332 edges (born_in, featured_in, directed_by, friend_of, etc.)
- Movie/family domain with 13 relation types

Queries:
- 171 training queries (45% 1-hop, 22% 2-hop, 33% 3-hop)
- 43 evaluation queries (67% baseline, 21% ambiguous, 12% noisy)

Stress test variants:
1. Ambiguous nodes (duplicate names)
2. Long chains (8-15 hops)
3. Noisy edges (20% spurious relations)
4. Missing nodes (10% removed)
5. Graph growth (incremental additions)

## Quick Start

### 1. Verify Data

```bash
python scripts/validate_data.py
```

Expected output (abridged):
```
[OK] Nodes: 99 (expected ~100)
[OK] Edges: 332 (expected ~300-350)
[OK] Train queries: 171
[OK] Eval queries: 43
ALL VALIDATIONS PASSED [OK]
```

### 2. Explore Data

```python
from pathlib import Path
from src.data.kg_loader import load_dataset, load_kg

data_dir = Path("data")
kg, queries = load_dataset(data_dir, split="train")

print(f"Nodes: {kg.num_nodes}, Edges: {kg.num_edges}")
print(f"Queries: {len(queries)}")

paris = kg.get_nodes_by_name("Paris")[0]
actors = kg.assoc("born_in", target_id=paris.node_id, node_type="Actor")
movies = kg.follow(actors[0], "featured_in", node_type="Movie")
directors = kg.follow(movies[0], "directed_by", node_type="Director")

print(f"Director: {kg.get_node(directors[0]).name}")
```

See `notebooks/01_explore_data.ipynb` for interactive examples.

### 3. Load Stress Variants

```python
kg_noisy = load_kg(data_dir, variant="noisy_edges")
noisy_count = sum(1 for edge in kg_noisy.edges if edge.is_noisy)
print(f"Noisy edges: {noisy_count}/{kg_noisy.num_edges}")

kg_missing = load_kg(data_dir, variant="missing_nodes")
print(f"Nodes: {kg_missing.num_nodes} (baseline had {kg.num_nodes})")
```

## Phase 1 Implementation Roadmap

Completed:
- [x] Data generation (movie/family KG, ~100 nodes)
- [x] Query generation (171 train, 43 eval, 1-3 hop templates)
- [x] Stress-test variants (5 scenarios)
- [x] Data loaders (`src/data/kg_loader.py`)
- [x] Logging schema (`src/evaluation/logger_schema.py`)
- [x] Validation script
- [x] Documentation (README, manifests)

Next steps (Days 1-7):
1. Implement models (`src/models/`)
   - `sr_fbam.py`: Frame Head + LSTM integrator + symbolic actions
   - `baselines.py`: Pure FBAM, Transformer, pure LSTM
   - Ensure 1M parameters each (Frame Head ~300K, Integrator ~500K, embeddings ~200K) per `Overview.md:142-146`.
2. Set up training (`src/training/`)
   - `train.py`: TBPTT=256, activation recomputation, micro-batching, gradient clipping
   - Use `ExperimentLogger` for hop tracing and aggregate metrics
   - Load KG + queries via `load_dataset()`
3. Run experiments
   - Train on baseline dataset
   - Evaluate on stress variants (`src/evaluation/stress_tests.py`)
   - Capture logs and plots (loss vs wall time, hops to answer, accuracy)
4. Draft paper Sections 4-7
   - Section 4: Architecture (Overview draft plus figures)
   - Section 5: Scaling hypothesis (record empirical alpha)
   - Section 6: Experimental design (module sizes, configs)
   - Section 7: Results (plots + hop traces from `results/` and `plots/`)

## Key Specifications

From `Overview.md:142-146`:
- Stack: PyTorch (AMP + `torch.checkpoint`), NetworkX, fixed-width text frames
- Model budget: 1M parameters (Frame Head 300K, Integrator 500K, embeddings 200K)
- Training: TBPTT=256, micro-batching, gradient clipping, activation recomputation
- Logging: Persist hop traces, confidence scores, and wall time

From `src/evaluation/stress_tests.py`:
- Baseline: ~0.95 accuracy, clean graph
- Ambiguous nodes: 5 duplicates, VOTE expected to resolve most ties (~0.80 accuracy)
- Long chains: 8-15 hops, expect <0.60 accuracy without curriculum
- Noisy edges: 20% spurious relations, target ~0.70 accuracy with VOTE
- Missing nodes: 10% removed, expect 10% no-answer rate
- Graph growth: Maintain memory bound via pruning (~2x cap) with stable accuracy

## Regeneration

```bash
# Adjust generation parameters in scripts/generate_data.py (seed=42 by default)
python scripts/generate_data.py
python scripts/validate_data.py
```

## References

- `Overview.md` - Whitepaper outline and architectural specs
- `checklist.md` - Implementation checklist
- `config/metrics_spec.yaml` - Baseline model specifications
- `src/evaluation/stress_tests.py` - Test scenario definitions
- `data/README.md` - Dataset documentation

## Dependencies

- Python 3.11+
- NetworkX (graph operations)
- PyTorch (model implementation)
- pandas (CSV loading)

Install (baseline tooling):
```bash
pip install torch networkx pandas jupyter
```

## Contact

For questions or issues:
- Data loading: `src/data/kg_loader.py` docstrings
- Stress tests: `src/evaluation/stress_tests.py`
- Architecture and roadmap: `Overview.md:101-174`

---

**Status:** Data complete and ready for Phase 1 model implementation.
