# MVE Implementation Checklist (from Overview.md:142-146)

## Stack Setup
- [ ] PyTorch with AMP (automatic mixed precision)
- [ ] torch.checkpoint for activation recomputation
- [ ] NetworkX for graph structure
- [ ] FAISS for approximate nearest neighbor (optional for Phase 1)
- [ ] SQLite for persistent graph storage (optional for Phase 1)

## Parameter Budget (MUST match across all models)
- [ ] SR-FBAM: 1M params
  - [ ] Frame Head: ~300K
  - [ ] Integrator (LSTM+MLP): ~500K
  - [ ] Embeddings: ~200K
- [ ] Transformer baseline: 1M +/-10%
- [ ] Pure LSTM baseline: 1M +/-10%
- [ ] Pure FBAM baseline: 1M +/-10%

## Training Config
- [ ] TBPTT = 256 steps
- [ ] Micro-batching (batch size = ?)
- [ ] Gradient clipping (threshold = ?)
- [ ] Activation recomputation enabled
- [ ] Logging hooks for hop traces (ExperimentLogger)
- [x] Data loaders wired into training harness (`src/training/train.py`)

## Logging Schema
- [x] HopTrace dataclass implemented
- [x] QueryLog dataclass implemented
- [x] JSON output per experiment run
- [ ] Auto-generated plots from JSON

## Operators (5 total)
- [ ] ASSOC: associate/query by relation
- [ ] FOLLOW: traverse edge
- [ ] VOTE: consensus/confidence
- [ ] WRITE: update graph
- [ ] HALT: terminate
