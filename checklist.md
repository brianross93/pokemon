# MVE Implementation Checklist (from Overview.md:142-146)

## Stack Setup
- [x] PyTorch with AMP (automatic mixed precision)
- [ ] torch.checkpoint for activation recomputation
- [ ] NetworkX for graph structure
- [ ] FAISS for approximate nearest neighbor (optional for Phase 1)
- [ ] SQLite for persistent graph storage (optional for Phase 1)

## Parameter Budget (MUST match across all models)
- [x] SR-FBAM: 1M params
  - [x] Frame Head: ~300K
  - [x] Integrator (LSTM+MLP): ~500K
  - [x] Embeddings: ~200K
- [x] Transformer baseline: 1M +/-10%
- [x] Pure LSTM baseline: 1M +/-10%
- [ ] Pure FBAM baseline: 1M +/-10%

- [x] Instructor-labeled action loss (`compute_action_loss`)

## Training Config
- [ ] TBPTT = 256 steps
- [x] Micro-batching (batch size configurable via --batch-size)
- [x] Gradient clipping (threshold = 1.0)
- [ ] Activation recomputation enabled
- [x] Logging hooks for hop traces (ExperimentLogger)
- [x] Data loaders wired into training harness (`src/training/train.py`)
- [x] SR-FBAM supervised training loop (`train_srfbam`)

## Logging Schema
- [x] HopTrace dataclass implemented
- [x] QueryLog dataclass implemented
- [x] JSON output per experiment run
- [x] Auto-generated plots from JSON
- [x] Plotting script + summary tooling (`scripts/plot_results.py`)

## Operators (5 total)
- [x] ASSOC: associate/query by relation
- [x] FOLLOW: traverse edge
- [x] VOTE: consensus/confidence
- [x] WRITE: update graph
- [x] HALT: terminate

## Deployment & Demo
- [x] Checkpoint saving & best-model tracking
- [x] Evaluation mode for reuse (`--mode eval`)
- [x] Interactive demo script (`scripts/demo.py`)
- [x] Notebook walkthrough (`notebooks/02_demo_inference.ipynb`)
- [x] Quick start docs & requirements
- [x] LSTM baseline pipeline (`src/training/train_lstm.py`)
- [x] Transformer baseline pipeline (`src/training/train_transformer.py`)
