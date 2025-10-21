"""
FBAM variant with sparse retrieval memory backed by a FAISS index.

This baseline approximates modern retrieval-style external memories by only
examining the top-k nearest neighbors for each query instead of attending over
all slots. Memory slots are kept as a ring buffer so the total capacity remains
fixed while the FAISS index supports sublinear lookups.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .fbam_code_baseline import FrameTransformer, PureFBAMCodeConfig

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    faiss = None


@dataclass
class FBAMSparseMemoryConfig(PureFBAMCodeConfig):
    """Configuration for FBAM with sparse FAISS-backed memory."""

    memory_slots: int = 500
    memory_dim: int = 128
    k_neighbors: int = 10
    index_description: str = "Flat"  # passed to faiss.index_factory
    min_write_gate: float = 1e-3


class FBAMSparseMemoryAgent(nn.Module):
    """FBAM variant that queries a FAISS index for top-k memory retrieval."""

    def __init__(
        self,
        num_actions: int,
        config: Optional[FBAMSparseMemoryConfig] = None,
    ) -> None:
        super().__init__()
        if faiss is None:
            raise ImportError(
                "faiss is required for FBAMSparseMemoryAgent. Install faiss-cpu (pip install faiss-cpu)"
            )
        self.config = config or FBAMSparseMemoryConfig()
        self.num_actions = num_actions

        self.frame_encoder = FrameTransformer(self.config)
        concat_dim = (
            self.config.frame_d_model
            + self.config.memory_dim
            + self.config.action_embedding_dim
        )

        self.action_embed = nn.Embedding(num_actions + 1, self.config.action_embedding_dim)
        self.integrator = nn.LSTMCell(concat_dim, self.config.integrator_hidden_dim)
        self.classifier = nn.Linear(self.config.integrator_hidden_dim, num_actions)
        self.dropout = nn.Dropout(self.config.integrator_dropout)

        self.query_proj = nn.Linear(self.config.integrator_hidden_dim, self.config.memory_dim)
        self.write_proj = nn.Linear(self.config.integrator_hidden_dim, self.config.memory_dim)
        self.write_gate = nn.Linear(self.config.integrator_hidden_dim, 1)

        self.start_action_index = num_actions

        self._profile_enabled: bool = False
        self._profile_stats: Dict[str, float] = {}
        self.reset_profile_stats()

    # ----- Profiling helpers ----------------------------------------------
    def reset_profile_stats(self) -> None:
        self._profile_stats = {
            "read_calls": 0.0,
            "write_calls": 0.0,
            "writes_skipped": 0.0,
            "slots_written": 0.0,
            "neighbors_requested": 0.0,
            "neighbors_returned": 0.0,
            "faiss_search_ms": 0.0,
            "query_proj_ms": 0.0,
            "read_time_ms": 0.0,
            "write_time_ms": 0.0,
            "write_proj_ms": 0.0,
            "gate_sum": 0.0,
            "index_size": 0.0,
        }

    def enable_profiling(self) -> None:
        self._profile_enabled = True
        self.reset_profile_stats()

    def disable_profiling(self) -> None:
        self._profile_enabled = False

    def profile_stats(self) -> Dict[str, float]:
        return dict(self._profile_stats)

    # ----- Memory helpers -------------------------------------------------
    def _new_index(self) -> faiss.Index:
        """Return a fresh FAISS index configured per the config."""
        if self.config.index_description.lower() == "flat":
            base = faiss.IndexFlatL2(self.config.memory_dim)
        else:
            base = faiss.index_factory(
                self.config.memory_dim,
                self.config.index_description,
                faiss.METRIC_L2,
            )
        
        # REQUIRE GPU FAISS - fail fast if not available
        if not (faiss.get_num_gpus() > 0):
            raise RuntimeError(f"FAISS GPU required but only {faiss.get_num_gpus()} GPUs available. Install faiss-gpu.")
        
        if not (hasattr(self, 'device') and str(self.device).startswith('cuda')):
            raise RuntimeError(f"Model must be on CUDA device for GPU FAISS, but model is on {getattr(self, 'device', 'unknown')}")
        
        try:
            gpu_resource = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, base)
            return faiss.IndexIDMap(gpu_index)
        except Exception as e:
            raise RuntimeError(f"Failed to create GPU FAISS index: {e}") from e

    def init_memory(self) -> Dict[str, object]:
        """Initialise ring-buffer memory state for an episode."""
        return {
            "embeddings": torch.zeros(
                self.config.memory_slots,
                self.config.memory_dim,
                dtype=torch.float32,
            ),
            "filled": [False] * self.config.memory_slots,
            "ids": torch.full((self.config.memory_slots,), -1, dtype=torch.long),
            "id_to_slot": {},  # type: ignore[var-annotated]
            "index": self._new_index(),
            "next_slot": 0,
            "count": 0,
            "next_id": 0,
        }

    # ----- Model helpers --------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> tuple[Tensor, Tensor]:
        device = device or self.device
        h = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        c = torch.zeros(batch_size, self.config.integrator_hidden_dim, device=device)
        return h, c

    def _faiss_search(
        self,
        memory: Dict[str, object],
        query: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        index: faiss.IndexIDMap = memory["index"]  # type: ignore[assignment]
        distances, ids = index.search(query, k)
        return distances, ids

    def _read_memory(self, h: Tensor, memory: Dict[str, object]) -> Tensor:
        if memory["count"] == 0:  # type: ignore[index]
            return torch.zeros(1, self.config.memory_dim, device=h.device)

        if self._profile_enabled:
            self._profile_stats["read_calls"] += 1

        query_start = perf_counter() if self._profile_enabled else None
        query_tensor = self.query_proj(h)
        if self._profile_enabled and query_start is not None:
            self._profile_stats["query_proj_ms"] += (perf_counter() - query_start) * 1000.0
        query = query_tensor.detach().cpu().numpy().astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        k = min(self.config.k_neighbors, memory["count"])  # type: ignore[index]
        search_start = perf_counter() if self._profile_enabled else None
        distances, ids = self._faiss_search(memory, query, k)
        if self._profile_enabled and search_start is not None:
            self._profile_stats["faiss_search_ms"] += (perf_counter() - search_start) * 1000.0
            self._profile_stats["neighbors_requested"] += float(k)

        id_to_slot: Dict[int, int] = memory["id_to_slot"]  # type: ignore[assignment]
        embeddings: torch.Tensor = memory["embeddings"]  # type: ignore[assignment]

        vectors: List[torch.Tensor] = []
        dists: List[float] = []
        retrieval_start = perf_counter() if self._profile_enabled else None
        for dist, idx in zip(distances[0], ids[0]):
            if idx < 0:
                continue
            slot = id_to_slot.get(int(idx))
            if slot is None:
                continue
            vectors.append(embeddings[slot])
            dists.append(float(dist))

        if not vectors:
            return torch.zeros(1, self.config.memory_dim, device=h.device)

        stacked = torch.stack(vectors).to(h.device)
        dist_tensor = torch.tensor(dists, dtype=torch.float32, device=h.device)
        weights = torch.softmax(-dist_tensor, dim=0).unsqueeze(1)
        retrieved = (stacked * weights).sum(dim=0, keepdim=True)

        if self._profile_enabled:
            self._profile_stats["neighbors_returned"] += float(len(vectors))
            if retrieval_start is not None:
                self._profile_stats["read_time_ms"] += (perf_counter() - retrieval_start) * 1000.0

        return retrieved

    def _write_memory(self, h: Tensor, memory: Dict[str, object]) -> None:
        gate = torch.sigmoid(self.write_gate(h)).item()
        if gate <= self.config.min_write_gate:
            if self._profile_enabled:
                self._profile_stats["writes_skipped"] += 1.0
                self._profile_stats["gate_sum"] += gate
            return

        if self._profile_enabled:
            self._profile_stats["write_calls"] += 1.0
            self._profile_stats["gate_sum"] += gate
            write_start = perf_counter()

        proj_start = perf_counter() if self._profile_enabled else None
        write_tensor = self.write_proj(h)
        if self._profile_enabled and proj_start is not None:
            self._profile_stats["write_proj_ms"] += (perf_counter() - proj_start) * 1000.0
        write_vec = (gate * write_tensor).detach().cpu().squeeze(0).to(torch.float32)
        vec_np = write_vec.numpy()[np.newaxis, :]

        slot = memory["next_slot"]  # type: ignore[index]
        filled: List[bool] = memory["filled"]  # type: ignore[assignment]
        ids_tensor: torch.Tensor = memory["ids"]  # type: ignore[assignment]
        id_to_slot: Dict[int, int] = memory["id_to_slot"]  # type: ignore[assignment]
        index: faiss.IndexIDMap = memory["index"]  # type: ignore[assignment]

        if filled[slot]:
            old_id = int(ids_tensor[slot])
            index.remove_ids(np.array([old_id], dtype=np.int64))
            id_to_slot.pop(old_id, None)

        new_id = int(memory["next_id"])  # type: ignore[index]
        memory["next_id"] = new_id + 1  # type: ignore[index]

        embeddings: torch.Tensor = memory["embeddings"]  # type: ignore[assignment]
        embeddings[slot] = write_vec
        ids_tensor[slot] = new_id
        id_to_slot[new_id] = slot
        filled[slot] = True

        index.add_with_ids(vec_np.astype(np.float32), np.array([new_id], dtype=np.int64))

        count = memory["count"]  # type: ignore[index]
        memory["count"] = min(count + 1, self.config.memory_slots)  # type: ignore[index]
        memory["next_slot"] = (slot + 1) % self.config.memory_slots  # type: ignore[index]

        if self._profile_enabled:
            self._profile_stats["slots_written"] += 1.0
            self._profile_stats["index_size"] = float(memory["count"])  # type: ignore[index]
            self._profile_stats["write_time_ms"] += (perf_counter() - write_start) * 1000.0

    # ----- Forward passes -------------------------------------------------
    def forward_episode(self, frames: Sequence[Tensor], actions: Tensor) -> Tensor:
        device = self.device
        h, c = self.init_hidden(device=device)
        memory = self.init_memory()

        logits: List[Tensor] = []
        actions = actions.to(device)
        prev_actions = torch.cat(
            [
                torch.tensor([self.start_action_index], dtype=torch.long, device=device),
                actions[:-1],
            ]
        )

        for step, frame in enumerate(frames):
            frame_tensor = frame.to(device=device)
            frame_embed = self.frame_encoder(frame_tensor).view(1, -1)
            if step == 0:
                retrieved = torch.zeros(1, self.config.memory_dim, device=device)
            else:
                retrieved = self._read_memory(h, memory)

            prev_idx = prev_actions[step].view(1)
            prev_action_embed = self.action_embed(prev_idx)
            lstm_input = torch.cat([frame_embed, retrieved, prev_action_embed], dim=1)
            h, c = self.integrator(lstm_input, (h, c))
            self._write_memory(h, memory)

            logit = self.classifier(self.dropout(h))
            logits.append(logit)

        return torch.cat(logits, dim=0)

    def predict_actions(self, frames: Sequence[Tensor]) -> Tensor:
        device = self.device
        h, c = self.init_hidden(device=device)
        memory = self.init_memory()
        prev_idx = torch.tensor([self.start_action_index], dtype=torch.long, device=device)
        preds: List[Tensor] = []

        for step, frame in enumerate(frames):
            frame_tensor = frame.to(device=device)
            frame_embed = self.frame_encoder(frame_tensor).view(1, -1)
            if step == 0:
                retrieved = torch.zeros(1, self.config.memory_dim, device=device)
            else:
                retrieved = self._read_memory(h, memory)
            prev_action_embed = self.action_embed(prev_idx)
            lstm_input = torch.cat([frame_embed, retrieved, prev_action_embed], dim=1)
            h, c = self.integrator(lstm_input, (h, c))
            self._write_memory(h, memory)
            logit = self.classifier(self.dropout(h))
            pred = logit.argmax(dim=1)
            preds.append(pred)
            prev_idx = pred

        return torch.cat(preds, dim=0)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
