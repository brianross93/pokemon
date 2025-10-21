"""
Utilities for synchronising learned slot embeddings with the typed overworld graph.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from src.overworld.memory.slot_bank import Slot, SlotBank
from src.pkmn_battle.graph.memory import GraphMemory
from src.pkmn_battle.graph.schema import Node, WriteOp


@dataclass
class HybridMemoryAdapter:
    """
    Bridges the slot bank and the typed overworld graph.

    The adapter exposes two primitives:
      * `project_slots_to_graph` materialises slot embeddings as `SlotView` nodes.
      * `ingest_graph_slots` rebuilds slots from previously materialised nodes.
    """

    max_embedding_head: int = 32

    def project_slots_to_graph(
        self,
        memory: GraphMemory,
        slot_bank: SlotBank,
        *,
        context: Optional[str] = None,
    ) -> int:
        """
        Emit SlotView nodes for any slots that are not yet represented in the graph.

        Returns the number of new nodes written.
        """

        existing_ids = {
            node.node_id
            for node in memory.assoc(type_="SlotView")
            if not context or node.attributes.get("context") == context
        }

        created = 0
        for slot in slot_bank.slots():
            if context and slot.metadata.get("context") != context:
                continue

            node_id = slot.metadata.get("slot_node_id") or self._make_node_id(slot, context)
            slot.metadata.setdefault("slot_node_id", node_id)

            if node_id in existing_ids:
                continue

            vector = slot.vector.detach().to("cpu").flatten()
            embedding = vector.tolist()

            attributes = {
                "context": slot.metadata.get("context"),
                "confidence": float(slot.confidence),
                "embedding": embedding,
                "embedding_head": embedding[: self.max_embedding_head],
                "embedding_dim": len(embedding),
                "embedding_norm": float(vector.norm().item()) if vector.numel() else 0.0,
                "slot_node_id": node_id,
                "metadata": self._sanitise_metadata(slot.metadata),
            }

            memory.write(WriteOp(kind="node", payload=Node(type="SlotView", node_id=node_id, attributes=attributes)))
            existing_ids.add(node_id)
            created += 1

        return created

    def ingest_graph_slots(
        self,
        memory: GraphMemory,
        slot_bank: SlotBank,
        *,
        context: Optional[str] = None,
    ) -> int:
        """
        Reconstruct slots from SlotView nodes that were previously materialised in the graph.

        Returns the number of slots added to the slot bank.
        """

        added = 0

        for node in memory.assoc(type_="SlotView"):
            if context and node.attributes.get("context") != context:
                continue

            node_id = node.node_id
            if slot_bank.contains_metadata("slot_node_id", node_id):
                continue

            embedding = node.attributes.get("embedding")
            if embedding is None:
                continue

            vector = torch.tensor(embedding, dtype=torch.float32)
            metadata = self._normalise_metadata(node.attributes.get("metadata") or {})
            metadata.setdefault("slot_node_id", node_id)
            metadata.setdefault("context", node.attributes.get("context"))

            confidence = float(node.attributes.get("confidence", 1.0))
            slot_bank.add_slot(vector, confidence=confidence, metadata=metadata)
            added += 1

        return added

    def _make_node_id(self, slot: Slot, context: Optional[str]) -> str:
        vector = slot.vector.detach().to("cpu").flatten()
        digest = hashlib.sha1(vector.numpy().tobytes()).hexdigest()[:16] if vector.numel() else "empty"
        ctx = context or slot.metadata.get("context") or "unknown"
        return f"slot:{ctx}:{digest}"

    def _sanitise_metadata(self, metadata: Dict[str, object]) -> Dict[str, object]:
        """Convert metadata into JSON-serialisable primitives."""

        return {key: self._to_jsonable(value) for key, value in metadata.items()}

    def _normalise_metadata(self, metadata: Dict[str, object]) -> Dict[str, object]:
        """
        Recreate a mutable metadata dictionary from the stored representation.
        """

        normalised: Dict[str, object] = {}
        for key, value in metadata.items():
            normalised[key] = value
        return normalised

    def _to_jsonable(self, value: object) -> object:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        return repr(value)


__all__ = ["HybridMemoryAdapter"]
