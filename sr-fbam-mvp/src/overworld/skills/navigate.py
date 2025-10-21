"""
NavigateSkill implements simple graph-based path finding for overworld planlets.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Sequence, Tuple

from .base import BaseSkill, SkillProgress, SkillStatus
from ..graph.overworld_memory import OverworldMemory


class NavigateSkill(BaseSkill):
    name = "NavigateSkill"

    ACTION_UP = {"kind": "button", "label": "UP"}
    ACTION_DOWN = {"kind": "button", "label": "DOWN"}
    ACTION_LEFT = {"kind": "button", "label": "LEFT"}
    ACTION_RIGHT = {"kind": "button", "label": "RIGHT"}
    ACTION_WAIT = {"kind": "wait"}
    LEGAL_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_WAIT)

    def __init__(self) -> None:
        super().__init__()
        self.target: Optional[Tuple[int, int]] = None
        self.path: Sequence[Tuple[int, int]] = []
        self.status: SkillStatus = SkillStatus.NOT_STARTED
        self._last_tile_path: Sequence[str] = ()

    def legal_actions(self, observation: Dict[str, object], graph: OverworldMemory) -> tuple[Dict[str, object], ...]:
        return self.LEGAL_ACTIONS

    def on_enter(self, planlet: any, graph: OverworldMemory) -> None:
        super().on_enter(planlet, graph)
        target_args = getattr(planlet, "args", {}) or {}
        target_info = target_args.get("target", {})
        if isinstance(target_info, dict):
            tile = target_info.get("tile")
            if isinstance(tile, Sequence) and len(tile) >= 2:
                self.target = (int(tile[0]), int(tile[1]))
        self.status = SkillStatus.IN_PROGRESS if self.target else SkillStatus.STALLED
        self._recompute_path(graph)

    def select_action(self, observation: Dict[str, object], graph: OverworldMemory) -> Dict[str, object]:
        if self.status == SkillStatus.STALLED or not self.target:
            return {"kind": "wait", "meta": {"reason": "no-path"}}

        current_tile_id = graph.latest_player_tile()
        if current_tile_id is None:
            return self.ACTION_WAIT

        current_coords = self._coords_from_tile_id(current_tile_id)
        summary = self.summary()
        gate_decision = str(summary.gate_stats.get("decision", "")) if summary is not None else ""

        if gate_decision in {"CACHE_HIT", "REUSE"}:
            reuse_coord = self._reuse_from_history(graph, current_tile_id)
            if reuse_coord is not None:
                action = self._direction_to_action(current_coords, reuse_coord)
                if action != self.ACTION_WAIT:
                    self.set_planner_hint(
                        {
                            "mode": gate_decision,
                            "strategy": "reuse-last-path",
                            "path_remaining": len(self.path),
                        }
                    )
                    return action

        if gate_decision == "EXTRACT":
            self._recompute_path(graph)

        if not self.path:
            if current_coords == self.target:
                self.status = SkillStatus.SUCCEEDED
                return self.ACTION_WAIT
            self._recompute_path(graph)

        if self.path and current_coords == self.path[0]:
            self.path = self.path[1:]

        if not self.path:
            if current_coords == self.target:
                self.status = SkillStatus.SUCCEEDED
            return self.ACTION_WAIT

        next_coord = self.path[0]
        action = self._direction_to_action(current_coords, next_coord)
        if action == self.ACTION_WAIT:
            # No valid direction; attempt recompute once
            self._recompute_path(graph)
            if self.path:
                next_coord = self.path[0]
                action = self._direction_to_action(current_coords, next_coord)
            else:
                self.status = SkillStatus.STALLED
        return action

    def progress(self, graph: OverworldMemory) -> SkillProgress:
        if self.status == SkillStatus.STALLED:
            return SkillProgress(status=SkillStatus.STALLED, reason="no-path")
        if self.status == SkillStatus.SUCCEEDED:
            return SkillProgress(status=SkillStatus.SUCCEEDED)
        if not self.target:
            return SkillProgress(status=SkillStatus.STALLED, reason="missing-target")

        current_tile_id = graph.latest_player_tile()
        if current_tile_id is None:
            return SkillProgress(status=SkillStatus.IN_PROGRESS)
        current_coords = self._coords_from_tile_id(current_tile_id)
        if current_coords == self.target and not self.path:
            self.status = SkillStatus.SUCCEEDED
            return SkillProgress(status=SkillStatus.SUCCEEDED)
        return SkillProgress(status=SkillStatus.IN_PROGRESS)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _recompute_path(self, graph: OverworldMemory) -> None:
        self.path = []
        if not self.target:
            self.status = SkillStatus.STALLED
            return

        start_tile_id = graph.latest_player_tile()
        if not start_tile_id:
            self.status = SkillStatus.STALLED
            return

        target_tile_id = None
        parent: Dict[str, Optional[str]] = {start_tile_id: None}
        queue = deque([start_tile_id])

        while queue:
            current_id = queue.popleft()
            if self._coords_from_tile_id(current_id) == self.target:
                target_tile_id = current_id
                break
            neighbors = list(graph.follow(src=current_id, relation="adjacent"))
            for neighbor in neighbors:
                node_id = neighbor.node_id
                if node_id not in parent and neighbor.attributes.get("passable", True):
                    parent[node_id] = current_id
                    queue.append(node_id)
            warp_neighbors = graph.follow(src=current_id, relation="warp_exit")
            for node in warp_neighbors:
                node_id = node.node_id
                if node_id not in parent:
                    parent[node_id] = current_id
                    queue.append(node_id)

        if target_tile_id is None:
            self.status = SkillStatus.STALLED
            return

        tile_path = []
        cursor = target_tile_id
        while cursor is not None:
            tile_path.append(cursor)
            cursor = parent.get(cursor)
        tile_path.reverse()

        coords_path = [self._coords_from_tile_id(tile_id) for tile_id in tile_path]

        if coords_path and coords_path[0] == self._coords_from_tile_id(start_tile_id):
            coords_path = coords_path[1:]

        self.path = tuple(coords_path)
        if not self.path:
            self.status = SkillStatus.SUCCEEDED
        else:
            self.status = SkillStatus.IN_PROGRESS
        self._last_tile_path = tuple(tile_path)
        if tile_path:
            self._write_last_path_edges(graph, tile_path)
        self.set_planner_hint({"target": list(self.target) if self.target else None, "path_len": len(self.path)})

    @staticmethod
    def _coords_from_tile_id(tile_id: str) -> Tuple[int, int]:
        try:
            _, _, x_str, y_str = tile_id.split(":")
            return int(x_str), int(y_str)
        except ValueError:
            return (0, 0)

    def _reuse_from_history(self, graph: OverworldMemory, tile_id: str) -> Optional[Tuple[int, int]]:
        if self._last_tile_path and tile_id in self._last_tile_path:
            idx = self._last_tile_path.index(tile_id)
            remaining_tiles = self._last_tile_path[idx + 1 :]
            if remaining_tiles:
                self.path = tuple(self._coords_from_tile_id(tid) for tid in remaining_tiles)
        if not self.path:
            next_nodes = graph.follow(src=tile_id, relation="last_path")
            if next_nodes:
                self.path = tuple([self._coords_from_tile_id(node.node_id) for node in next_nodes])
        if not self.path:
            return None
        return self.path[0]

    def _write_last_path_edges(self, graph: OverworldMemory, tile_ids: Sequence[str]) -> None:
        if len(tile_ids) < 2:
            return
        planlet_id = getattr(self.planlet, "id", None)
        for src_id, dst_id in zip(tile_ids[:-1], tile_ids[1:]):
            graph.write(OverworldMemory.make_last_path_edge(src_id, dst_id, planlet_id=planlet_id))

    def _direction_to_action(self, current: Tuple[int, int], target: Tuple[int, int]) -> Dict[str, object]:
        cur_x, cur_y = current
        tgt_x, tgt_y = target
        delta_x = tgt_x - cur_x
        delta_y = tgt_y - cur_y
        if abs(delta_x) > 1 or abs(delta_y) > 1:
            return {"kind": "wait", "meta": {"reason": "warp_exit", "target": list(target)}}
        if tgt_y < cur_y:
            return self.ACTION_UP
        if tgt_y > cur_y:
            return self.ACTION_DOWN
        if tgt_x < cur_x:
            return self.ACTION_LEFT
        if tgt_x > cur_x:
            return self.ACTION_RIGHT
        return self.ACTION_WAIT
