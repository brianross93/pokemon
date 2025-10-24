"""
Planner prompt utilities for the overworld agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.overworld.graph.overworld_memory import OverworldMemory
from src.srfbam.tasks.events import PlanletEvent

from .planner_llm import PlanletSpec


@dataclass(frozen=True)
class PlannerPrompt:
    goal: Optional[str]
    entity_summary: List[Dict[str, object]]
    menu_context: List[Dict[str, object]]
    plan_history: List[Dict[str, object]]
    outcomes: Dict[str, object]
    pending_planlets: List[Dict[str, object]]


class PlannerPromptBuilder:
    """Constructs compact planner prompts that capture the latest context."""

    def __init__(self, *, max_history: int = 6) -> None:
        self._max_history = max(1, int(max_history))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build(
        self,
        *,
        goal: Optional[str],
        memory: OverworldMemory,
        history: Sequence[PlanletEvent],
        pending: Sequence[PlanletSpec],
    ) -> PlannerPrompt:
        entity_summary = self._summarise_entities(memory)
        menu_context = self._summarise_menus(memory)
        history_entries, outcomes = self._summarise_history(history)
        pending_entries = [
            {
                "id": spec.id,
                "kind": spec.kind,
                "timeout_steps": spec.timeout_steps,
            }
            for spec in pending
        ]
        return PlannerPrompt(
            goal=goal,
            entity_summary=entity_summary,
            menu_context=menu_context,
            plan_history=history_entries,
            outcomes=outcomes,
            pending_planlets=pending_entries,
        )

    def format_prompt(self, prompt: PlannerPrompt) -> str:
        lines: List[str] = []

        goal_text = prompt.goal or "N/A"
        lines.append(f"Goal: {goal_text}")
        lines.append("")
        lines.append(
            "Supported planlet kinds: NAVIGATE_TO, INTERACT, MENU_SEQUENCE (buttons array), OPEN_MENU, USE_ITEM, HANDLE_ENCOUNTER."
        )
        lines.append(
            "When menus are closed, prefer NAVIGATE_TO with args.target.tile=[x,y] to reach a tile, or INTERACT with args.entity_id referencing a sprite/NPC. When menus are open, emit MENU_SEQUENCE planlets with explicit buttons (e.g., START, A, A)."
        )
        lines.append("")

        lines.append("Entity Summary:")
        for entry in prompt.entity_summary:
            lines.append(f"- {entry['type']}: {entry['count']}")
        if not prompt.entity_summary:
            lines.append("- (no entities observed)")
        lines.append("")

        lines.append("Menu Context:")
        if prompt.menu_context:
            for menu in prompt.menu_context:
                path = " > ".join(menu.get("path", [])) or "(root)"
                lines.append(f"- path={path} open={menu.get('open', False)} state={menu.get('state')}")
        else:
            lines.append("- (no menus open or detected)")
        lines.append("")

        lines.append("Recent Plan History:")
        if prompt.plan_history:
            for entry in prompt.plan_history:
                reason = entry.get("reason")
                reason_text = f" reason={reason}" if reason else ""
                lines.append(
                    f"- [{entry['status']}] {entry['kind']} ({entry['planlet_id']}){reason_text}"
                )
        else:
            lines.append("- (no history recorded)")
        lines.append("")

        outcomes = prompt.outcomes
        lines.append(
            f"Outcomes: successes={outcomes.get('successes', 0)} "
            f"failures={sum(outcomes.get('failures', {}).values())} "
            f"delta={outcomes.get('delta', 0)}"
        )
        failure_taxonomy = outcomes.get("failures", {})
        if failure_taxonomy:
            lines.append("Failure Taxonomy:")
            for reason, count in failure_taxonomy.items():
                lines.append(f"- {reason}: {count}")
        else:
            lines.append("Failure Taxonomy: none")
        lines.append("")

        lines.append("Pending Planlets:")
        if prompt.pending_planlets:
            for spec in prompt.pending_planlets:
                lines.append(
                    f"- {spec['kind']} ({spec['id']}) timeout={spec['timeout_steps']}"
                )
        else:
            lines.append("- (no pending planlets)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _summarise_entities(self, memory: OverworldMemory) -> List[Dict[str, object]]:
        summary = memory.summarise_nodes()
        return [
            {"type": entity_type, "count": count}
            for entity_type, count in sorted(summary.items())
        ]

    def _summarise_history(
        self, history: Sequence[PlanletEvent]
    ) -> tuple[List[Dict[str, object]], Dict[str, object]]:
        history_entries: List[Dict[str, object]] = []
        failure_counts: Dict[str, int] = {}
        successes = 0

        for event in history:
            if event.status == "PLANLET_COMPLETE":
                successes += 1
            elif event.status == "PLANLET_STALLED" and event.reason:
                failure_counts[event.reason] = failure_counts.get(event.reason, 0) + 1

        recent = history[-self._max_history :]
        for event in recent:
            history_entries.append(
                {
                    "planlet_id": event.planlet_id,
                    "kind": event.planlet_kind,
                    "status": event.status,
                    "reason": event.reason,
                }
            )

        failures_total = sum(failure_counts.values())
        outcomes = {
            "successes": successes,
            "failures": failure_counts,
            "delta": successes - failures_total,
        }
        return history_entries, outcomes

    def _summarise_menus(self, memory: OverworldMemory) -> List[Dict[str, object]]:
        context: List[Dict[str, object]] = []
        assoc_fn = getattr(memory, "assoc", None)
        if assoc_fn is None:
            return context
        for node in assoc_fn(type_="MenuState"):
            attributes = dict(node.attributes)
            context.append(
                {
                    "path": list(attributes.get("path", [])),
                    "open": bool(attributes.get("open")),
                    "node_id": node.node_id,
                    "state": attributes.get("state"),
                }
            )
        return context


__all__ = ["PlannerPromptBuilder", "PlannerPrompt"]
