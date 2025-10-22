"""
Stub plan compiler that binds validated planlets to skill identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from .planner_llm import PlanBundle, PlanletSpec


@dataclass(frozen=True)
class CompiledPlanlet:
    """Runtime representation of an executable planlet."""

    spec: PlanletSpec
    skill: str


@dataclass(frozen=True)
class CompiledPlan:
    """Compiled plan ready for execution."""

    plan_id: str
    goal: Optional[str]
    planlets: List[CompiledPlanlet]


class PlanCompilationError(RuntimeError):
    """Raised when no skill binding exists for a planlet spec."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class PlanCompiler:
    """
    Resolves planlet kinds to skill implementations.

    The default registry is intentionally small for now—additional bindings can be
    injected via the constructor as the overworld skill library grows.
    """

    DEFAULT_REGISTRY: Mapping[str, str] = {
        "NAVIGATE_TO": "NavigateSkill",
        "HEAL_AT_CENTER": "HealSkill",
        "BUY_ITEM": "ShopSkill",
        "TALK_TO": "TalkSkill",
        "OPEN_MENU": "MenuSkill",
        "MENU_SEQUENCE": "MenuSequenceSkill",
        "USE_ITEM": "UseItemSkill",
        "INTERACT": "InteractSkill",
        "PICKUP_ITEM": "PickupSkill",
        "WAIT": "WaitSkill",
        "HANDLE_ENCOUNTER": "EncounterSkill",
    }

    def __init__(self, registry: Optional[Mapping[str, str]] = None) -> None:
        self._registry = dict(registry or self.DEFAULT_REGISTRY)

    def compile(self, bundle: PlanBundle) -> CompiledPlan:
        if not bundle.planlets:
            raise PlanCompilationError("Received bundle with no planlets to compile.")
        if len(bundle.planlets) > 8:
            raise PlanCompilationError("Plan bundle exceeded maximum of 8 planlets.")

        compiled: List[CompiledPlanlet] = []
        for planlet in bundle.planlets:
            if planlet.timeout_steps <= 0:
                raise PlanCompilationError("Planlet timeout_steps must be > 0.")
            skill = self._registry.get(planlet.kind)
            if skill is None:
                raise PlanCompilationError(f"No skill registered for planlet kind '{planlet.kind}'.")
            compiled.append(CompiledPlanlet(spec=planlet, skill=skill))
        return CompiledPlan(plan_id=bundle.plan_id, goal=bundle.goal, planlets=compiled)

    def registry(self) -> Dict[str, str]:
        """Return a copy of the current kind→skill map."""

        return dict(self._registry)
