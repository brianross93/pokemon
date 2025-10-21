"""
Environment abstractions for Pokemon battle agents.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypedDict


class LegalAction(TypedDict):
    """Canonical action description used by the policy head."""

    kind: str  # "move" | "switch" | "forfeit"
    index: int
    meta: Dict[str, object]


class BattleObs(TypedDict, total=False):
    """High-level battle observation emitted by adapters."""

    turn: int
    my_active: Dict[str, object]
    opp_active: Dict[str, object]
    my_party: List[Dict[str, object]]
    opp_party: List[Dict[str, object]]
    field: Dict[str, object]
    raw: Dict[str, object]


class EnvAdapter(ABC):
    """
    Common interface for Pokemon battle environments.

    Concrete implementations (Blue RAM, Showdown replay/live adapters)
    should translate their native telemetry into `BattleObs` and expose
    the set of legal actions each turn.
    """

    @abstractmethod
    def reset(self) -> BattleObs:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def observe(self) -> BattleObs:
        """Return the latest observation without advancing the environment."""

    @abstractmethod
    def legal_actions(self) -> List[LegalAction]:
        """Return the set of currently legal actions."""

    @abstractmethod
    def step(self, action: LegalAction) -> BattleObs:
        """Apply an action and advance the environment one turn."""

    def close(self) -> None:
        """Optional cleanup hook."""
        return None

