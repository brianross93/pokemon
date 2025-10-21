"""
MenuSkill orchestrates path-driven menu navigation with recovery hooks.
"""

from __future__ import annotations

from typing import Dict

from .menu_controller import MenuDrivenSkill
from .neural import BUTTONS


class MenuSkill(MenuDrivenSkill):
    name = "MenuSkill"

    def legal_actions(self, observation: Dict[str, object], graph: object) -> tuple[Dict[str, object], ...]:
        return tuple(BUTTONS[label] for label in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "WAIT"))


__all__ = ["MenuSkill"]
