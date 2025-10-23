from __future__ import annotations

PLANLET_SCHEMA = {
    "type": "object",
    "required": [
        "planlet_id",
        "kind",
        "seed_frame_id",
        "format",
        "side",
        "goal",
        "script",
    ],
    "properties": {
        "planlet_id": {"type": "string"},
        "id": {"type": "string"},
        "kind": {
            "enum": [
                "BATTLE",
                "NAVIGATE_TO",
                "HEAL_AT_CENTER",
                "BUY_ITEM",
                "TALK_TO",
                "OPEN_MENU",
                "MENU_SEQUENCE",
                "USE_ITEM",
                "INTERACT",
                "PICKUP_ITEM",
                "WAIT",
                "HANDLE_ENCOUNTER",
            ]
        },
        "seed_frame_id": {"type": "integer"},
        "format": {"type": "string"},
        "side": {"enum": ["p1", "p2"]},
        "goal": {"type": "string"},
        "rationale": {"type": "string"},
        "args": {"type": "object"},
        "pre": {"type": "array", "items": {"type": "object"}},
        "post": {"type": "array", "items": {"type": "object"}},
        "hints": {"type": "object"},
        "timeout_steps": {"type": "integer"},
        "recovery": {"type": "array", "items": {"type": "object"}},
        "script": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "op": {"type": "string"},
                    "actor": {"type": "string"},
                    "move": {"type": "string"},
                    "target": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                    "fallback": {"type": "object"},
                    "buttons": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    "additionalProperties": True,
}
