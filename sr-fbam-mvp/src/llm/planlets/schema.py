from __future__ import annotations

PLANLET_SCHEMA = {
    "type": "object",
    "required": [
        "planlet_id",
        "seed_frame_id",
        "format",
        "side",
        "goal",
        "script"
    ],
    "properties": {
        "planlet_id": {"type": "string"},
        "seed_frame_id": {"type": "integer"},
        "format": {"type": "string"},
        "side": {"enum": ["p1", "p2"]},
        "goal": {"type": "string"},
        "rationale": {"type": "string"},
        "preconditions": {"type": "array", "items": {"type": "string"}},
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
                    "target": {"type": "string"},
                    "fallback": {"type": "object"}
                }
            }
        }
    },
    "additionalProperties": True
}
