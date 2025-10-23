"""
Middleware scaffolding that connects emulator telemetry to the symbolic
knowledge modules. The package exposes interfaces for Pokemon Blue adapters
and controller utilities that decide whether to query cached beliefs or defer
to an LLM planner.
"""

from .pokemon_adapter import (  # noqa: F401
    ObservationBundle,
    PokemonAction,
    PokemonBlueAdapter,
    PokemonObservation,
)
from .controller import (  # noqa: F401
    GateDecision,
    GateMode,
    SymbolicController,
)
