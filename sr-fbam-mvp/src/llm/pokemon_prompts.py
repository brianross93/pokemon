"""Prompts for Pokemon Blue learning system."""

from typing import Dict, Any

class PokemonPrompts:
    """Prompts for Pokemon Blue decision making."""
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for Pokemon Blue decision making."""
        return """You operate the SR-FBAM gameplay loop for Pokemon Blue. Each prompt provides the current SR-FBAM summary: phase, gate statistics, feature snapshot, memory usage, and recent action context.

Choose exactly one action per turn from: WALK, FIGHT, A, UP, DOWN, LEFT, RIGHT, WAIT.

Your objective is to explore efficiently, enter battles in grass when appropriate, and progress toward finding target encounters. Reply with a single uppercase action token and nothing else."""

    @staticmethod
    def get_decision_prompt(context: Dict[str, Any]) -> str:
        """Get the decision prompt based on current context."""
        phase = context.get("phase", "unknown")
        control_ready = context.get("control_ready", False)
        area = context.get("area", {})
        area_id = area.get("id", "unknown")
        mode = area.get("mode", "unknown")
        step = area.get("step", 0)

        srfbam = context.get("srfbam", {})
        gate = srfbam.get("gate", {})
        position = srfbam.get("position", "unknown")
        state_flags = srfbam.get("state_flags", {})
        snapshot = srfbam.get("feature_snapshot", {})
        memory = srfbam.get("memory") or {}
        embedding_norm = srfbam.get("embedding_norm", 0.0)

        recent = context.get("recent_action", {})
        target_found = context.get("target_found", False)
        stuck_frames = context.get("stuck_frames", 0)

        prompt = f"""SR-FBAM controller snapshot
- Phase: {phase} (control_ready={control_ready})
- Area: {area_id} | mode={mode} | step={step}
- Gate decision: {gate.get('decision', 'N/A')} (cache={gate.get('cache_hit_rate', 0.0):.3f}, reuse={gate.get('reuse_rate', 0.0):.3f}, extract={gate.get('extract_rate', 0.0):.3f})
- Approx position: {position}
- State flags: grass={state_flags.get('in_grass', False)} battle={state_flags.get('in_battle', False)} joy_locked={state_flags.get('joy_ignored', False)}
- Feature snapshot: area_fraction={snapshot.get('area_fraction', 0.0):.3f} joy={snapshot.get('joy_ignore_level', 0.0):.3f} menu_state={snapshot.get('menu_state_level', 0.0):.3f} cursor={snapshot.get('menu_cursor_level', 0.0):.3f}
- Embedding norm: {embedding_norm:.3f}
- Memory usage: cache_entries={memory.get('cache_entries', 0)} summary_files={memory.get('summary_files', 0)} steps_recorded={memory.get('steps_recorded', 0)}
- Recent action: {recent.get('name', 'unknown')} (source={recent.get('source', 'unknown')})
- Target found: {target_found} | stuck_frames={stuck_frames}

Select the next action. Respond with exactly one uppercase token: WALK, FIGHT, A, UP, DOWN, LEFT, RIGHT, or WAIT."""

        return prompt

    @staticmethod
    def get_learning_prompt(context: Dict[str, Any], previous_actions: list) -> str:
        """Get the learning prompt for reflecting on past actions."""
        area = context.get("area", {})
        step = area.get("step", context.get("step_count", 0))
        recent = ", ".join(previous_actions[-10:]) if previous_actions else "N/A"
        srfbam = context.get("srfbam", {})
        position = srfbam.get("position", "unknown")
        phase = context.get("phase", "unknown")

        prompt = f"""Reflection checkpoint after {step} steps.

Last actions: {recent}
Current phase: {phase}
Approx position: {position}
Target found: {context.get('target_found', False)}

What have you learned about navigating this area? Suggest a brief strategy (1-2 sentences) and end with the next single-word action."""
        
        return prompt

