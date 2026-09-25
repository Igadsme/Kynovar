"""Evaluation-only export of hidden laws.

Discovery, planning, and theory code must not import this module.
"""

from __future__ import annotations

from typing import Any

from kynovar.simulator.world import World


def ground_truth_record(world: World) -> dict[str, Any]:
    """Hidden laws for scoring. This is not an observation."""
    state = world.internal_state()
    return {
        "record_type": "ground_truth",
        "warning": (
            "Evaluation only. Not an observation. Do not supply this record to "
            "discovery, planning, or theory code."
        ),
        "universe_id": state.universe_id,
        "seed": state.seed,
        "difficulty": state.difficulty,
        "difficulty_description": state.difficulty_description,
        "integrator": state.integrator,
        "dimension": state.dimension,
        "dt": state.dt,
        "time": state.time,
        "step_index": state.step_index,
        "collisions_enabled": state.collisions_enabled,
        "restitution": state.restitution,
        "forces": [
            {
                "law": name,
                "parameters": {key: value for key, value in parameters},
            }
            for name, parameters in state.forces
        ],
    }
