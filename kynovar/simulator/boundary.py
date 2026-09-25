"""Public serialization and checks that an artifact carries no hidden laws."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kynovar.simulator.state import BODY_OBSERVATION_FIELDS, BodyObservation, Observation
from kynovar.simulator.world import InternalState

# Keys that belong to hidden laws, evaluation records, or the universe seed.
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "k",
        "p",
        "gx",
        "gy",
        "coefficient",
        "stiffness",
        "rest_length",
        "softening",
        "restitution",
        "seed",
        "forces",
        "force",
        "force_law",
        "hidden",
        "hidden_parameters",
        "ground_truth",
        "parameters",
        "difficulty",
        "difficulty_description",
        "integrator",
        "law",
        "collisions_enabled",
    }
)

FORBIDDEN_PUBLIC_STRINGS = (
    "pairwise_power_law",
    "constant_gravity",
    "linear_drag",
    "quadratic_drag",
    "semi_implicit_euler",
    "ground_truth",
    "hidden_parameters",
)


class InformationLeakError(RuntimeError):
    """A public record contains a hidden-law field."""


def body_to_dict(body: BodyObservation) -> dict[str, float | int]:
    if tuple(BODY_OBSERVATION_FIELDS) != (
        "id",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "mass",
        "radius",
    ):
        raise InformationLeakError("Body observation fields changed without a schema review.")
    return {
        "id": body.id,
        "x": body.x,
        "y": body.y,
        "vx": body.vx,
        "vy": body.vy,
        "ax": body.ax,
        "ay": body.ay,
        "mass": body.mass,
        "radius": body.radius,
    }


def observation_to_dict(observation: Observation) -> dict[str, Any]:
    return {
        "schema_version": observation.schema_version,
        "universe_id": observation.universe_id,
        "time": observation.time,
        "step_index": observation.step_index,
        "bodies": [body_to_dict(body) for body in observation.bodies],
    }


def trajectory_to_dict(trajectory: Any) -> dict[str, Any]:
    """Serialize a laboratory Trajectory. InternalState is rejected."""
    if isinstance(trajectory, InternalState):
        raise InformationLeakError("Refusing to serialize internal state as a public trajectory.")
    from kynovar.laboratory.experiment import Trajectory

    if not isinstance(trajectory, Trajectory):
        raise TypeError(f"Expected a Trajectory, got {type(trajectory).__name__}.")
    payload = {
        "record_type": "trajectory",
        "schema_version": trajectory.schema_version,
        "universe_id": trajectory.universe_id,
        "experiment_id": trajectory.experiment_id,
        "dt": trajectory.dt,
        "observations": [observation_to_dict(frame) for frame in trajectory.observations],
    }
    assert_public_record(payload)
    return payload


def iter_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.append(str(key))
            found.extend(iter_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(iter_keys(item))
    return found


def assert_public_record(payload: Mapping[str, Any]) -> None:
    leaked = set(iter_keys(payload)) & FORBIDDEN_PUBLIC_KEYS
    if leaked:
        raise InformationLeakError(f"Public record contains hidden keys: {sorted(leaked)}.")
    rendered = repr(payload)
    for token in FORBIDDEN_PUBLIC_STRINGS:
        if token in rendered:
            raise InformationLeakError(f"Public record contains hidden token {token!r}.")
