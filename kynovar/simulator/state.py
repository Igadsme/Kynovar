"""Observable kinematic state. These types must not grow hidden-law fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBSERVATION_SCHEMA_VERSION = 1

BODY_OBSERVATION_FIELDS = (
    "id",
    "x",
    "y",
    "vx",
    "vy",
    "ax",
    "ay",
    "mass",
    "radius",
)


class SimulatorError(ValueError):
    """Invalid simulator input or an inconsistent state."""


def require_finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise SimulatorError(f"{name} must be finite, got {value!r}.")
    return number


@dataclass(frozen=True)
class BodyInit:
    """Initial conditions an experimenter is allowed to choose."""

    id: int
    x: float
    y: float
    vx: float
    vy: float
    mass: float
    radius: float

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, int):
            raise SimulatorError(f"body id must be an int, got {self.id!r}.")
        object.__setattr__(self, "x", require_finite(self.x, "x"))
        object.__setattr__(self, "y", require_finite(self.y, "y"))
        object.__setattr__(self, "vx", require_finite(self.vx, "vx"))
        object.__setattr__(self, "vy", require_finite(self.vy, "vy"))
        mass = require_finite(self.mass, "mass")
        radius = require_finite(self.radius, "radius")
        if mass <= 0.0:
            raise SimulatorError(f"mass must be positive, got {mass}.")
        if radius < 0.0:
            raise SimulatorError(f"radius must be non-negative, got {radius}.")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class BodyObservation:
    """Measurements of one body at one time. Acceleration is F/m from continuous forces."""

    id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    mass: float
    radius: float


@dataclass(frozen=True)
class Observation:
    """Everything Kynovar is allowed to measure at one instant."""

    schema_version: int
    universe_id: str
    time: float
    step_index: int
    bodies: tuple[BodyObservation, ...]
