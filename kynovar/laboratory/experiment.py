"""Experiment specifications. These fields are the only controls an experimenter gets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from kynovar.simulator.state import BodyInit, Observation, SimulatorError

_EXPERIMENT_FIELDS = frozenset({"objects", "duration", "dt", "experiment_id"})
_OBJECT_FIELDS = frozenset({"id", "x", "y", "vx", "vy", "mass", "radius"})
_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9_-]+")


def step_count(duration: float, dt: float) -> int:
    """Number of integration steps. `duration` must be an integer multiple of `dt`."""
    if dt <= 0.0 or not np.isfinite(dt):
        raise SimulatorError(f"dt must be positive and finite, got {dt}.")
    if duration < 0.0 or not np.isfinite(duration):
        raise SimulatorError(f"duration must be non-negative and finite, got {duration}.")
    if duration == 0.0:
        return 0
    ratio = duration / dt
    count = int(round(ratio))
    if count < 1 or abs(ratio - count) > 1e-6:
        raise SimulatorError(
            f"duration ({duration}) must be an integer multiple of dt ({dt})."
        )
    return count


@dataclass(frozen=True)
class Experiment:
    objects: tuple[BodyInit, ...]
    duration: float
    dt: float
    experiment_id: str = ""

    def __post_init__(self) -> None:
        objects = tuple(self.objects)
        if not objects:
            raise SimulatorError("An experiment needs at least one object.")
        for body in objects:
            if not isinstance(body, BodyInit):
                raise SimulatorError(
                    f"Experiment objects must be BodyInit instances, got {type(body).__name__}."
                )
        object.__setattr__(self, "objects", objects)
        dt = float(self.dt)
        duration = float(self.duration)
        if dt <= 0.0 or not np.isfinite(dt):
            raise SimulatorError(f"dt must be positive and finite, got {self.dt}.")
        if duration < 0.0 or not np.isfinite(duration):
            raise SimulatorError(f"duration must be non-negative, got {self.duration}.")
        step_count(duration, dt)
        if self.experiment_id and _EXPERIMENT_ID.fullmatch(self.experiment_id) is None:
            raise SimulatorError(
                "experiment_id must contain only letters, digits, underscores, and hyphens."
            )
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "duration", duration)


@dataclass(frozen=True)
class Trajectory:
    schema_version: int
    universe_id: str
    experiment_id: str
    dt: float
    observations: tuple[Observation, ...]

    def positions(self) -> np.ndarray:
        """Array of shape (time, body, 2)."""
        return np.asarray(
            [[[body.x, body.y] for body in frame.bodies] for frame in self.observations],
            dtype=np.float64,
        )

    def velocities(self) -> np.ndarray:
        return np.asarray(
            [[[body.vx, body.vy] for body in frame.bodies] for frame in self.observations],
            dtype=np.float64,
        )

    def accelerations(self) -> np.ndarray:
        return np.asarray(
            [[[body.ax, body.ay] for body in frame.bodies] for frame in self.observations],
            dtype=np.float64,
        )


def experiment_from_mapping(data: Mapping[str, Any]) -> Experiment:
    """Build an experiment from a mapping and reject hidden-law controls."""
    unknown = set(data) - _EXPERIMENT_FIELDS
    if unknown:
        raise SimulatorError(
            "Experiment cannot set "
            f"{sorted(unknown)}. Hidden physical laws are not experiment controls."
        )
    missing = {"objects", "duration", "dt"} - set(data)
    if missing:
        raise SimulatorError(f"Experiment is missing {sorted(missing)}.")
    objects = tuple(_body_from_mapping(item) for item in data["objects"])
    return Experiment(
        objects=objects,
        duration=float(data["duration"]),
        dt=float(data["dt"]),
        experiment_id=str(data.get("experiment_id", "")),
    )


def _body_from_mapping(item: Mapping[str, Any]) -> BodyInit:
    if not isinstance(item, Mapping):
        raise SimulatorError("Each experiment object must be a mapping.")
    body_id = item.get("id")
    if isinstance(body_id, bool) or not isinstance(body_id, int):
        raise SimulatorError(f"object id must be an int, got {body_id!r}.")
    unknown = set(item) - _OBJECT_FIELDS
    if unknown:
        raise SimulatorError(
            f"Experiment object cannot set {sorted(unknown)}. "
            "Acceleration and force-law parameters are not controls."
        )
    missing = _OBJECT_FIELDS - set(item)
    if missing:
        raise SimulatorError(f"Experiment object is missing {sorted(missing)}.")
    return BodyInit(
        id=int(item["id"]),
        x=float(item["x"]),
        y=float(item["y"]),
        vx=float(item["vx"]),
        vy=float(item["vy"]),
        mass=float(item["mass"]),
        radius=float(item["radius"]),
    )


def bodies_from_observation(observation: Observation) -> tuple[BodyInit, ...]:
    """Rebuild experimenter-visible initial conditions from a measurement."""
    return tuple(
        BodyInit(
            id=body.id,
            x=body.x,
            y=body.y,
            vx=body.vx,
            vy=body.vy,
            mass=body.mass,
            radius=body.radius,
        )
        for body in observation.bodies
    )
