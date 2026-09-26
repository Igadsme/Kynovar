"""3D worlds with the same measurement boundary as the 2D `World`.

The 2D simulator and its observation schema are unchanged. This module adds
a parallel 3D world, observation type, and laboratory. Force laws are reused:
pairwise power laws, drags, and springs are written for any dimension.
`observe` returns kinematics only; `internal_state` is evaluation-only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kynovar.simulator.forces import ForceLaw
from kynovar.simulator.integrator import Integrator, SemiImplicitEuler
from kynovar.simulator.state import SimulatorError, require_finite

BODY3D_OBSERVATION_FIELDS = ("id", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "mass", "radius")


@dataclass(frozen=True)
class BodyInit3D:
    id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float
    radius: float

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, int):
            raise SimulatorError(f"body id must be an int, got {self.id!r}.")
        for name in ("x", "y", "z", "vx", "vy", "vz", "mass", "radius"):
            object.__setattr__(self, name, require_finite(getattr(self, name), name))
        if self.mass <= 0.0:
            raise SimulatorError(f"mass must be positive, got {self.mass}.")
        if self.radius < 0.0:
            raise SimulatorError(f"radius must be non-negative, got {self.radius}.")


@dataclass(frozen=True)
class BodyObservation3D:
    id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    mass: float
    radius: float


@dataclass(frozen=True)
class Observation3D:
    universe_id: str
    time: float
    step_index: int
    bodies: tuple[BodyObservation3D, ...]


class World3D:
    def __init__(self, bodies: Sequence[BodyInit3D], force_laws: Sequence[ForceLaw], *, dt: float, universe_id: str, integrator: Integrator | None = None) -> None:
        if not bodies:
            raise SimulatorError("A world needs at least one body.")
        if dt <= 0.0 or not np.isfinite(dt):
            raise SimulatorError(f"dt must be positive and finite, got {dt}.")
        ids = [body.id for body in bodies]
        if len(set(ids)) != len(ids):
            raise SimulatorError(f"Body ids must be unique, got {ids}.")
        for law in force_laws:
            if not isinstance(law, ForceLaw):
                raise SimulatorError(f"Expected a ForceLaw, got {type(law).__name__}.")
        self._universe_id = str(universe_id)
        self._dt = float(dt)
        self._integrator = integrator or SemiImplicitEuler()
        self._force_laws = tuple(force_laws)
        self._ids = tuple(ids)
        self._positions = np.array([[b.x, b.y, b.z] for b in bodies], dtype=np.float64)
        self._velocities = np.array([[b.vx, b.vy, b.vz] for b in bodies], dtype=np.float64)
        self._masses = np.array([b.mass for b in bodies], dtype=np.float64)
        self._radii = np.array([b.radius for b in bodies], dtype=np.float64)
        self._step_index = 0

    @property
    def universe_id(self) -> str:
        return self._universe_id

    def fork(self, bodies: Sequence[BodyInit3D], dt: float) -> "World3D":
        return World3D(bodies, self._force_laws, dt=dt, universe_id=self._universe_id, integrator=type(self._integrator)())

    def _accelerations(self, positions, velocities) -> np.ndarray:
        total = np.zeros_like(positions)
        for law in self._force_laws:
            contribution = np.asarray(law.forces(positions, velocities, self._masses), dtype=np.float64)
            if contribution.shape != positions.shape:
                raise SimulatorError(f"{law.name} returned shape {contribution.shape}, expected {positions.shape}.")
            total += contribution
        return total / self._masses[:, None]

    def step(self) -> None:
        self._positions, self._velocities = self._integrator.step(self._positions, self._velocities, self._accelerations, self._dt)
        self._step_index += 1

    def observe(self) -> Observation3D:
        a = self._accelerations(self._positions, self._velocities)
        p, v = self._positions, self._velocities
        bodies = tuple(
            BodyObservation3D(self._ids[i], *map(float, p[i]), *map(float, v[i]), *map(float, a[i]), float(self._masses[i]), float(self._radii[i]))
            for i in range(len(self._ids))
        )
        return Observation3D(self._universe_id, self._step_index * self._dt, self._step_index, bodies)

    def internal_state(self) -> dict:
        """Hidden laws and full state. For evaluation only."""
        return {
            "universe_id": self._universe_id,
            "dimension": 3,
            "forces": [(law.name, dict(law.hidden_parameters())) for law in self._force_laws],
            "bodies": self.observe().bodies,
        }


@dataclass(frozen=True)
class Experiment3D:
    objects: tuple[BodyInit3D, ...]
    duration: float
    dt: float

    def __post_init__(self) -> None:
        if not self.objects:
            raise SimulatorError("An experiment needs at least one object.")
        steps = self.duration / self.dt
        if self.dt <= 0 or abs(steps - round(steps)) > 1e-9:
            raise SimulatorError("duration must be a positive integer multiple of dt.")


class Laboratory3D:
    """Runs 3D experiments; returns observations only."""

    def __init__(self, world: World3D) -> None:
        self._world = world
        self.experiments_run = 0

    @property
    def universe_id(self) -> str:
        return self._world.universe_id

    def run(self, experiment: Experiment3D) -> list[Observation3D]:
        if type(experiment) is not Experiment3D:
            raise TypeError("Laboratory3D.run accepts Experiment3D only.")
        simulation = self._world.fork(experiment.objects, experiment.dt)
        frames = [simulation.observe()]
        for _ in range(int(round(experiment.duration / experiment.dt))):
            simulation.step()
            frames.append(simulation.observe())
        self.experiments_run += 1
        return frames


def observation_array_3d(frames: list[Observation3D]) -> np.ndarray:
    """(frames, bodies, 11): x, y, z, vx, vy, vz, ax, ay, az, mass, radius."""
    return np.asarray([[[b.x, b.y, b.z, b.vx, b.vy, b.vz, b.ax, b.ay, b.az, b.mass, b.radius] for b in f.bodies] for f in frames], dtype=np.float64)
