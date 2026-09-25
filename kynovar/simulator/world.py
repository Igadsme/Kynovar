"""The simulated world and the measurement boundary.

`observe` returns kinematics only. `internal_state` additionally returns hidden
laws and is reserved for evaluation. Discovery code must not call it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kynovar.simulator.collisions import CollisionSettings, resolve_collisions
from kynovar.simulator.forces import ForceLaw
from kynovar.simulator.integrator import Integrator, SemiImplicitEuler
from kynovar.simulator.state import (
    OBSERVATION_SCHEMA_VERSION,
    BodyInit,
    BodyObservation,
    Observation,
    SimulatorError,
)


@dataclass(frozen=True)
class InternalState:
    """Ground truth. Evaluation may read this. Discovery must not."""

    universe_id: str
    seed: int | None
    difficulty: int | None
    difficulty_description: str
    integrator: str
    dimension: int
    dt: float
    time: float
    step_index: int
    collisions_enabled: bool
    restitution: float | None
    forces: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    bodies: tuple[BodyObservation, ...]


class World:
    """Deterministic 2D world with a strict observation boundary."""

    def __init__(
        self,
        bodies: Sequence[BodyInit],
        force_laws: Sequence[ForceLaw],
        *,
        dt: float,
        universe_id: str,
        seed: int | None = None,
        difficulty: int | None = None,
        difficulty_description: str = "",
        integrator: Integrator | None = None,
        collision: CollisionSettings | None = None,
        dimension: int = 2,
    ) -> None:
        if dimension != 2:
            raise SimulatorError(
                "Only 2D worlds are implemented. 3D simulation is a later milestone."
            )
        if not universe_id:
            raise SimulatorError("universe_id must be a non-empty string.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise SimulatorError("seed must be an int or None.")
        if dt <= 0.0 or not np.isfinite(dt):
            raise SimulatorError(f"dt must be positive and finite, got {dt}.")
        if not bodies:
            raise SimulatorError("A world needs at least one body.")
        ids = [body.id for body in bodies]
        if len(set(ids)) != len(ids):
            raise SimulatorError(f"Body ids must be unique, got {ids}.")
        for law in force_laws:
            if not isinstance(law, ForceLaw):
                raise SimulatorError(f"Expected a ForceLaw, got {type(law).__name__}.")

        self._universe_id = str(universe_id)
        self._seed = seed
        self._difficulty = difficulty
        self._difficulty_description = difficulty_description
        self._dimension = 2
        self._dt = float(dt)
        self._integrator = integrator or SemiImplicitEuler()
        self._collision = collision or CollisionSettings(enabled=False)
        self._force_laws = tuple(force_laws)
        self._ids = tuple(ids)
        self._positions = np.array([[body.x, body.y] for body in bodies], dtype=np.float64)
        self._velocities = np.array([[body.vx, body.vy] for body in bodies], dtype=np.float64)
        self._masses = np.array([body.mass for body in bodies], dtype=np.float64)
        self._radii = np.array([body.radius for body in bodies], dtype=np.float64)
        self._step_index = 0
        self._time = 0.0

    def __repr__(self) -> str:
        return (
            f"World(universe_id={self._universe_id!r}, bodies={len(self._ids)}, "
            f"time={self._time}, step={self._step_index})"
        )

    @property
    def universe_id(self) -> str:
        return self._universe_id

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def time(self) -> float:
        return self._time

    def fork(self, bodies: Sequence[BodyInit], dt: float) -> World:
        """New body state and timestep. Hidden laws are the same objects."""
        return World(
            bodies=bodies,
            force_laws=self._force_laws,
            dt=dt,
            universe_id=self._universe_id,
            seed=self._seed,
            difficulty=self._difficulty,
            difficulty_description=self._difficulty_description,
            integrator=type(self._integrator)(),
            collision=self._collision,
            dimension=self._dimension,
        )

    def _capture_guard(self) -> tuple[object, ...]:
        """Integrity key so the laboratory can prove it did not edit the laws."""
        return (
            self._force_fingerprint(),
            self._collision.enabled,
            self._collision.restitution,
            self._step_index,
            self._time,
            self._dt,
        )

    def _assert_same_guard(self, guard: tuple[object, ...]) -> None:
        if self._capture_guard() != guard:
            raise RuntimeError("Hidden laws or the prototype clock changed during an experiment.")

    def step(self) -> None:
        """Advance one timestep: continuous forces, then optional collisions."""

        def acceleration_fn(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
            return self._accelerations(positions, velocities)

        new_positions, new_velocities = self._integrator.step(
            self._positions,
            self._velocities,
            acceleration_fn,
            self._dt,
        )
        if self._collision.enabled:
            if self._collision.restitution is None:
                raise SimulatorError("enabled collisions require restitution.")
            new_positions, new_velocities, _events = resolve_collisions(
                new_positions,
                new_velocities,
                self._masses,
                self._radii,
                self._collision.restitution,
            )
        self._positions = new_positions
        self._velocities = new_velocities
        self._step_index += 1
        self._time = self._step_index * self._dt

    def observe(self) -> Observation:
        """Measurements at the current time. Hidden laws are not included.

        Reported acceleration is the continuous-force acceleration m^{-1} F
        evaluated at the current positions and velocities. Collision impulses
        are visible only as velocity changes, not as an infinite acceleration.
        """
        accelerations = self._accelerations(self._positions, self._velocities)
        bodies = tuple(
            BodyObservation(
                id=self._ids[index],
                x=float(self._positions[index, 0]),
                y=float(self._positions[index, 1]),
                vx=float(self._velocities[index, 0]),
                vy=float(self._velocities[index, 1]),
                ax=float(accelerations[index, 0]),
                ay=float(accelerations[index, 1]),
                mass=float(self._masses[index]),
                radius=float(self._radii[index]),
            )
            for index in range(len(self._ids))
        )
        return Observation(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            universe_id=self._universe_id,
            time=float(self._time),
            step_index=self._step_index,
            bodies=bodies,
        )

    def internal_state(self) -> InternalState:
        """Full state, including hidden laws. For evaluation only."""
        observation = self.observe()
        return InternalState(
            universe_id=self._universe_id,
            seed=self._seed,
            difficulty=self._difficulty,
            difficulty_description=self._difficulty_description,
            integrator=self._integrator.name,
            dimension=self._dimension,
            dt=self._dt,
            time=observation.time,
            step_index=observation.step_index,
            collisions_enabled=self._collision.enabled,
            restitution=self._collision.restitution,
            forces=self._force_fingerprint(),
            bodies=observation.bodies,
        )

    def _accelerations(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        total_force = np.zeros_like(positions, dtype=np.float64)
        for law in self._force_laws:
            contribution = np.asarray(
                law.forces(positions, velocities, self._masses),
                dtype=np.float64,
            )
            if contribution.shape != positions.shape:
                raise SimulatorError(
                    f"{law.name} returned shape {contribution.shape}, expected {positions.shape}."
                )
            total_force += contribution
        return total_force / self._masses[:, None]

    def _force_fingerprint(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        return tuple(
            (law.name, tuple(sorted(law.hidden_parameters().items())))
            for law in self._force_laws
        )
