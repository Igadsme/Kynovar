"""Numerical integrators.

`Integrator.step` receives an acceleration callback so a later multi-stage
method (RK4) can evaluate forces at intermediate states. Only semi-implicit
Euler is implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from kynovar.simulator.state import SimulatorError

AccelerationFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class Integrator(ABC):
    name: ClassVar[str]

    @abstractmethod
    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        acceleration_fn: AccelerationFn,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return updated positions and velocities. Inputs are not mutated."""


class SemiImplicitEuler(Integrator):
    """Symplectic Euler: update velocity, then position with the new velocity.

    For constant acceleration the discrete state after n steps is exact:

        v_n = v_0 + n * a * dt
        x_n = x_0 + n * dt * v_0 + a * dt^2 * n * (n + 1) / 2
    """

    name: ClassVar[str] = "semi_implicit_euler"

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        acceleration_fn: AccelerationFn,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if dt <= 0.0 or not np.isfinite(dt):
            raise SimulatorError(f"dt must be positive and finite, got {dt}.")
        if positions.shape != velocities.shape or positions.ndim != 2:
            raise SimulatorError(
                "positions and velocities must have the same shape (N, D), "
                f"got {positions.shape} and {velocities.shape}."
            )
        accelerations = np.asarray(acceleration_fn(positions, velocities), dtype=np.float64)
        if accelerations.shape != positions.shape:
            raise SimulatorError(
                "acceleration_fn must return shape "
                f"{positions.shape}, got {accelerations.shape}."
            )
        if not np.isfinite(accelerations).all():
            raise SimulatorError("acceleration_fn returned a non-finite value.")
        new_velocities = velocities + accelerations * dt
        new_positions = positions + new_velocities * dt
        return new_positions, new_velocities
