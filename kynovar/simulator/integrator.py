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


class RungeKutta4(Integrator):
    """Classical fourth-order Runge-Kutta. Used as a numerical reference."""

    name: ClassVar[str] = "rk4"

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

        def accel(x: np.ndarray, v: np.ndarray) -> np.ndarray:
            value = np.asarray(acceleration_fn(x, v), dtype=np.float64)
            if value.shape != positions.shape or not np.isfinite(value).all():
                raise SimulatorError("acceleration_fn returned an invalid value.")
            return value

        k1x, k1v = velocities, accel(positions, velocities)
        k2x = velocities + 0.5 * dt * k1v
        k2v = accel(positions + 0.5 * dt * k1x, k2x)
        k3x = velocities + 0.5 * dt * k2v
        k3v = accel(positions + 0.5 * dt * k2x, k3x)
        k4x = velocities + dt * k3v
        k4v = accel(positions + dt * k3x, k4x)
        new_positions = positions + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        new_velocities = velocities + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return new_positions, new_velocities
