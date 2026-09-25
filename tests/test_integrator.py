"""Semi-implicit Euler against its closed form."""

from __future__ import annotations

import numpy as np
import pytest

from kynovar.simulator.integrator import SemiImplicitEuler
from kynovar.simulator.state import SimulatorError


def _constant_acceleration(acceleration: np.ndarray):
    def acceleration_fn(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        return np.repeat(acceleration[None, :], positions.shape[0], axis=0)

    return acceleration_fn


def test_step_does_not_mutate_inputs() -> None:
    positions = np.array([[0.0, 1.0]])
    velocities = np.array([[2.0, -3.0]])
    original_positions = positions.copy()
    original_velocities = velocities.copy()
    SemiImplicitEuler().step(
        positions,
        velocities,
        _constant_acceleration(np.array([0.0, -9.81])),
        0.1,
    )
    np.testing.assert_array_equal(positions, original_positions)
    np.testing.assert_array_equal(velocities, original_velocities)


def test_constant_acceleration_matches_discrete_closed_form() -> None:
    """v_n = v0 + n a dt; x_n = x0 + n dt v0 + a dt^2 n (n+1) / 2."""
    dt = 0.01
    steps = 100
    acceleration = np.array([0.25, -9.81])
    position = np.array([[1.5, -2.0]])
    velocity = np.array([[0.5, 2.0]])
    integrator = SemiImplicitEuler()
    for _ in range(steps):
        position, velocity = integrator.step(
            position,
            velocity,
            _constant_acceleration(acceleration),
            dt,
        )
    expected_velocity = np.array([0.5, 2.0]) + steps * acceleration * dt
    expected_position = (
        np.array([1.5, -2.0])
        + steps * dt * np.array([0.5, 2.0])
        + acceleration * dt**2 * steps * (steps + 1) / 2
    )
    np.testing.assert_allclose(velocity[0], expected_velocity, atol=1e-12)
    np.testing.assert_allclose(position[0], expected_position, atol=1e-12)


def test_free_particle_is_linear() -> None:
    dt = 0.05
    steps = 20
    position = np.array([[-1.0, 4.0]])
    velocity = np.array([[3.0, -2.0]])
    integrator = SemiImplicitEuler()
    for _ in range(steps):
        position, velocity = integrator.step(
            position,
            velocity,
            lambda positions, velocities: np.zeros_like(positions),
            dt,
        )
    np.testing.assert_allclose(velocity[0], [3.0, -2.0])
    np.testing.assert_allclose(position[0], [-1.0 + steps * dt * 3.0, 4.0 + steps * dt * -2.0])


def test_global_error_shrinks_as_timestep_shrinks() -> None:
    """Error versus x = x0 + v0 t + 0.5 a t^2 is 0.5 a dt t for this method."""
    acceleration = np.array([0.0, -9.81])
    t_final = 1.0
    errors = []
    for dt in (0.1, 0.01, 0.001):
        steps = int(round(t_final / dt))
        position = np.array([[0.0, 0.0]])
        velocity = np.array([[0.0, 0.0]])
        integrator = SemiImplicitEuler()
        for _ in range(steps):
            position, velocity = integrator.step(
                position,
                velocity,
                _constant_acceleration(acceleration),
                dt,
            )
        analytical = 0.5 * acceleration[1] * t_final**2
        errors.append(abs(float(position[0, 1]) - analytical))
        discrete_gap = 0.5 * acceleration[1] * dt * t_final
        assert float(position[0, 1]) == pytest.approx(analytical + discrete_gap, abs=1e-9)
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_rejects_non_positive_dt_and_bad_shapes() -> None:
    positions = np.zeros((1, 2))
    velocities = np.zeros((1, 2))
    with pytest.raises(SimulatorError):
        SemiImplicitEuler().step(
            positions,
            velocities,
            lambda p, v: np.zeros_like(p),
            0.0,
        )
    with pytest.raises(SimulatorError):
        SemiImplicitEuler().step(
            positions,
            velocities,
            lambda p, v: np.zeros((1, 3)),
            0.1,
        )
    with pytest.raises(SimulatorError):
        SemiImplicitEuler().step(
            positions,
            velocities,
            lambda p, v: np.array([[np.nan, 0.0]]),
            0.1,
        )
