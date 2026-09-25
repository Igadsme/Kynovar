"""Collision impulses against the analytic restitution formula."""

from __future__ import annotations

import numpy as np
import pytest

from kynovar.simulator.collisions import CollisionSettings, resolve_collisions
from kynovar.simulator.state import SimulatorError


def test_equal_mass_elastic_exchange() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = np.array([[1.0, 0.0], [-1.0, 0.0]])
    masses = np.array([1.0, 1.0])
    radii = np.array([0.6, 0.6])
    new_positions, new_velocities, events = resolve_collisions(
        positions, velocities, masses, radii, restitution=1.0
    )
    np.testing.assert_allclose(new_velocities, [[-1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(new_positions, [[-0.1, 0.0], [1.1, 0.0]])
    assert events[0].impulse == pytest.approx(2.0)
    assert events[0].overlap == pytest.approx(0.2)


def test_equal_mass_inelastic_stops() -> None:
    _, velocities, events = resolve_collisions(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [-1.0, 0.0]]),
        np.array([1.0, 1.0]),
        np.array([0.6, 0.6]),
        restitution=0.0,
    )
    np.testing.assert_allclose(velocities, 0.0, atol=1e-12)
    assert events[0].impulse == pytest.approx(1.0)


def test_partial_restitution() -> None:
    """e = 0.5, equal mass, approach speed 2 -> separation speed 1."""
    _, velocities, events = resolve_collisions(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [-1.0, 0.0]]),
        np.array([1.0, 1.0]),
        np.array([0.6, 0.6]),
        restitution=0.5,
    )
    np.testing.assert_allclose(velocities, [[-0.5, 0.0], [0.5, 0.0]])
    assert events[0].impulse == pytest.approx(1.5)
    separation = float(velocities[1, 0] - velocities[0, 0])
    assert separation == pytest.approx(1.0)


def test_unequal_mass_elastic_impulse() -> None:
    """m = (1, 3), v = (1, -1), e = 1 -> v' = (-2, 0), J = 3."""
    _, velocities, events = resolve_collisions(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [-1.0, 0.0]]),
        np.array([1.0, 3.0]),
        np.array([0.6, 0.6]),
        restitution=1.0,
    )
    np.testing.assert_allclose(velocities, [[-2.0, 0.0], [0.0, 0.0]])
    assert events[0].impulse == pytest.approx(3.0)
    momentum = 1.0 * velocities[0] + 3.0 * velocities[1]
    np.testing.assert_allclose(momentum, [1.0 * 1.0 + 3.0 * -1.0, 0.0])


def test_separating_overlap_has_no_impulse() -> None:
    positions, velocities, events = resolve_collisions(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[-1.0, 0.0], [1.0, 0.0]]),
        np.array([1.0, 1.0]),
        np.array([0.6, 0.6]),
        restitution=1.0,
    )
    np.testing.assert_allclose(velocities, [[-1.0, 0.0], [1.0, 0.0]])
    assert events[0].impulse == 0.0
    assert float(np.linalg.norm(positions[1] - positions[0])) == pytest.approx(1.2)


def test_resolution_is_deterministic() -> None:
    args = (
        np.array([[0.0, 0.0], [0.5, 0.1], [0.2, 0.8]]),
        np.array([[0.4, 0.0], [-0.2, -0.1], [0.0, -0.3]]),
        np.array([1.0, 2.0, 1.5]),
        np.array([0.4, 0.4, 0.4]),
    )
    first = resolve_collisions(*args, restitution=0.3)
    second = resolve_collisions(*args, restitution=0.3)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_restitution_bounds() -> None:
    with pytest.raises(SimulatorError):
        CollisionSettings(enabled=True, restitution=None)
    with pytest.raises(SimulatorError):
        CollisionSettings(enabled=True, restitution=1.2)
    with pytest.raises(SimulatorError):
        resolve_collisions(
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.array([1.0]),
            np.array([1.0]),
            restitution=-0.1,
        )
    assert CollisionSettings(enabled=False).restitution is None
