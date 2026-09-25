"""Force laws against closed-form magnitudes and Newton's third law."""

from __future__ import annotations

import numpy as np

from kynovar.simulator.forces import (
    CONTINUOUS_FORCE_LAWS,
    ConstantGravity,
    LinearDrag,
    PairwisePowerLaw,
    QuadraticDrag,
    SpringForce,
)


def test_registry_matches_shipped_simulator_config() -> None:
    import yaml

    from kynovar.utils.paths import find_repo_root

    document = yaml.safe_load(
        (find_repo_root() / "configs" / "simulator" / "default.yaml").read_text(encoding="utf-8")
    )
    assert set(document["continuous_force_laws"]) == set(CONTINUOUS_FORCE_LAWS)
    assert document["impulsive"] == ["collision"]
    assert document["dimension"] == 2
    assert document["integrator"] == "semi_implicit_euler"


def test_constant_gravity_is_mass_times_g() -> None:
    forces = ConstantGravity(0.0, -9.81).forces(
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        np.array([2.0]),
    )
    np.testing.assert_allclose(forces, [[0.0, 2.0 * -9.81]])


def test_target_power_law_force() -> None:
    """F = 4 m1 m2 / r^3 with m1=2, m2=3, r=2 is exactly 3."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0]])
    masses = np.array([2.0, 3.0])
    forces = PairwisePowerLaw(k=4.0, p=3.0).forces(positions, np.zeros((2, 2)), masses)
    np.testing.assert_allclose(forces, [[3.0, 0.0], [-3.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)


def test_power_law_softening_and_coincidence() -> None:
    masses = np.array([1.0, 1.0])
    softened = PairwisePowerLaw(k=2.0, p=2.0, softening=0.1).forces(
        np.array([[0.0, 0.0], [0.01, 0.0]]),
        np.zeros((2, 2)),
        masses,
    )
    # r_eff = 0.1, F = 2 / 0.1^2 = 200.
    np.testing.assert_allclose(softened[0, 0], 200.0, atol=1e-9)
    exact = PairwisePowerLaw(k=2.0, p=2.0, softening=0.1).forces(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.zeros((2, 2)),
        masses,
    )
    np.testing.assert_allclose(exact[0], [2.0, 0.0], atol=1e-12)
    coincident = PairwisePowerLaw(k=2.0, p=2.0).forces(
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        masses,
    )
    np.testing.assert_allclose(coincident, 0.0)


def test_pairwise_laws_obey_newtons_third_law() -> None:
    positions = np.array([[0.0, 0.0], [1.2, -0.4], [-0.7, 0.9], [0.2, 1.4]])
    velocities = np.array([[0.3, -0.2], [-0.1, 0.4], [0.5, 0.1], [-0.4, -0.3]])
    masses = np.array([0.5, 1.5, 2.0, 0.8])
    for law in (
        PairwisePowerLaw(k=1.7, p=2.4),
        SpringForce(stiffness=3.0, rest_length=0.5),
    ):
        forces = law.forces(positions, velocities, masses)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)


def test_linear_and_quadratic_drag() -> None:
    velocities = np.array([[4.0, 0.0]])
    masses = np.array([2.0])
    linear = LinearDrag(0.5).forces(np.zeros((1, 2)), velocities, masses)
    np.testing.assert_allclose(linear, [[-2.0, 0.0]])
    # |v| = 5 for (3, 4). F = -0.2 * 5 * (3, 4) = (-3, -4).
    quadratic = QuadraticDrag(0.2).forces(
        np.zeros((1, 2)),
        np.array([[3.0, 4.0]]),
        np.array([1.0]),
    )
    np.testing.assert_allclose(quadratic, [[-3.0, -4.0]])


def test_spring_rest_and_extension() -> None:
    rest = SpringForce(stiffness=5.0, rest_length=2.0).forces(
        np.array([[0.0, 0.0], [2.0, 0.0]]),
        np.zeros((2, 2)),
        np.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(rest, 0.0, atol=1e-12)
    # extension = 2, stiffness = 2, magnitude = 4.
    stretched = SpringForce(stiffness=2.0, rest_length=1.0).forces(
        np.array([[0.0, 0.0], [3.0, 0.0]]),
        np.zeros((2, 2)),
        np.array([2.0, 2.0]),
    )
    np.testing.assert_allclose(stretched, [[4.0, 0.0], [-4.0, 0.0]])


def test_hidden_parameters_are_explicit() -> None:
    assert ConstantGravity(1.0, -2.0).hidden_parameters() == {"gx": 1.0, "gy": -2.0}
    assert PairwisePowerLaw(4.0, 3.0, softening=1e-8).hidden_parameters()["k"] == 4.0
    assert LinearDrag(0.25).hidden_parameters() == {"coefficient": 0.25}
    assert QuadraticDrag(0.25).hidden_parameters() == {"coefficient": 0.25}
    assert SpringForce(1.5, 0.4).hidden_parameters() == {"stiffness": 1.5, "rest_length": 0.4}
