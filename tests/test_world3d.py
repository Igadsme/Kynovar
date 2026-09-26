"""3D physics extension: dynamics, 2D consistency, boundary, and 3D discovery."""

from __future__ import annotations

import numpy as np
import pytest

from kynovar.discovery.evidence import pairwise_evidence
from kynovar.discovery.lab import DesignRanges, ExperimentClient, ExperimentDesign, observable_trouble, random_two_body_3d
from kynovar.discovery.power_sum import power_sum_search
from kynovar.evaluation.law_recovery import recovery_record
from kynovar.laboratory import Experiment, Laboratory
from kynovar.discovery.expression import parse_expression
from kynovar.simulator.forces import PairwisePowerLaw
from kynovar.simulator.state import BodyInit, SimulatorError
from kynovar.simulator.world import World
from kynovar.simulator.world3d import BodyInit3D, Experiment3D, Laboratory3D, World3D, observation_array_3d


def _world3d(k=2.5, p=2.0):
    placeholder = (BodyInit3D(0, 0, 0, 0, 0, 0, 0, 1.0, 0.05),)
    return World3D(placeholder, (PairwisePowerLaw(k=k, p=p),), dt=0.01, universe_id="K-3D")


def _pair3d():
    return (
        BodyInit3D(0, -0.6, 0.2, -0.3, 0.0, 0.1, 0.05, 1.2, 0.05),
        BodyInit3D(1, 0.7, -0.1, 0.4, 0.0, -0.08, 0.02, 0.8, 0.05),
    )


def test_3d_momentum_is_conserved_and_acceleration_matches_law() -> None:
    frames = Laboratory3D(_world3d()).run(Experiment3D(_pair3d(), 1.0, 0.01))
    states = observation_array_3d(frames)
    momentum = (states[:, :, 3:6] * states[:, :, 9:10]).sum(axis=1)
    assert np.allclose(momentum, momentum[0], atol=1e-12)
    delta = states[0, 1, 0:3] - states[0, 0, 0:3]
    r = np.linalg.norm(delta)
    expected = 2.5 * 1.2 * 0.8 / r**2 / 1.2 * delta / r
    assert np.allclose(states[0, 0, 6:9], expected)
    # Semi-implicit Euler: v' = v + a dt, x' = x + v' dt.
    v_next = states[0, :, 3:6] + states[0, :, 6:9] * 0.01
    assert np.allclose(states[1, :, 3:6], v_next)
    assert np.allclose(states[1, :, 0:3], states[0, :, 0:3] + v_next * 0.01)


def test_3d_world_matches_2d_world_in_the_plane() -> None:
    bodies2d = (BodyInit(0, -0.6, 0.2, 0.0, 0.1, 1.2, 0.05), BodyInit(1, 0.7, -0.1, 0.0, -0.08, 0.8, 0.05))
    bodies3d = tuple(BodyInit3D(b.id, b.x, b.y, 0.0, b.vx, b.vy, 0.0, b.mass, b.radius) for b in bodies2d)
    world2d = World(bodies2d, (PairwisePowerLaw(2.5, 2.0),), dt=0.01, universe_id="K-2D")
    trajectory = Laboratory(world2d).run(Experiment(objects=bodies2d, duration=0.5, dt=0.01))
    xy2d = trajectory.positions()
    states = observation_array_3d(Laboratory3D(_world3d()).run(Experiment3D(bodies3d, 0.5, 0.01)))
    assert np.allclose(states[:, :, 0:2], xy2d, atol=1e-12)
    assert np.all(states[:, :, 2] == 0.0)


def test_3d_boundary_and_validation() -> None:
    lab = Laboratory3D(_world3d())
    with pytest.raises(TypeError):
        lab.run(Experiment(objects=(BodyInit(0, 0, 0, 0, 0, 1, 0.1),), duration=0.1, dt=0.01))
    with pytest.raises(SimulatorError):
        Experiment3D(_pair3d(), 0.105, 0.01)
    with pytest.raises(SimulatorError):
        BodyInit3D(0, 0, 0, 0, 0, 0, 0, -1.0, 0.1)
    observation = lab.run(Experiment3D(_pair3d(), 0.02, 0.01))[0]
    assert not hasattr(observation, "forces") and not hasattr(observation.bodies[0], "k")


def test_discovery_from_3d_observations() -> None:
    client = ExperimentClient(Laboratory3D(_world3d(3.0, 1.5)))
    rng = np.random.default_rng(0)
    runs = []
    for _ in range(8):
        states = client.run(random_two_body_3d(rng, DesignRanges(duration=0.5)))
        assert states.shape[-1] == 11
        if observable_trouble(states) is None:
            runs.append(states)
    evidence = pairwise_evidence(runs)
    assert np.abs(evidence.transverse).max() < 1e-9
    assert np.allclose(evidence.target, 3.0 * evidence.data["m1"] * evidence.data["m2"] / evidence.data["r"] ** 1.5, rtol=1e-9)
    half = len(evidence) // 2
    first = {k: v[:half] for k, v in evidence.data.items()}
    second = {k: v[half:] for k, v in evidence.data.items()}
    result = power_sum_search(first, evidence.target[:half], second, evidence.target[half:], evidence.variables)
    assert recovery_record(result["expression"], parse_expression("3*m1*m2*r**-1.5", ("m1", "m2", "r")))["recovered"]


def test_design_dimension_detection() -> None:
    assert ExperimentDesign.two_body(1, 1, 1.5).dimension == 2
    assert random_two_body_3d(np.random.default_rng(1), DesignRanges()).dimension == 3
