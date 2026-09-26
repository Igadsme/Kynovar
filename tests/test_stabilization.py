"""Milestone 2 stabilization: regimes, observable rejection, RK4, interaction GNN."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch

from kynovar.data.generate import accept_trajectory, default_settings, generate_dataset
from kynovar.data.normalize import Normalizer
from kynovar.data.ood import regime_ood_cases
from kynovar.data.regimes import Acceptance, load_regimes
from kynovar.evaluation.regimes import integrate_positions, trajectory_statistics
from kynovar.models.factory import build_model
from kynovar.simulator.state import BodyInit
from kynovar.simulator.universe import load_power_law_config
from kynovar.utils.paths import find_repo_root

ROOT = find_repo_root()


def _normalizer() -> Normalizer:
    return Normalizer(
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
        np.zeros(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )


def _states(positions, accelerations=None):
    positions = np.asarray(positions, dtype=np.float64)
    frames, bodies = positions.shape[:2]
    states = np.zeros((frames, bodies, 8))
    states[:, :, 0:2] = positions
    if accelerations is not None:
        states[:, :, 4:6] = accelerations
    states[:, :, 6] = 1.0
    states[:, :, 7] = 0.05
    return states


def test_regimes_load_with_expected_names() -> None:
    regimes = load_regimes()
    assert {"stable", "challenging", "extreme", "controlled_two_body"} <= set(regimes)
    assert regimes["extreme"].acceptance is None
    stable = regimes["stable"].config(load_power_law_config())
    assert stable.p_range == (1.0, 2.5)
    assert regimes["controlled_two_body"].config(load_power_law_config()).count_range == (2, 2)


def test_acceptance_rejects_close_pass_and_escape() -> None:
    acceptance = Acceptance(max_position=6.0, max_speed=4.0, max_acceleration=40.0, min_separation=0.3)
    calm = _states([[[-1.0, 0.0], [1.0, 0.0]]] * 5)
    assert acceptance.accepts(calm)
    assert accept_trajectory(calm, acceptance)
    close = _states([[[-0.1, 0.0], [0.1, 0.0]]] * 5)
    assert not acceptance.accepts(close)
    escaped = _states([[[-1.0, 0.0], [9.0, 0.0]]] * 5)
    assert not acceptance.accepts(escaped)
    violent = _states([[[-1.0, 0.0], [1.0, 0.0]]] * 5, accelerations=100.0)
    assert not acceptance.accepts(violent)
    assert accept_trajectory(violent, None)


def test_trajectory_statistics_flags_near_singularity() -> None:
    stats = trajectory_statistics(_states([[[-1.0, 0.0], [1.0, 0.0]], [[-0.02, 0.0], [0.02, 0.0]]]))
    assert stats["near_singular_frames"] == 1
    assert stats["overlap_frames"] == 1
    assert stats["min_distance"] == pytest.approx(0.04)


def test_rk4_beats_euler_on_circular_orbit() -> None:
    k, p, r = 3.0, 2.0, 2.0
    speed = float(np.sqrt(k / r**p * r / 2.0))
    bodies = (
        BodyInit(id=0, x=-1.0, y=0.0, vx=0.0, vy=-speed, mass=1.0, radius=0.05),
        BodyInit(id=1, x=1.0, y=0.0, vx=0.0, vy=speed, mass=1.0, radius=0.05),
    )
    reference = integrate_positions(k, p, bodies, 0.0005, 2.0, "rk4")
    euler = integrate_positions(k, p, bodies, 0.01, 2.0, "euler")
    rk4 = integrate_positions(k, p, bodies, 0.01, 2.0, "rk4")
    euler_error = np.abs(euler[-1] - reference[-1]).max()
    rk4_error = np.abs(rk4[-1] - reference[-1]).max()
    assert rk4_error < 1e-5 < euler_error
    # A circular orbit keeps its radius.
    assert np.linalg.norm(rk4[-1, 1] - rk4[-1, 0]) == pytest.approx(r, rel=1e-4)


def test_regime_dataset_accepts_every_trajectory() -> None:
    target = ROOT / "datasets" / "generated" / "_pytest_regime"
    shutil.rmtree(target, ignore_errors=True)
    try:
        settings = default_settings("_pytest_regime", 2, 2, 1.0, 0.01, 7, regime="stable")
        generate_dataset(target, settings)
        manifest = json.loads((target / "manifest.json").read_text())
        run = json.loads((target / "operator" / "run.json").read_text())
        assert run["regime"] == "stable" and run["acceptance"]["min_separation"] == 0.3
        acceptance = settings.acceptance
        for entry in manifest["universes"]:
            archive = np.load(target / entry["file"])
            for number, experiment in enumerate(entry["experiments"]):
                assert experiment["attempts"] >= 1
                assert acceptance.accepts(archive[f"states_{number:04d}"].astype(np.float64))
        plain = default_settings("_pytest_regime", 2, 2, 1.0, 0.01, 7)
        assert plain.acceptance is None
    finally:
        shutil.rmtree(target, ignore_errors=True)


def test_regime_ood_cases_are_graded() -> None:
    config = load_regimes()["stable"].config(load_power_law_config())
    cases = regime_ood_cases(config, "stable", 2.0, 0.01)
    severities = {case.severity for case in cases}
    assert severities <= {"near", "moderate", "extreme"} and len(severities) == 3
    names = {case.name for case in cases}
    assert {"exponent_near", "exponent_moderate", "body_count", "long_horizon"} <= names


def _random_window(bodies: int, history: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(1, history, bodies, 8, generator=generator)
    states[..., 6] = states[..., 6].abs() + 0.5
    states[..., 7] = 0.05
    return states


@pytest.mark.parametrize("name", ["interaction_gnn_single", "interaction_gnn"])
def test_interaction_gnn_is_translation_invariant_and_permutation_equivariant(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name, _normalizer(), hidden_dim=16, gnn_layers=2, sequence_length=4).eval()
    states = _random_window(4, model.history)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        base = model.acceleration(states, mask)
        shifted = states.clone()
        shifted[..., 0:2] += torch.tensor([3.0, -2.0])
        assert torch.allclose(model.acceleration(shifted, mask), base, atol=1e-4)
        order = torch.tensor([2, 0, 3, 1])
        permuted = model.acceleration(states[:, :, order], mask)
        assert torch.allclose(permuted, base[:, order], atol=1e-4)
