"""Model shapes, integration, metrics, device selection, and a tiny training run."""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest
import torch

from kynovar.data.dataset import TrajectoryStore, collate_windows
from kynovar.data.generate import default_settings, generate_dataset
from kynovar.data.normalize import Normalizer, fit_normalizer
from kynovar.data.splits import split_universe_ids
from kynovar.evaluation.metrics import RunningScore, rmse
from kynovar.evaluation.plots import plot_horizon_curves, plot_trajectory
from kynovar.evaluation.predict import evaluate_horizons, prediction_loss, rollout_future
from kynovar.evaluation.report import render_benchmark_markdown
from kynovar.laboratory import Experiment, Laboratory
from kynovar.models.factory import MODEL_NAMES, build_model, trainable_parameter_count
from kynovar.models.features import design_matrix, edge_raw
from kynovar.models.integrate import semi_implicit_euler
from kynovar.models.train import load_trained_model, train_model
from kynovar.simulator.universe import generate_universe
from kynovar.utils.device import select_device
from kynovar.utils.paths import find_repo_root

ROOT = find_repo_root()


def _identity_normalizer() -> Normalizer:
    return Normalizer(
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
        np.zeros(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )


def test_metric_closed_form() -> None:
    predicted = np.array([[0.0, 0.0]])
    truth = np.array([[3.0, 4.0]])
    assert rmse(predicted, truth) == pytest.approx(math.sqrt(12.5))
    score = RunningScore()
    score.update(predicted, truth, np.array([True]))
    summary = score.as_dict()
    assert summary["rmse"] == pytest.approx(math.sqrt(12.5))
    assert summary["mae"] == pytest.approx(3.5)
    assert summary["relative_rmse"] == pytest.approx(1.0)
    assert summary["components"] == 2


def test_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.backends.mps.is_available():
        assert select_device("auto").type == "mps"
        assert select_device("mps").type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    assert select_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError):
        select_device("cuda")
    with pytest.raises(RuntimeError):
        select_device("mps")


def test_edge_geometry_and_integration() -> None:
    position = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    velocity = torch.zeros(1, 2, 2)
    edges = edge_raw(position, velocity)
    assert edges.shape == (1, 2, 2, 5)
    assert torch.allclose(edges[0, 0, 1, :3], torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(edges[0, 1, 0, :2], torch.tensor([-3.0, -4.0]))
    new_position, new_velocity = semi_implicit_euler(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0, -2.0]]),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor(0.1),
    )
    assert torch.allclose(new_velocity, torch.tensor([[1.0, -2.0]]))
    assert torch.allclose(new_position, torch.tensor([[0.1, -0.2]]))
    stepped_position, stepped_velocity = semi_implicit_euler(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[4.0, 0.0]]),
        torch.tensor([[-2.0, 1.0]]),
        torch.tensor(0.1),
    )
    assert torch.allclose(stepped_velocity, torch.tensor([[3.8, 0.1]]))
    assert torch.allclose(stepped_position, torch.tensor([[0.38, 0.01]]))


def test_recorded_acceleration_matches_the_next_simulator_state() -> None:
    world = generate_universe(5, difficulty=2)
    laboratory = Laboratory(world)
    trajectory = laboratory.run(
        Experiment(objects=_bodies_from_world(world), duration=0.2, dt=0.05)
    )
    dt = torch.tensor(0.05)
    for current, nxt in zip(trajectory.observations, trajectory.observations[1:]):
        position = torch.tensor([[body.x, body.y] for body in current.bodies])
        velocity = torch.tensor([[body.vx, body.vy] for body in current.bodies])
        acceleration = torch.tensor([[body.ax, body.ay] for body in current.bodies])
        predicted_position, predicted_velocity = semi_implicit_euler(position, velocity, acceleration, dt)
        truth_position = torch.tensor([[body.x, body.y] for body in nxt.bodies])
        truth_velocity = torch.tensor([[body.vx, body.vy] for body in nxt.bodies])
        assert torch.allclose(predicted_position, truth_position, atol=1e-6)
        assert torch.allclose(predicted_velocity, truth_velocity, atol=1e-6)


def _bodies_from_world(world):
    from kynovar.laboratory.experiment import bodies_from_observation

    return bodies_from_observation(world.observe())


def test_baseline_and_gnn_shapes_for_variable_body_counts() -> None:
    normalizer = _identity_normalizer()
    states = torch.randn(2, 4, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 4] = False
    for name in MODEL_NAMES:
        model = build_model(name, normalizer, hidden_dim=8, gnn_layers=1, sequence_length=4)
        acceleration = model.acceleration(states[:, -model.history :], mask)
        assert acceleration.shape == (2, 5, 2)
        assert torch.count_nonzero(acceleration[1, 4]) == 0
    features = design_matrix(states[:, -1], mask, normalizer)
    assert features.shape == (2, 5, 12)
    narrow = torch.randn(1, 1, 2, 8)
    wide = torch.randn(1, 1, 7, 8)
    gnn = build_model("gnn", normalizer, hidden_dim=8, gnn_layers=2, sequence_length=1)
    assert gnn.acceleration(narrow, torch.ones(1, 2, dtype=torch.bool)).shape == (1, 2, 2)
    assert gnn.acceleration(wide, torch.ones(1, 7, dtype=torch.bool)).shape == (1, 7, 2)
    padded, padded_mask = collate_windows(
        [narrow[0].numpy(), wide[0].numpy()],
    )
    assert padded.shape == (2, 1, 7, 8)
    assert padded_mask[0, 2:].sum() == 0
    assert gnn.acceleration(padded, padded_mask).shape == (2, 7, 2)
    assert trainable_parameter_count(build_model("constant_velocity", normalizer, 8, 1, 1)) == 0


def test_constant_velocity_rollout_is_exact() -> None:
    model = build_model("constant_velocity", _identity_normalizer(), 8, 1, 1)
    history = torch.zeros(1, 1, 1, 8)
    history[0, 0, 0, 2] = 2.0
    future, acceleration = rollout_future(
        model,
        history,
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor(0.25),
        4,
    )
    assert future.shape == (1, 4, 1, 8)
    assert acceleration.abs().sum() == 0
    assert future[0, -1, 0, 0].item() == pytest.approx(2.0)
    assert future[0, -1, 0, 2].item() == pytest.approx(2.0)


def test_rollout_loss_backpropagates() -> None:
    normalizer = _identity_normalizer()
    model = build_model("gnn", normalizer, hidden_dim=8, gnn_layers=1, sequence_length=1)
    states = torch.randn(2, 4, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    loss = prediction_loss(
        model,
        states,
        mask,
        torch.tensor(0.1),
        {"acceleration": 1.0, "position": 1.0, "velocity": 1.0, "rollout": 1.0},
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_tiny_training_checkpoint_and_rollout() -> None:
    directory = ROOT / "datasets" / "generated" / "_pytest_train"
    checkpoint_dir = ROOT / "checkpoints" / "_pytest_train"
    plot_dir = ROOT / "results" / "plots" / "_pytest_train"
    shutil.rmtree(directory, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    shutil.rmtree(plot_dir, ignore_errors=True)
    settings = default_settings(
        name="_pytest_train",
        worlds=6,
        experiments_per_world=2,
        duration=0.4,
        dt=0.1,
        dataset_seed=23,
        difficulty=2,
    )
    try:
        generate_dataset(directory, settings)
        store = TrajectoryStore(directory)
        splits = split_universe_ids(store.universe_ids, 23)
        normalizer = fit_normalizer(store, splits["train"])
        for name in MODEL_NAMES:
            payload = train_model(
                name,
                store,
                splits["train"],
                splits["validation"],
                normalizer,
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                weight_decay=1e-4,
                seed=3,
                device_name="cpu",
                sequence_length=2,
                stride=1,
                rollout_steps=1,
                hidden_dim=8,
                gnn_layers=1,
                patience=1,
                loss_weights={"acceleration": 1.0, "position": 1.0, "velocity": 1.0, "rollout": 0.0},
                checkpoint_path=checkpoint_dir / f"{name}.pt",
                num_workers=0,
            )
            assert math.isfinite(float(payload["best_val_position_rmse"]))
            assert payload["trainable_parameters"] == trainable_parameter_count(
                build_model(name, normalizer, 8, 1, 2)
            )
            model, loaded = load_trained_model(checkpoint_dir / f"{name}.pt", normalizer, torch.device("cpu"))
            assert loaded["model_name"] == name
            assert int(loaded["trainable_parameters"]) == int(payload["trainable_parameters"])
            scores = evaluate_horizons(model, store, splits["test"], [1, 3], stride=1, device=torch.device("cpu"))
            assert set(scores) == {"1", "3"}
            for horizon in scores.values():
                assert horizon["position"]["rmse"] >= 0.0
                assert horizon["velocity"]["rmse"] >= 0.0
                assert horizon["acceleration"]["rmse"] >= 0.0
                assert math.isfinite(horizon["position"]["rmse"])
        rollout_payload = train_model(
            "gnn",
            store,
            splits["train"],
            splits["validation"],
            normalizer,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=3,
            device_name="cpu",
            sequence_length=2,
            stride=1,
            rollout_steps=2,
            hidden_dim=8,
            gnn_layers=1,
            patience=1,
            loss_weights={"acceleration": 1.0, "position": 1.0, "velocity": 1.0, "rollout": 1.0},
            checkpoint_path=checkpoint_dir / "gnn_rollout.pt",
        )
        assert math.isfinite(float(rollout_payload["train_seconds"]))
        store.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(plot_dir, ignore_errors=True)


def test_report_copies_measured_values() -> None:
    measured = 0.123456789
    payload = {
        "title": "measured",
        "device": "cpu",
        "dataset": "unit",
        "models": [
            {
                "model": "constant_velocity",
                "trainable_parameters": 0,
                "train_seconds": 0.5,
                "horizons": {"1": {"position": {"rmse": measured}}},
            }
        ],
    }
    text = render_benchmark_markdown(payload)
    assert "0.12345679" in text
    assert "999" not in text
    plot_dir = ROOT / "results" / "plots" / "_pytest_report"
    shutil.rmtree(plot_dir, ignore_errors=True)
    try:
        plot_horizon_curves(payload["models"], plot_dir / "horizon.png", "unit")
        plot_trajectory(
            np.zeros((4, 1, 2)),
            np.ones((4, 1, 2)),
            plot_dir / "trajectory.png",
            "unit",
        )
        assert (plot_dir / "horizon.png").stat().st_size > 0
        assert (plot_dir / "trajectory.png").stat().st_size > 0
    finally:
        shutil.rmtree(plot_dir, ignore_errors=True)
