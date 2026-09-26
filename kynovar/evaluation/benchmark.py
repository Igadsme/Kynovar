"""Prepare a dataset and train or evaluate every dynamics model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kynovar.data.config import RunConfig
from kynovar.data.dataset import TrajectoryStore
from kynovar.data.generate import GenerationSettings, default_settings, generate_dataset
from kynovar.data.normalize import fit_normalizer, load_normalizer, save_normalizer
from kynovar.data.ood import ood_cases, regime_ood_cases
from kynovar.data.splits import assert_disjoint_splits, read_splits, split_universe_ids, write_splits
from kynovar.evaluation.plots import (
    plot_benchmark_bars,
    plot_horizon_curves,
    plot_loss_curves,
    plot_ood_bars,
    plot_trajectory,
)
from kynovar.evaluation.predict import evaluate_horizons, rollout_future
from kynovar.evaluation.report import render_benchmark_markdown, render_ood_markdown, write_benchmark_csv, write_json
from kynovar.models.train import load_trained_model, train_model
from kynovar.simulator.universe import load_power_law_config
from kynovar.utils.device import select_device
from kynovar.utils.paths import find_repo_root, validate_startup
from kynovar.utils.reproducibility import runtime_record


def prepare_dataset(config: RunConfig, root: Path | None = None) -> tuple[TrajectoryStore, dict[str, list[str]], object]:
    repo = root or find_repo_root()
    validate_startup()
    dataset_dir = repo / "datasets" / "generated" / config.name
    settings = default_settings(
        name=config.name,
        worlds=config.worlds,
        experiments_per_world=config.experiments_per_world,
        duration=config.duration,
        dt=config.dt,
        dataset_seed=config.dataset_seed,
        difficulty=config.difficulty,
        regime=config.regime,
    )
    generate_dataset(dataset_dir, settings)
    store = TrajectoryStore(dataset_dir)
    splits = split_universe_ids(store.universe_ids, config.split_assignment_id, config.split_fractions)
    assert_disjoint_splits(splits)
    split_path = repo / "datasets" / "splits" / config.name / "splits.json"
    write_splits(split_path, splits, config.split_assignment_id)
    normalizer = fit_normalizer(store, splits["train"])
    save_normalizer(repo / "datasets" / "processed" / config.name / "normalization.json", normalizer)
    return store, splits, normalizer


def load_prepared(config: RunConfig, root: Path | None = None):
    repo = root or find_repo_root()
    dataset_dir = repo / "datasets" / "generated" / config.name
    store = TrajectoryStore(dataset_dir)
    splits = read_splits(repo / "datasets" / "splits" / config.name / "splits.json")
    normalizer = load_normalizer(repo / "datasets" / "processed" / config.name / "normalization.json")
    return store, splits, normalizer


def run_benchmark(config: RunConfig, retrain: bool = False, include_rollout_model: bool = True) -> dict:
    repo = find_repo_root()
    store, splits, normalizer = prepare_dataset(config)
    device = select_device(config.device)
    jobs = [(name, name, {}) for name in config.models]
    rollout_name = f"{config.rollout_comparison_model}_rollout"
    if include_rollout_model:
        jobs.append((rollout_name, config.rollout_comparison_model, {"rollout_steps": 4, "lambda_rollout": 1.0}))
    checkpoint_dir = repo / "checkpoints" / config.name
    histories = {}
    measured = []
    for checkpoint_name, architecture, overrides in jobs:
        checkpoint = checkpoint_dir / f"{checkpoint_name}.pt"
        if retrain or not checkpoint.exists():
            weights = dict(config.loss_weights)
            rollout_steps = config.rollout_steps
            if "lambda_rollout" in overrides:
                weights["rollout"] = float(overrides["lambda_rollout"])
            if "rollout_steps" in overrides:
                rollout_steps = int(overrides["rollout_steps"])
            payload = train_model(
                architecture,
                store,
                splits["train"],
                splits["validation"],
                normalizer,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                seed=config.train_seed,
                device_name=config.device,
                sequence_length=config.sequence_length,
                stride=config.stride,
                rollout_steps=rollout_steps,
                hidden_dim=config.hidden_dim,
                gnn_layers=config.gnn_layers,
                patience=config.patience,
                loss_weights=weights,
                checkpoint_path=checkpoint,
                num_workers=config.num_workers,
            )
            histories[checkpoint_name] = payload["epoch_history"]
        model, payload = load_trained_model(checkpoint, normalizer, device)
        horizons = _feasible(store, splits["test"], int(model.history), config.horizons)
        scores = evaluate_horizons(model, store, splits["test"], horizons, stride=config.eval_stride, device=device)
        if checkpoint_name not in histories and "epoch_history" in payload:
            histories[checkpoint_name] = payload["epoch_history"]
        measured.append(
            {
                "model": checkpoint_name,
                "architecture": architecture,
                "trainable_parameters": int(payload["trainable_parameters"]),
                "train_seconds": float(payload["train_seconds"]),
                "best_epoch": int(payload["best_epoch"]),
                "best_val_position_rmse": float(payload["best_val_position_rmse"]),
                "epochs_ran": int(payload["epochs_ran"]),
                "horizons": scores,
            }
        )
        print(f"evaluated {checkpoint_name} horizons={list(scores)}", flush=True)
    primary = [row for row in measured if row["model"] != rollout_name]
    loss = (
        "Learned models minimize a Huber loss on residuals divided by the training-set "
        "median absolute deviation. Reported RMSE stays in physical units."
    )
    if config.relative_acceleration:
        loss = (
            "The acceleration term is a Huber loss on (a_hat - a) / (|a| + s), where s is the training "
            "acceleration scale; position and velocity terms use the median absolute deviation. "
            "Reported RMSE stays in physical units."
        )
    report = {
        "title": f"Kynovar dynamics benchmark ({config.name})",
        "dataset": config.name,
        "regime": config.regime,
        "device": str(device),
        "run": runtime_record(repo, seed=config.train_seed),
        "config": {key: (list(value) if isinstance(value, tuple) else value) for key, value in config.__dict__.items()},
        "universes": _split_counts(splits),
        "eval_stride": config.eval_stride,
        "models": primary,
        "rollout_training_comparison": [
            row for row in measured if row["model"] in {config.rollout_comparison_model, rollout_name}
        ],
        "rollout_training_note": (
            f"{config.rollout_comparison_model} is trained to predict the next step. "
            f"{rollout_name} uses the same architecture with rollout_steps=4 and lambda_rollout=1."
        ),
        "loss": loss,
    }
    # runtime_record includes seed when passed. The benchmark record is an operator
    # report, not a model input. Drop it from any path the dataset loader reads.
    out = repo / "results" / "benchmarks" / config.name
    write_json(out / "benchmark.json", report)
    write_benchmark_csv(out / "benchmark.csv", primary)
    (out / "benchmark.md").write_text(render_benchmark_markdown(report), encoding="utf-8")
    plot_dir = repo / "results" / "plots" / config.name
    plot_loss_curves(histories, plot_dir / "training_curves.png")
    plot_horizon_curves(primary, plot_dir / "rollout_error.png", "Test position RMSE vs horizon")
    shared = sorted(set.intersection(*[set(row["horizons"]) for row in primary]), key=lambda item: int(item))
    if shared:
        chosen = [item for item in shared if int(item) in {1, 10, 50}] or shared[:3]
        plot_benchmark_bars(primary, plot_dir / "benchmark_bars.png", chosen)
    graph_models = [name for name in config.models if "gnn" in name]
    if graph_models:
        _plot_example(store, splits["test"], checkpoint_dir / f"{graph_models[-1]}.pt", normalizer, device, plot_dir)
    store.close()
    return report


def run_ood(config: RunConfig) -> dict:
    repo = find_repo_root()
    base = load_power_law_config()
    device = select_device(config.device)
    normalizer = load_normalizer(repo / "datasets" / "processed" / config.name / "normalization.json")
    checkpoint_dir = repo / "checkpoints" / config.name
    if config.regime is None:
        cases = ood_cases(base, config.duration, config.dt)
    else:
        cases = regime_ood_cases(base, config.regime, config.duration, config.dt)
    scenarios = []
    for case in cases:
        dataset_dir = repo / "datasets" / "generated" / f"{config.name}_ood_{case.name}"
        settings = GenerationSettings(
            name=f"{config.name}_ood_{case.name}",
            worlds=config.ood_worlds,
            experiments_per_world=config.ood_experiments,
            duration=case.duration,
            dt=config.dt,
            dataset_seed=config.dataset_seed + case.dataset_seed_offset,
            difficulty=config.difficulty,
            config=case.config,
            acceptance=case.acceptance,
            regime=case.regime,
        )
        generate_dataset(dataset_dir, settings)
        store = TrajectoryStore(dataset_dir)
        model_rows = []
        for checkpoint_name in config.models:
            model, payload = load_trained_model(checkpoint_dir / f"{checkpoint_name}.pt", normalizer, device)
            horizons = _feasible(store, store.universe_ids, int(model.history), case.horizons)
            scores = evaluate_horizons(
                model, store, store.universe_ids, horizons, stride=config.eval_stride, device=device
            )
            model_rows.append(
                {
                    "model": checkpoint_name,
                    "trainable_parameters": int(payload["trainable_parameters"]),
                    "train_seconds": float(payload["train_seconds"]),
                    "horizons": scores,
                }
            )
        scenarios.append(
            {
                "name": case.name,
                "severity": case.severity,
                "description": case.description,
                "universes": len(store.universe_ids),
                "models": model_rows,
            }
        )
        store.close()
        print(f"ood scenario={case.name}", flush=True)
    payload = {
        "title": f"Kynovar OOD evaluation ({config.name})",
        "dataset": config.name,
        "device": str(device),
        "run": runtime_record(repo),
        "scenarios": scenarios,
    }
    out = repo / "results" / "ood" / config.name
    write_json(out / "ood.json", payload)
    (out / "ood.md").write_text(render_ood_markdown(payload), encoding="utf-8")
    plot_horizon_curves(
        scenarios[0]["models"],
        repo / "results" / "plots" / config.name / "ood_exponent.png",
        "OOD exponent position RMSE",
    )
    plot_ood_bars(scenarios, repo / "results" / "plots" / config.name / "ood_bars.png")
    return payload


def _feasible(store: TrajectoryStore, universe_ids: list[str], history: int, horizons: tuple[int, ...]) -> list[int]:
    frames = []
    wanted = set(universe_ids)
    for record in store.manifest["universes"]:
        if record["universe_id"] not in wanted:
            continue
        for experiment in record["experiments"]:
            frames.append(int(experiment["frames"]))
    if not frames:
        raise ValueError("The split has no trajectories.")
    shortest = min(frames)
    allowed = [int(horizon) for horizon in horizons if int(horizon) <= shortest - history and int(horizon) >= 1]
    if not allowed:
        raise ValueError(
            f"No requested horizon fits trajectories of length {shortest} with history {history}."
        )
    return allowed


def _split_counts(splits: dict[str, list[str]]) -> dict[str, int]:
    return {name: len(ids) for name, ids in splits.items()}


def _plot_example(store, test_ids, checkpoint_path: Path, normalizer, device, plot_dir: Path) -> None:
    universe_id = test_ids[0]
    states_np = store.load_states(universe_id, 0)
    model, _payload = load_trained_model(checkpoint_path, normalizer, device)
    horizon = min(100, states_np.shape[0] - model.history)
    if horizon < 1:
        return
    time = model.history - 1
    window = torch.from_numpy(states_np[time - model.history + 1 : time + 1]).unsqueeze(0).to(device)
    mask = torch.ones(1, states_np.shape[1], dtype=torch.bool, device=device)
    dt = torch.tensor(store.dt, dtype=torch.float32, device=device)
    future, _acc = rollout_future(model, window, mask, dt, horizon)
    predicted = future[0, :, :, 0:2].detach().cpu().numpy()
    truth = states_np[time + 1 : time + 1 + horizon, :, 0:2]
    plot_trajectory(truth, predicted, plot_dir / "trajectory_overlay.png", f"{checkpoint_path.stem} rollout {universe_id}")
    np.savez_compressed(
        plot_dir / "trajectory_overlay.npz",
        truth=truth,
        predicted=predicted,
        universe_id=np.asarray(universe_id),
    )
