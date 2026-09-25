"""Research plots drawn from measured histories and rollout scores."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(histories: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    for name, rows in histories.items():
        if len(rows) < 2:
            continue
        epochs = [row["epoch"] for row in rows]
        axis.plot(epochs, [row["train_loss"] for row in rows], label=f"{name} train")
        axis.plot(epochs, [row["val_position_rmse"] for row in rows], linestyle="--", label=f"{name} val RMSE")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss / validation position RMSE")
    axis.legend(fontsize=8)
    axis.set_title("Training curves")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_horizon_curves(models: list[dict], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    for model in models:
        horizons = sorted(model["horizons"], key=lambda item: int(item))
        axis.plot(
            [int(item) for item in horizons],
            [model["horizons"][item]["position"]["rmse"] for item in horizons],
            marker="o",
            label=model["model"],
        )
    axis.set_xlabel("Rollout horizon (steps)")
    axis.set_ylabel("Position RMSE")
    axis.set_title(title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_benchmark_bars(models: list[dict], path: Path, horizons: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    names = [model["model"] for model in models]
    width = 0.8 / max(len(horizons), 1)
    positions = np.arange(len(names))
    for index, horizon in enumerate(horizons):
        values = [model["horizons"][horizon]["position"]["rmse"] for model in models]
        axis.bar(positions + index * width, values, width=width, label=f"{horizon}-step")
    axis.set_xticks(positions + width * (len(horizons) - 1) / 2)
    axis.set_xticklabels(names, rotation=15)
    axis.set_ylabel("Position RMSE")
    axis.set_title("Held-out rollout error")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_ood_bars(scenarios: list[dict], path: Path) -> None:
    """One-step position RMSE for every OOD scenario. The axis is logarithmic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not scenarios:
        return
    names = [scenario["name"] for scenario in scenarios]
    model_names = [model["model"] for model in scenarios[0]["models"]]
    figure, axis = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(model_names), 1)
    positions = np.arange(len(names))
    for index, model_name in enumerate(model_names):
        values = []
        for scenario in scenarios:
            match = next(model for model in scenario["models"] if model["model"] == model_name)
            horizon = sorted(match["horizons"], key=lambda item: int(item))[0]
            values.append(match["horizons"][horizon]["position"]["rmse"])
        axis.bar(positions + index * width, values, width=width, label=model_name)
    axis.set_yscale("log")
    axis.set_xticks(positions + width * (len(model_names) - 1) / 2)
    axis.set_xticklabels(names, rotation=15)
    axis.set_ylabel("1-step position RMSE")
    axis.set_title("OOD position RMSE")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_trajectory(truth: np.ndarray, predicted: np.ndarray, path: Path, title: str) -> None:
    """`truth` and `predicted` have shape (steps, bodies, 2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    for body in range(truth.shape[1]):
        axis.plot(truth[:, body, 0], truth[:, body, 1], label=f"body {body} actual")
        axis.plot(
            predicted[:, body, 0],
            predicted[:, body, 1],
            linestyle="--",
            label=f"body {body} predicted",
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
