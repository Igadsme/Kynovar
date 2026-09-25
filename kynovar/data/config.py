"""Experiment yaml used by the dataset, training, and benchmark commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunConfig:
    name: str
    worlds: int
    experiments_per_world: int
    duration: float
    dt: float
    difficulty: int
    dataset_seed: int
    split_fractions: tuple[float, float, float]
    split_assignment_id: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    train_seed: int
    device: str
    sequence_length: int
    stride: int
    rollout_steps: int
    hidden_dim: int
    gnn_layers: int
    patience: int
    horizons: tuple[int, ...]
    num_workers: int
    lambda_acceleration: float
    lambda_position: float
    lambda_velocity: float
    lambda_rollout: float

    @property
    def loss_weights(self) -> dict[str, float]:
        return {
            "acceleration": self.lambda_acceleration,
            "position": self.lambda_position,
            "velocity": self.lambda_velocity,
            "rollout": self.lambda_rollout,
        }


def load_run_config(path: Path) -> RunConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    dataset = document["dataset"]
    split = document["split"]
    train = document["train"]
    evaluation = document["eval"]
    fractions = (float(split["train"]), float(split["validation"]), float(split["test"]))
    return RunConfig(
        name=str(document["name"]),
        worlds=int(dataset["worlds"]),
        experiments_per_world=int(dataset["experiments_per_world"]),
        duration=float(dataset["duration"]),
        dt=float(dataset["dt"]),
        difficulty=int(dataset["difficulty"]),
        dataset_seed=int(dataset["seed"]),
        split_fractions=fractions,
        split_assignment_id=int(split["seed"]),
        epochs=int(train["epochs"]),
        batch_size=int(train["batch_size"]),
        learning_rate=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
        train_seed=int(train["seed"]),
        device=str(train.get("device", "auto")),
        sequence_length=int(train["sequence_length"]),
        stride=int(train["stride"]),
        rollout_steps=int(train["rollout_steps"]),
        hidden_dim=int(train["hidden_dim"]),
        gnn_layers=int(train["gnn_layers"]),
        patience=int(train.get("patience", 5)),
        horizons=tuple(int(item) for item in evaluation["horizons"]),
        num_workers=int(evaluation.get("num_workers", 0)),
        lambda_acceleration=float(train.get("lambda_acceleration", 1.0)),
        lambda_position=float(train.get("lambda_position", 1.0)),
        lambda_velocity=float(train.get("lambda_velocity", 1.0)),
        lambda_rollout=float(train.get("lambda_rollout", 1.0)),
    )


def apply_overrides(config: RunConfig, **overrides: object) -> RunConfig:
    data = config.__dict__.copy()
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return RunConfig(**data)
