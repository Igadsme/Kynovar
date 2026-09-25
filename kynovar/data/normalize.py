"""Training-split normalization. Validation and test trajectories are not included."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kynovar.data.dataset import TrajectoryStore
from kynovar.data.schema import ACC, EDGE_FEATURES, NODE_INDEX


class Normalizer(nn.Module):
    """Affine map stored as buffers so checkpoints keep the training statistics.

    `node_mean` and `accel_mean` hold medians. `node_std` and `accel_std` hold
    the robust scales. Names stay stable so a checkpoint can round-trip.
    """

    def __init__(
        self,
        node_mean: np.ndarray,
        node_std: np.ndarray,
        edge_mean: np.ndarray,
        edge_std: np.ndarray,
        accel_mean: np.ndarray,
        accel_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("node_mean", _buffer(node_mean))
        self.register_buffer("node_std", _buffer(node_std))
        self.register_buffer("edge_mean", _buffer(edge_mean))
        self.register_buffer("edge_std", _buffer(edge_std))
        self.register_buffer("accel_mean", _buffer(accel_mean))
        self.register_buffer("accel_std", _buffer(accel_std))

    def encode_nodes(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.node_mean) / self.node_std

    def encode_edges(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.edge_mean) / self.edge_std

    def encode_accel(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.accel_mean) / self.accel_std

    def decode_accel(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.accel_std + self.accel_mean

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "node_mean": self.node_mean.detach().cpu().tolist(),
            "node_std": self.node_std.detach().cpu().tolist(),
            "edge_mean": self.edge_mean.detach().cpu().tolist(),
            "edge_std": self.edge_std.detach().cpu().tolist(),
            "accel_mean": self.accel_mean.detach().cpu().tolist(),
            "accel_std": self.accel_std.detach().cpu().tolist(),
            "node_features": ["x", "y", "vx", "vy", "mass", "radius"],
            "edge_features": list(EDGE_FEATURES),
            "center": "median",
            "scale": "1.4826 * median_absolute_deviation",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "Normalizer":
        return cls(
            np.asarray(payload["node_mean"], dtype=np.float32),
            np.asarray(payload["node_std"], dtype=np.float32),
            np.asarray(payload["edge_mean"], dtype=np.float32),
            np.asarray(payload["edge_std"], dtype=np.float32),
            np.asarray(payload["accel_mean"], dtype=np.float32),
            np.asarray(payload["accel_std"], dtype=np.float32),
        )


def fit_normalizer(store: TrajectoryStore, train_universe_ids: list[str]) -> Normalizer:
    """Estimate a center and scale from training universes only.

    The center is the median. The scale is 1.4826 times the median absolute
    deviation, which stays on the scale of the bulk of a heavy-tailed
    trajectory. Validation and test universes are not read.
    """
    node_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    accel_chunks: list[np.ndarray] = []
    for _universe_id, _experiment, states in store.iter_split(train_universe_ids):
        node_chunks.append(states[:, :, NODE_INDEX].reshape(-1, 6).astype(np.float64))
        accel_chunks.append(states[:, :, ACC].reshape(-1, 2).astype(np.float64))
        edges = _edge_samples(states)
        if edges.size:
            edge_chunks.append(edges)
    if not node_chunks or not accel_chunks or not edge_chunks:
        raise ValueError("Training split did not contain bodies and pairwise edges.")
    node_center, node_scale = robust_center_scale(np.concatenate(node_chunks, axis=0))
    edge_center, edge_scale = robust_center_scale(np.concatenate(edge_chunks, axis=0))
    accel_center, accel_scale = robust_center_scale(np.concatenate(accel_chunks, axis=0))
    return Normalizer(node_center, node_scale, edge_center, edge_scale, accel_center, accel_scale)


def save_normalizer(path: Path, normalizer: Normalizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalizer.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_normalizer(path: Path) -> Normalizer:
    return Normalizer.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _edge_samples(states: np.ndarray) -> np.ndarray:
    rows = []
    for frame in states:
        positions = frame[:, :2]
        velocities = frame[:, 2:4]
        count = frame.shape[0]
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                delta = positions[j] - positions[i]
                distance = float(np.linalg.norm(delta))
                relative = velocities[j] - velocities[i]
                rows.append([delta[0], delta[1], max(distance, 1e-8), relative[0], relative[1]])
    if not rows:
        return np.zeros((0, 5), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and scaled MAD. A zero MAD falls back to the 75th absolute deviation."""
    data = np.asarray(values, dtype=np.float64)
    center = np.median(data, axis=0)
    absolute = np.abs(data - center)
    scale = 1.4826 * np.median(absolute, axis=0)
    weak = scale < 1e-6
    if np.any(weak):
        fallback = np.percentile(absolute, 75, axis=0)
        scale = np.where(weak, np.maximum(fallback, 1e-6), scale)
    return center.astype(np.float32), scale.astype(np.float32)


def _buffer(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)
