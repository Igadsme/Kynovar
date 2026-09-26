"""Stream trajectory windows from per-universe npz files."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from kynovar.data.schema import STATE_FEATURES
from kynovar.simulator.boundary import assert_public_record


class TrajectoryStore:
    """Manifest plus on-demand npz reads. The full dataset is not kept in memory."""

    def __init__(self, root: Path, cache_items: int = 256) -> None:
        self.root = Path(root)
        self.cache_items = cache_items
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert_public_record(self.manifest)
        if list(self.manifest["feature_names"]) != list(STATE_FEATURES):
            raise ValueError("Dataset feature layout does not match the observable state schema.")
        self._open_path: Path | None = None
        self._open_file = None

    @property
    def dt(self) -> float:
        return float(self.manifest["dt"])

    @property
    def universe_ids(self) -> list[str]:
        return [str(item["universe_id"]) for item in self.manifest["universes"]]

    def universe_record(self, universe_id: str) -> dict:
        for item in self.manifest["universes"]:
            if item["universe_id"] == universe_id:
                return item
        raise KeyError(universe_id)

    def load_states(self, universe_id: str, experiment_index: int) -> np.ndarray:
        """One experiment. A bounded LRU keeps at most `cache_items` decoded arrays."""
        key = (universe_id, experiment_index)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached.copy()
        record = self.universe_record(universe_id)
        path = self.root / str(record["file"])
        archive = self._archive(path)
        states = np.array(archive[f"states_{experiment_index:04d}"], dtype=np.float32, copy=True)
        if self.cache_items > 0:
            self._cache[key] = states
            while len(self._cache) > self.cache_items:
                self._cache.popitem(last=False)
        return states.copy()

    def iter_split(self, universe_ids: list[str]):
        wanted = set(universe_ids)
        for record in self.manifest["universes"]:
            if record["universe_id"] not in wanted:
                continue
            for index, experiment in enumerate(record["experiments"]):
                yield record["universe_id"], experiment, self.load_states(record["universe_id"], index)

    def _archive(self, path: Path):
        if self._open_path != path:
            if self._open_file is not None:
                self._open_file.close()
            self._open_file = np.load(path, allow_pickle=False)
            self._open_path = path
        return self._open_file

    def close(self) -> None:
        self._cache.clear()
        if self._open_file is not None:
            self._open_file.close()
            self._open_file = None
            self._open_path = None


class WindowDataset(Dataset):
    """Windows of observable states. The index is built from the manifest alone."""

    def __init__(
        self,
        store: TrajectoryStore,
        universe_ids: list[str],
        sequence_length: int,
        horizon: int,
        stride: int = 1,
    ) -> None:
        if sequence_length < 1 or horizon < 1 or stride < 1:
            raise ValueError("sequence_length, horizon, and stride must be positive.")
        self.store = store
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.window = sequence_length + horizon
        self.index: list[tuple[str, int, int]] = []
        wanted = set(universe_ids)
        for record in store.manifest["universes"]:
            if record["universe_id"] not in wanted:
                continue
            for experiment_index, experiment in enumerate(record["experiments"]):
                frames = int(experiment["frames"])
                for start in range(0, frames - self.window + 1, stride):
                    self.index.append((str(record["universe_id"]), experiment_index, start))
        if not self.index:
            raise ValueError(
                "No windows fit the requested sequence length and horizon. "
                "Increase trajectory duration or reduce the horizon."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> np.ndarray:
        universe_id, experiment_index, start = self.index[index]
        states = self.store.load_states(universe_id, experiment_index)
        return states[start : start + self.window]


def collate_windows(batch: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable body counts. Mask is true for real bodies."""
    width = batch[0].shape[0]
    bodies = max(item.shape[1] for item in batch)
    states = torch.zeros(len(batch), width, bodies, batch[0].shape[2], dtype=torch.float32)
    mask = torch.zeros(len(batch), bodies, dtype=torch.bool)
    for index, item in enumerate(batch):
        count = item.shape[1]
        states[index, :, :count] = torch.from_numpy(np.array(item, dtype=np.float32, copy=True))
        mask[index, :count] = True
    return states, mask
