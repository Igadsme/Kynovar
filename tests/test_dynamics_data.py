"""Dataset generation, splits, leakage checks, and streaming."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from kynovar.data.dataset import TrajectoryStore, WindowDataset, collate_windows
from kynovar.data.generate import DatasetGenerationError, default_settings, generate_dataset
from kynovar.data.normalize import fit_normalizer, robust_center_scale
from kynovar.data.ood import ood_cases
from kynovar.data.schema import ACC, NODE_INDEX, STATE_FEATURES
from kynovar.data.splits import assert_disjoint_splits, split_counts, split_universe_ids
from kynovar.simulator.boundary import FORBIDDEN_PUBLIC_KEYS, InformationLeakError, assert_public_record
from kynovar.simulator.universe import load_power_law_config
from kynovar.utils.paths import find_repo_root

ROOT = find_repo_root()
SCAN_PATHS = (
    "kynovar/data",
    "kynovar/models",
    "kynovar/evaluation/predict.py",
    "kynovar/evaluation/metrics.py",
    "kynovar/evaluation/benchmark.py",
    "kynovar/evaluation/plots.py",
    "kynovar/evaluation/report.py",
    "scripts/generate_dataset.py",
    "scripts/train_dynamics.py",
    "scripts/run_benchmark.py",
    "scripts/run_ood.py",
)
FORBIDDEN_SOURCE = ("internal_state", "hidden_parameters", "ground_truth", "PairwisePowerLaw")


def test_split_counts_and_disjoint_universes() -> None:
    assert split_counts(20) == (14, 3, 3)
    assert split_counts(6) == (4, 1, 1)
    assert split_counts(3) == (1, 1, 1)
    ids = [f"K-{index:04d}" for index in range(20)]
    first = split_universe_ids(ids, 42)
    second = split_universe_ids(ids, 42)
    assert first == second
    assert_disjoint_splits(first)
    assert (len(first["train"]), len(first["validation"]), len(first["test"])) == (14, 3, 3)
    assigned = first["train"] + first["validation"] + first["test"]
    assert sorted(assigned) == sorted(ids)
    other = split_universe_ids(ids, 43)
    assert other != first


def test_prediction_sources_do_not_name_hidden_laws() -> None:
    for relative in SCAN_PATHS:
        path = ROOT / relative
        files = [path] if path.is_file() else list(path.rglob("*.py"))
        for source in files:
            if source.name.startswith("._"):
                continue
            text = source.read_text(encoding="utf-8")
            for token in FORBIDDEN_SOURCE:
                assert token not in text, f"{source.relative_to(ROOT)} contains {token}"


def test_robust_scale_ignores_a_single_outlier() -> None:
    values = np.array([[1.0], [2.0], [3.0], [4.0], [1.0e6]], dtype=np.float64)
    center, scale = robust_center_scale(values)
    assert center[0] == pytest.approx(3.0)
    assert scale[0] == pytest.approx(1.4826)
    assert scale[0] < 10.0


def test_node_features_exclude_acceleration() -> None:
    assert set(NODE_INDEX).isdisjoint(set(range(ACC.start, ACC.stop)))
    assert STATE_FEATURES[4:6] == ("ax", "ay")


def test_ood_ranges_leave_the_training_prior() -> None:
    base = load_power_law_config()
    cases = {case.name: case for case in ood_cases(base, duration=3.0, dt=0.02)}
    assert cases["exponent"].config.p_range[0] >= base.p_range[1]
    assert cases["body_count"].config.count_range[0] > base.count_range[1]
    assert cases["mass"].config.mass_range[0] >= base.mass_range[1]
    assert cases["velocity"].config.velocity_range[0] > base.velocity_range[1]
    assert cases["long_horizon"].duration > 3.0


def test_generated_dataset_is_deterministic_and_resumable(tmp_path_name: str = "_pytest_dataset") -> None:
    left = ROOT / "datasets" / "generated" / f"{tmp_path_name}_a"
    right = ROOT / "datasets" / "generated" / f"{tmp_path_name}_b"
    shutil.rmtree(left, ignore_errors=True)
    shutil.rmtree(right, ignore_errors=True)
    settings = default_settings(
        name=tmp_path_name,
        worlds=3,
        experiments_per_world=2,
        duration=0.4,
        dt=0.1,
        dataset_seed=11,
        difficulty=2,
    )
    try:
        generate_dataset(left, settings)
        generate_dataset(right, settings)
        _assert_same_dataset(left, right)
        target = left / "universes" / "0002.npz"
        target.unlink()
        completed = left / "completed.jsonl"
        lines = completed.read_text(encoding="utf-8").splitlines()
        completed.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        generate_dataset(left, settings)
        _assert_same_dataset(left, right)
        assert not list((left / "universes").glob("*.partial.npz"))
        changed = default_settings(
            name=tmp_path_name,
            worlds=4,
            experiments_per_world=2,
            duration=0.4,
            dt=0.1,
            dataset_seed=11,
            difficulty=2,
        )
        with pytest.raises(DatasetGenerationError):
            generate_dataset(left, changed)
    finally:
        shutil.rmtree(left, ignore_errors=True)
        shutil.rmtree(right, ignore_errors=True)


def test_dataset_has_no_hidden_parameters_and_streams() -> None:
    directory = ROOT / "datasets" / "generated" / "_pytest_stream"
    shutil.rmtree(directory, ignore_errors=True)
    settings = default_settings(
        name="_pytest_stream",
        worlds=6,
        experiments_per_world=2,
        duration=0.4,
        dt=0.1,
        dataset_seed=19,
        difficulty=2,
    )
    try:
        manifest = generate_dataset(directory, settings)
        assert_public_record(manifest)
        rendered = json.dumps(manifest)
        for key in FORBIDDEN_PUBLIC_KEYS:
            assert f'"{key}"' not in rendered
        store = TrajectoryStore(directory)
        assert store._open_file is None
        calls = {"count": 0}
        original = store.load_states

        def counting(universe_id: str, experiment_index: int):
            calls["count"] += 1
            return original(universe_id, experiment_index)

        store.load_states = counting  # type: ignore[method-assign]
        splits = split_universe_ids(store.universe_ids, 19)
        assert_disjoint_splits(splits)
        dataset = WindowDataset(store, splits["train"], sequence_length=2, horizon=1, stride=1)
        assert calls["count"] == 0
        window = dataset[0]
        assert calls["count"] == 1
        assert window.shape[0] == 3
        assert window.shape[2] == 8
        states, mask = collate_windows([window, dataset[1]])
        assert states.shape[0] == 2
        assert mask.dtype == torch.bool
        for universe_id, experiment, arrays in store.iter_split(store.universe_ids):
            assert universe_id.startswith("K-")
            assert arrays.shape[-1] == 8
            assert arrays.shape[0] == experiment["frames"]
            record = store.universe_record(universe_id)
            archive_path = directory / str(record["file"])
            with np.load(archive_path, allow_pickle=False) as archive:
                assert set(archive.files) <= {
                    "experiment_ids",
                    *(f"states_{index:04d}" for index in range(settings.experiments_per_world)),
                    *(f"ids_{index:04d}" for index in range(settings.experiments_per_world)),
                }
        leaked = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        leaked["k"] = 1.0
        with pytest.raises(InformationLeakError):
            assert_public_record(leaked)
        normalizer = fit_normalizer(store, splits["train"])
        again = fit_normalizer(store, splits["train"])
        assert np.allclose(normalizer.node_mean.numpy(), again.node_mean.numpy())
        train_only = normalizer.node_mean.detach().cpu().numpy().copy()
        # Changing a held-out universe must not change training statistics.
        test_id = splits["test"][0]
        test_states = original(test_id, 0)
        test_states = test_states + 50.0
        record = store.universe_record(test_id)
        path = directory / str(record["file"])
        store.close()
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        payload["states_0000"] = test_states
        np.savez_compressed(path, **payload)
        refreshed = TrajectoryStore(directory)
        shifted = fit_normalizer(refreshed, splits["train"])
        assert np.allclose(shifted.node_mean.detach().cpu().numpy(), train_only)
        refreshed.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _assert_same_dataset(left: Path, right: Path) -> None:
    left_manifest = json.loads((left / "manifest.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((right / "manifest.json").read_text(encoding="utf-8"))
    assert left_manifest == right_manifest
    for item in left_manifest["universes"]:
        with np.load(left / item["file"], allow_pickle=False) as a, np.load(right / item["file"], allow_pickle=False) as b:
            assert a.files == b.files
            for name in a.files:
                assert np.array_equal(a[name], b[name])
