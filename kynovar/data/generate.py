"""Generate observable trajectory datasets. Realized force parameters are not stored."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kynovar.data.initial_conditions import sample_bodies
from kynovar.data.regimes import Acceptance, load_regimes
from kynovar.data.schema import SCHEMA_VERSION, STATE_FEATURES
from kynovar.laboratory import Experiment, Laboratory
from kynovar.simulator.boundary import assert_public_record
from kynovar.simulator.universe import UniverseConfig, generate_universe, load_power_law_config

_UINT64 = 2**64


class DatasetGenerationError(RuntimeError):
    """The dataset directory does not match the requested generation settings."""


@dataclass(frozen=True)
class GenerationSettings:
    name: str
    worlds: int
    experiments_per_world: int
    duration: float
    dt: float
    dataset_seed: int
    difficulty: int
    config: UniverseConfig
    acceptance: Acceptance | None = None
    regime: str | None = None

    def signature(self) -> str:
        payload = {
            "name": self.name,
            "worlds": self.worlds,
            "experiments_per_world": self.experiments_per_world,
            "duration": self.duration,
            "dt": self.dt,
            "dataset_seed": self.dataset_seed,
            "difficulty": self.difficulty,
            "coefficient_range": list(self.config.k_range),
            "exponent_range": list(self.config.p_range),
            "count_range": list(self.config.count_range),
            "mass_range": list(self.config.mass_range),
            "radius_range": list(self.config.radius_range),
            "position_range": list(self.config.position_range),
            "velocity_range": list(self.config.velocity_range),
            "minimum_center_distance": self.config.minimum_center_distance,
        }
        if self.acceptance is not None:
            payload["acceptance"] = self.acceptance.as_dict()
            payload["regime"] = self.regime
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def operator_record(self) -> dict[str, object]:
        return {
            "signature": self.signature(),
            "name": self.name,
            "worlds": self.worlds,
            "experiments_per_world": self.experiments_per_world,
            "duration": self.duration,
            "dt": self.dt,
            "dataset_seed": self.dataset_seed,
            "difficulty": self.difficulty,
            "coefficient_range": list(self.config.k_range),
            "exponent_range": list(self.config.p_range),
            "count_range": list(self.config.count_range),
            "mass_range": list(self.config.mass_range),
            "radius_range": list(self.config.radius_range),
            "position_range": list(self.config.position_range),
            "velocity_range": list(self.config.velocity_range),
            "minimum_center_distance": self.config.minimum_center_distance,
            "acceptance": None if self.acceptance is None else self.acceptance.as_dict(),
            "regime": self.regime,
        }


class RejectedUniverse(RuntimeError):
    """No accepted experiment was found for one universe within the attempt budget."""


def mix_seed(*parts: int) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def trajectory_arrays(trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Stack an observed trajectory into `(frames, bodies, 8)` and body ids."""
    frames = []
    ids = [body.id for body in trajectory.observations[0].bodies]
    for observation in trajectory.observations:
        frames.append(
            [
                [
                    body.x,
                    body.y,
                    body.vx,
                    body.vy,
                    body.ax,
                    body.ay,
                    body.mass,
                    body.radius,
                ]
                for body in observation.bodies
            ]
        )
    return np.asarray(frames, dtype=np.float32), np.asarray(ids, dtype=np.int32)


def generate_dataset(output_dir: Path, settings: GenerationSettings) -> dict[str, object]:
    """Write one npz per universe. Completed universes are left in place."""
    output_dir.mkdir(parents=True, exist_ok=True)
    universe_dir = output_dir / "universes"
    universe_dir.mkdir(parents=True, exist_ok=True)
    operator_dir = output_dir / "operator"
    operator_dir.mkdir(parents=True, exist_ok=True)
    run_path = operator_dir / "run.json"
    signature = settings.signature()
    if run_path.exists():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise DatasetGenerationError(
                f"{output_dir} was built with different settings. Choose a new dataset name."
            )
    else:
        run_path.write_text(json.dumps(settings.operator_record(), indent=2) + "\n", encoding="utf-8")

    completed = _read_completed(output_dir / "completed.jsonl")
    universes: list[dict[str, object]] = []
    for index in range(settings.worlds):
        previous = completed.get(index)
        target = universe_dir / f"{index:04d}.npz"
        if previous is not None and target.exists():
            universes.append(previous)
            continue
        meta = _write_universe(target, index, settings)
        _append_completed(output_dir / "completed.jsonl", index, meta)
        universes.append(meta)
        print(
            f"generated universe {index + 1}/{settings.worlds} id={meta['universe_id']}",
            flush=True,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": settings.name,
        "dt": settings.dt,
        "duration": settings.duration,
        "feature_names": list(STATE_FEATURES),
        "universes": universes,
    }
    assert_public_record(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_universe(path: Path, index: int, settings: GenerationSettings) -> dict[str, object]:
    replacements = 0
    while True:
        # Replacement 0 reproduces the original seed stream for unfiltered datasets.
        parts = (settings.dataset_seed, index, 1) if replacements == 0 else (settings.dataset_seed, index, 1, replacements)
        world = generate_universe(mix_seed(*parts), difficulty=settings.difficulty, config=settings.config)
        try:
            payload, experiment_meta, experiment_ids = _run_experiments(world, index, replacements, settings)
            break
        except RejectedUniverse:
            replacements += 1
            if replacements > 50:
                raise DatasetGenerationError(
                    f"Universe slot {index}: 50 replacement universes had no accepted experiment."
                ) from None
    payload["experiment_ids"] = np.asarray(experiment_ids)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(partial, **payload)
    partial.replace(path)
    meta: dict[str, object] = {
        "universe_id": world.universe_id,
        "file": f"universes/{path.name}",
        "experiments": experiment_meta,
    }
    if settings.acceptance is not None:
        meta["replacements"] = replacements
    return meta


def accept_trajectory(states: np.ndarray, acceptance: Acceptance | None) -> bool:
    return acceptance is None or acceptance.accepts(states)


def _run_experiments(world, index: int, replacement: int, settings: GenerationSettings):
    laboratory = Laboratory(world)
    payload: dict[str, np.ndarray] = {}
    experiment_meta = []
    experiment_ids = []
    budget = 1 if settings.acceptance is None else settings.acceptance.max_attempts
    for experiment_index in range(settings.experiments_per_world):
        experiment_id = f"E-{experiment_index:04d}"
        accepted = None
        for attempt in range(budget):
            if replacement == 0 and attempt == 0:
                body_seed = mix_seed(settings.dataset_seed, index, experiment_index, 2)
            else:
                body_seed = mix_seed(settings.dataset_seed, index, experiment_index, 2, replacement, attempt)
            bodies = sample_bodies(np.random.Generator(np.random.PCG64(body_seed)), settings.config)
            trajectory = laboratory.run(
                Experiment(objects=bodies, duration=settings.duration, dt=settings.dt, experiment_id=experiment_id)
            )
            states, ids = trajectory_arrays(trajectory)
            if accept_trajectory(states, settings.acceptance):
                accepted = (states, ids, attempt + 1)
                break
        if accepted is None:
            raise RejectedUniverse(world.universe_id)
        states, ids, attempts = accepted
        payload[f"states_{experiment_index:04d}"] = states
        payload[f"ids_{experiment_index:04d}"] = ids
        experiment_ids.append(experiment_id)
        record = {"experiment_id": experiment_id, "frames": int(states.shape[0]), "bodies": int(states.shape[1])}
        if settings.acceptance is not None:
            record["attempts"] = attempts
        experiment_meta.append(record)
    return payload, experiment_meta, experiment_ids


def _read_completed(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    completed: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        completed[int(record["index"])] = record["meta"]
    return completed


def _append_completed(path: Path, index: int, meta: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"index": index, "meta": meta}) + "\n")


def default_settings(
    name: str,
    worlds: int,
    experiments_per_world: int,
    duration: float,
    dt: float,
    dataset_seed: int,
    difficulty: int = 2,
    config: UniverseConfig | None = None,
    regime: str | None = None,
) -> GenerationSettings:
    base = config or load_power_law_config()
    acceptance = None
    if regime is not None:
        chosen = load_regimes()[regime]
        base = chosen.config(base)
        acceptance = chosen.acceptance
    return GenerationSettings(
        name=name,
        worlds=worlds,
        experiments_per_world=experiments_per_world,
        duration=duration,
        dt=dt,
        dataset_seed=dataset_seed,
        difficulty=difficulty,
        config=base,
        acceptance=acceptance,
        regime=regime,
    )
