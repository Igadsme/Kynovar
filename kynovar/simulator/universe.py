"""Procedural universes whose pairwise power law is hidden from observers.

Implemented difficulties:

- Level 1 samples the coefficient k and fixes the exponent.
- Level 2 samples both k and p in F = k m1 m2 / r^p.

Levels 3 through 8 are specified in the config and rejected until implemented.
The public universe id is a hash prefix. It is not the seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from kynovar.simulator.collisions import CollisionSettings
from kynovar.simulator.forces import PairwisePowerLaw
from kynovar.simulator.state import BodyInit, SimulatorError
from kynovar.simulator.world import World
from kynovar.utils.paths import find_repo_root

IMPLEMENTED_DIFFICULTIES = frozenset({1, 2})
_UINT64 = 2**64


class UnsupportedDifficulty(SimulatorError):
    """The requested universe difficulty has no generator yet."""


@dataclass(frozen=True)
class UniverseConfig:
    k_range: tuple[float, float]
    p_range: tuple[float, float]
    softening: float
    count_range: tuple[int, int]
    mass_range: tuple[float, float]
    radius_range: tuple[float, float]
    position_range: tuple[float, float]
    velocity_range: tuple[float, float]
    minimum_center_distance: float
    placement_attempts: int
    fixed_exponent: float
    default_dt: float
    descriptions: dict[int, str]


def public_universe_id(seed: int) -> str:
    digest = hashlib.sha256(f"kynovar-universe-v1:{seed}".encode("utf-8")).hexdigest()[:8]
    return f"K-{digest.upper()}"


def default_power_law_config_path() -> Path:
    return find_repo_root() / "configs" / "simulator" / "power_law.yaml"


def load_power_law_config(path: Path | None = None) -> UniverseConfig:
    config_path = path or default_power_law_config_path()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SimulatorError(f"{config_path} did not contain a mapping.")
    if int(document.get("dimension", 0)) != 2:
        raise SimulatorError("The power-law universe config must request dimension 2.")
    if document.get("integrator") != "semi_implicit_euler":
        raise SimulatorError("The power-law universe config must request semi_implicit_euler.")

    collisions = document.get("collisions") or {}
    if collisions.get("enabled"):
        raise SimulatorError(
            "The power-law universe generator does not randomize collisions yet."
        )

    pairwise = document["pairwise"]
    bodies = document["bodies"]
    levels = {int(level): payload for level, payload in document["difficulty"]["levels"].items()}
    descriptions: dict[int, str] = {}
    expected_samples = {1: ["k"], 2: ["k", "p"]}
    for raw_level, payload in levels.items():
        level = int(raw_level)
        claimed = bool(payload.get("implemented", False))
        if claimed != (level in IMPLEMENTED_DIFFICULTIES):
            raise SimulatorError(
                f"{config_path} marks difficulty {level} implemented={claimed}, "
                f"but the generator allowlist is {sorted(IMPLEMENTED_DIFFICULTIES)}."
            )
        descriptions[level] = str(payload.get("description", ""))
        if level in expected_samples and list(payload.get("sample", [])) != expected_samples[level]:
            raise SimulatorError(
                f"Difficulty {level} must sample {expected_samples[level]}."
            )

    fixed_exponent = float(levels[1]["fixed_exponent"])
    softening = float(pairwise["softening"])
    minimum_center_distance = float(bodies["minimum_center_distance"])
    placement_attempts = int(bodies["placement_attempts"])
    default_dt = float(document.get("default_dt", 0.01))
    if softening < 0.0 or not np.isfinite(softening):
        raise SimulatorError("softening must be finite and non-negative.")
    if minimum_center_distance <= 0.0 or not np.isfinite(minimum_center_distance):
        raise SimulatorError("minimum_center_distance must be positive.")
    if placement_attempts < 1:
        raise SimulatorError("placement_attempts must be at least 1.")
    if default_dt <= 0.0 or not np.isfinite(default_dt):
        raise SimulatorError("default_dt must be positive.")
    if not np.isfinite(fixed_exponent):
        raise SimulatorError("fixed_exponent must be finite.")
    return UniverseConfig(
        k_range=_float_range(pairwise["k_range"], "k_range"),
        p_range=_float_range(pairwise["p_range"], "p_range"),
        softening=float(pairwise["softening"]),
        count_range=_int_range(bodies["count_range"], "count_range"),
        mass_range=_positive_range(bodies["mass_range"], "mass_range"),
        radius_range=_non_negative_range(bodies["radius_range"], "radius_range"),
        position_range=_float_range(bodies["position_range"], "position_range"),
        velocity_range=_float_range(bodies["velocity_range"], "velocity_range"),
        minimum_center_distance=minimum_center_distance,
        placement_attempts=placement_attempts,
        fixed_exponent=fixed_exponent,
        default_dt=default_dt,
        descriptions=descriptions,
    )


def generate_universe(
    seed: int,
    difficulty: int = 2,
    config: UniverseConfig | None = None,
) -> World:
    """Build a reproducible world. The same seed and difficulty rebuild the same laws."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _UINT64:
        raise SimulatorError("seed must be an int in [0, 2**64).")
    if difficulty not in IMPLEMENTED_DIFFICULTIES:
        implemented = ", ".join(str(level) for level in sorted(IMPLEMENTED_DIFFICULTIES))
        raise UnsupportedDifficulty(
            f"Difficulty {difficulty} is not implemented. Implemented levels: {implemented}."
        )
    resolved = config or load_power_law_config()
    if resolved.default_dt <= 0.0:
        raise SimulatorError("default_dt must be positive.")
    rng = np.random.Generator(np.random.PCG64(seed))
    coefficient = float(rng.uniform(resolved.k_range[0], resolved.k_range[1]))
    if difficulty == 1:
        exponent = float(resolved.fixed_exponent)
    else:
        exponent = float(rng.uniform(resolved.p_range[0], resolved.p_range[1]))
    bodies = _sample_bodies(rng, resolved)
    return World(
        bodies=bodies,
        force_laws=(
            PairwisePowerLaw(
                k=coefficient,
                p=exponent,
                softening=resolved.softening,
            ),
        ),
        dt=resolved.default_dt,
        universe_id=public_universe_id(seed),
        seed=seed,
        difficulty=difficulty,
        difficulty_description=resolved.descriptions.get(difficulty, ""),
        collision=CollisionSettings(enabled=False),
    )


def _sample_bodies(rng: np.random.Generator, config: UniverseConfig) -> tuple[BodyInit, ...]:
    count = int(rng.integers(config.count_range[0], config.count_range[1] + 1))
    masses = rng.uniform(config.mass_range[0], config.mass_range[1], size=count)
    radii = rng.uniform(config.radius_range[0], config.radius_range[1], size=count)
    velocities = rng.uniform(config.velocity_range[0], config.velocity_range[1], size=(count, 2))
    positions = _place_positions(rng, count, config)
    return tuple(
        BodyInit(
            id=index,
            x=float(positions[index, 0]),
            y=float(positions[index, 1]),
            vx=float(velocities[index, 0]),
            vy=float(velocities[index, 1]),
            mass=float(masses[index]),
            radius=float(radii[index]),
        )
        for index in range(count)
    )


def _place_positions(
    rng: np.random.Generator,
    count: int,
    config: UniverseConfig,
) -> np.ndarray:
    placed: list[np.ndarray] = []
    minimum = config.minimum_center_distance
    for _index in range(count):
        accepted: np.ndarray | None = None
        for _attempt in range(config.placement_attempts):
            candidate = rng.uniform(config.position_range[0], config.position_range[1], size=2)
            if all(float(np.linalg.norm(candidate - other)) >= minimum for other in placed):
                accepted = candidate
                break
        if accepted is None:
            raise SimulatorError(
                f"Failed to place body {_index} after {config.placement_attempts} attempts."
            )
        placed.append(accepted)
    return np.vstack(placed)


def _float_range(values: list[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise SimulatorError(f"{name} must be a pair, got {values!r}.")
    low, high = float(values[0]), float(values[1])
    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        raise SimulatorError(f"{name} must satisfy low < high, got {values!r}.")
    return low, high


def _positive_range(values: list[float], name: str) -> tuple[float, float]:
    low, high = _float_range(values, name)
    if low <= 0.0:
        raise SimulatorError(f"{name} values must be positive, got {values!r}.")
    return low, high


def _non_negative_range(values: list[float], name: str) -> tuple[float, float]:
    low, high = _float_range(values, name)
    if low < 0.0:
        raise SimulatorError(f"{name} values must be non-negative, got {values!r}.")
    return low, high


def _int_range(values: list[int], name: str) -> tuple[int, int]:
    if len(values) != 2:
        raise SimulatorError(f"{name} must be a pair, got {values!r}.")
    low, high = int(values[0]), int(values[1])
    if low < 1 or low > high:
        raise SimulatorError(f"{name} must satisfy 1 <= low <= high, got {values!r}.")
    return low, high
