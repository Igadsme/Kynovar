"""Named sampling regimes with observable acceptance rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from kynovar.simulator.universe import UniverseConfig
from kynovar.utils.paths import find_repo_root

_TUPLE_KEYS = {"k_range", "p_range", "mass_range", "position_range", "velocity_range", "radius_range"}


@dataclass(frozen=True)
class Acceptance:
    """Bounds on an observed trajectory. Every value is measured, not hidden."""

    max_position: float
    max_speed: float
    max_acceleration: float
    min_separation: float
    max_attempts: int = 60

    def accepts(self, states: np.ndarray) -> bool:
        if not np.isfinite(states).all():
            return False
        position = states[:, :, 0:2]
        if np.linalg.norm(position, axis=-1).max() > self.max_position:
            return False
        if np.linalg.norm(states[:, :, 2:4], axis=-1).max() > self.max_speed:
            return False
        if np.linalg.norm(states[:, :, 4:6], axis=-1).max() > self.max_acceleration:
            return False
        bodies = states.shape[1]
        for i in range(bodies):
            for j in range(i + 1, bodies):
                if np.linalg.norm(position[:, j] - position[:, i], axis=-1).min() < self.min_separation:
                    return False
        return True

    def as_dict(self) -> dict[str, float | int]:
        return {
            "max_position": self.max_position,
            "max_speed": self.max_speed,
            "max_acceleration": self.max_acceleration,
            "min_separation": self.min_separation,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True)
class Regime:
    name: str
    description: str
    prior: dict[str, object]
    acceptance: Acceptance | None

    def config(self, base: UniverseConfig) -> UniverseConfig:
        overrides: dict[str, object] = {}
        for key, value in self.prior.items():
            if key in _TUPLE_KEYS:
                overrides[key] = (float(value[0]), float(value[1]))
            elif key == "count_range":
                overrides[key] = (int(value[0]), int(value[1]))
            elif key == "minimum_center_distance":
                overrides[key] = float(value)
            else:
                raise ValueError(f"Unknown regime prior key {key!r}.")
        return replace(base, **overrides)


def load_regimes(path: Path | None = None) -> dict[str, Regime]:
    config_path = path or find_repo_root() / "configs" / "simulator" / "regimes.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    regimes: dict[str, Regime] = {}
    for name, payload in document["regimes"].items():
        acceptance = payload.get("acceptance")
        regimes[name] = Regime(
            name=name,
            description=str(payload.get("description", "")),
            prior=dict(payload.get("prior") or {}),
            acceptance=None if acceptance is None else Acceptance(**acceptance),
        )
    return regimes
