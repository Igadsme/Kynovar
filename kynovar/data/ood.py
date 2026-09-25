"""Out-of-distribution dataset settings.

Sampling boxes are shifted relative to the training prior. Realized coefficients
are still chosen inside the simulator and are not written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from kynovar.simulator.universe import UniverseConfig


@dataclass(frozen=True)
class OODCase:
    name: str
    description: str
    config: UniverseConfig
    dataset_seed_offset: int
    duration: float
    horizons: tuple[int, ...]


def ood_cases(base: UniverseConfig, duration: float, dt: float) -> list[OODCase]:
    long_duration = max(duration * 2.0, duration + 100.0 * dt)
    return [
        OODCase(
            name="exponent",
            description="Pairwise exponent sampled from [4, 6), above the training prior [0.5, 4).",
            config=replace(base, p_range=(4.0, 6.0)),
            dataset_seed_offset=1000,
            duration=duration,
            horizons=(1, 10, 50),
        ),
        OODCase(
            name="body_count",
            description="Five or six bodies. Training worlds use the default count range of two to four.",
            config=replace(
                base,
                count_range=(5, 6),
                position_range=(-5.0, 5.0),
                minimum_center_distance=0.5,
                placement_attempts=500,
            ),
            dataset_seed_offset=2000,
            duration=duration,
            horizons=(1, 10, 50),
        ),
        OODCase(
            name="mass",
            description="Masses sampled from [6, 12), above the training prior [0.5, 5).",
            config=replace(base, mass_range=(6.0, 12.0)),
            dataset_seed_offset=3000,
            duration=duration,
            horizons=(1, 10, 50),
        ),
        OODCase(
            name="velocity",
            description="Each initial velocity component is sampled from [1.2, 2.2), outside the training prior [-0.5, 0.5).",
            config=replace(base, velocity_range=(1.2, 2.2)),
            dataset_seed_offset=4000,
            duration=duration,
            horizons=(1, 10, 50),
        ),
        OODCase(
            name="long_horizon",
            description="New universes from the training prior, integrated for twice as long.",
            config=base,
            dataset_seed_offset=5000,
            duration=long_duration,
            horizons=(1, 50, 100, 150),
        ),
    ]
