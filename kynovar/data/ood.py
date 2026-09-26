"""Out-of-distribution dataset settings.

Sampling boxes are shifted relative to the training prior. Realized coefficients
are still chosen inside the simulator and are not written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from kynovar.data.regimes import Acceptance, load_regimes
from kynovar.simulator.universe import UniverseConfig


@dataclass(frozen=True)
class OODCase:
    name: str
    description: str
    config: UniverseConfig
    dataset_seed_offset: int
    duration: float
    horizons: tuple[int, ...]
    acceptance: Acceptance | None = None
    regime: str | None = None
    severity: str = ""


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


def regime_ood_cases(base: UniverseConfig, regime: str, duration: float, dt: float) -> list[OODCase]:
    """Shifts away from a named training regime, graded near, moderate, and extreme."""
    regimes = load_regimes()
    train = regimes[regime]
    config = train.config(base)
    accept = train.acceptance
    horizons = (1, 10, 50, 100)
    k_low, k_high = config.k_range
    p_low, p_high = config.p_range
    m_low, m_high = config.mass_range
    v_high = config.velocity_range[1]
    width = p_high - p_low
    return [
        OODCase(
            name="exponent_near",
            description=f"Exponent in [{p_high}, {p_high + 0.25 * width}), just above training [{p_low}, {p_high}).",
            config=replace(config, p_range=(p_high, p_high + 0.25 * width)),
            dataset_seed_offset=11000, duration=duration, horizons=horizons, acceptance=accept, regime=regime, severity="near",
        ),
        OODCase(
            name="exponent_moderate",
            description=f"Exponent in [{p_high + 0.25 * width}, {p_high + 0.75 * width}).",
            config=replace(config, p_range=(p_high + 0.25 * width, p_high + 0.75 * width)),
            dataset_seed_offset=12000, duration=duration, horizons=horizons, acceptance=accept, regime=regime, severity="moderate",
        ),
        OODCase(
            name="coefficient",
            description=f"Coefficient in [{k_high}, {2 * k_high}), above training [{k_low}, {k_high}).",
            config=replace(config, k_range=(k_high, 2.0 * k_high)),
            dataset_seed_offset=13000, duration=duration, horizons=horizons, acceptance=accept, regime=regime, severity="moderate",
        ),
        OODCase(
            name="mass",
            description=f"Masses in [{m_high}, {2 * m_high}), above training [{m_low}, {m_high}).",
            config=replace(config, mass_range=(m_high, 2.0 * m_high)),
            dataset_seed_offset=14000, duration=duration, horizons=horizons, acceptance=accept, regime=regime, severity="moderate",
        ),
        OODCase(
            name="velocity",
            description=f"Velocity components in [{v_high}, {v_high + 0.5}), above training |v| < {v_high}.",
            config=replace(config, velocity_range=(v_high, v_high + 0.5)),
            dataset_seed_offset=15000, duration=duration, horizons=horizons, acceptance=accept, regime=regime, severity="near",
        ),
        OODCase(
            name="body_count",
            description="Five or six bodies in a wider box. Training uses two to four.",
            config=replace(config, count_range=(5, 6), position_range=(-3.5, 3.5), placement_attempts=1000),
            dataset_seed_offset=16000, duration=duration, horizons=horizons, acceptance=accept, regime=regime, severity="moderate",
        ),
        OODCase(
            name="long_horizon",
            description="Training regime, integrated for twice as long.",
            config=config,
            dataset_seed_offset=17000, duration=2.0 * duration, horizons=(1, 50, 100, 200, 300),
            acceptance=accept, regime=regime, severity="near",
        ),
        OODCase(
            name="challenging_regime",
            description="Challenging regime: full default prior, close encounters allowed.",
            config=regimes["challenging"].config(base),
            dataset_seed_offset=18000, duration=duration, horizons=horizons,
            acceptance=regimes["challenging"].acceptance, regime="challenging", severity="moderate",
        ),
        OODCase(
            name="extreme_regime",
            description="Extreme regime: full default prior, no rejection, ejections included.",
            config=regimes["extreme"].config(base),
            dataset_seed_offset=19000, duration=duration, horizons=horizons,
            acceptance=None, regime="extreme", severity="extreme",
        ),
    ]
