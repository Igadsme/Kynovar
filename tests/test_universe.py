"""Hidden power-law universes are reproducible and refuse unimplemented levels."""

from __future__ import annotations

from pathlib import Path

import pytest

from kynovar.simulator.state import SimulatorError
from kynovar.simulator.universe import (
    IMPLEMENTED_DIFFICULTIES,
    UnsupportedDifficulty,
    generate_universe,
    load_power_law_config,
    public_universe_id,
)
from kynovar.utils.paths import find_repo_root


def _parameters(seed: int, difficulty: int) -> dict[str, float]:
    forces = generate_universe(seed, difficulty=difficulty).internal_state().forces
    return dict(forces[0][1])


def test_seed_zero_level_two_parameters_stay_fixed() -> None:
    """The level-2 PCG64 stream for seed 0 is part of the reproducibility contract."""
    parameters = _parameters(0, difficulty=2)
    assert parameters["k"] == 6.551136029553816
    assert parameters["p"] == 1.4442534981735462
    assert parameters["softening"] == 1e-08


def test_shipped_config_matches_the_generator_allowlist() -> None:
    config = load_power_law_config()
    assert config.k_range == (0.5, 10.0)
    assert config.p_range == (0.5, 4.0)
    assert config.fixed_exponent == 2.0
    assert config.default_dt == 0.01
    assert IMPLEMENTED_DIFFICULTIES == frozenset({1, 2})
    assert config.descriptions[1]
    assert config.descriptions[2]


def test_level_one_fixes_the_exponent_and_samples_the_coefficient() -> None:
    for seed in range(20):
        parameters = _parameters(seed, difficulty=1)
        assert parameters["p"] == 2.0
        assert 0.5 <= parameters["k"] < 10.0


def test_level_two_samples_coefficient_and_exponent() -> None:
    exponents = []
    for seed in range(30):
        parameters = _parameters(seed, difficulty=2)
        assert 0.5 <= parameters["k"] < 10.0
        assert 0.5 <= parameters["p"] < 4.0
        exponents.append(parameters["p"])
    assert any(abs(exponent - 2.0) > 1e-6 for exponent in exponents)
    assert len(set(exponents)) > 1


def test_same_seed_rebuilds_the_same_world() -> None:
    first = generate_universe(1234, difficulty=2)
    second = generate_universe(1234, difficulty=2)
    assert first.internal_state().forces == second.internal_state().forces
    assert first.observe() == second.observe()
    assert first.universe_id == second.universe_id
    for _ in range(25):
        first.step()
        second.step()
    assert first.observe() == second.observe()


def test_public_id_is_not_the_seed() -> None:
    seed = 987654321
    universe_id = public_universe_id(seed)
    assert universe_id == public_universe_id(seed)
    assert universe_id != public_universe_id(seed + 1)
    assert str(seed) not in universe_id
    assert universe_id.startswith("K-")
    assert generate_universe(seed, difficulty=1).universe_id == universe_id


def test_unimplemented_difficulties_fail_clearly() -> None:
    for level in (0, 3, 4, 5, 6, 7, 8):
        with pytest.raises(UnsupportedDifficulty, match="not implemented"):
            generate_universe(1, difficulty=level)


def test_seed_must_be_a_uint64() -> None:
    with pytest.raises(SimulatorError):
        generate_universe(-1, difficulty=1)
    with pytest.raises(SimulatorError):
        generate_universe(2**64, difficulty=1)
    with pytest.raises(SimulatorError):
        generate_universe(True, difficulty=1)  # type: ignore[arg-type]


def test_config_cannot_claim_an_unimplemented_level(tmp_path: Path) -> None:
    source = (find_repo_root() / "configs" / "simulator" / "power_law.yaml").read_text(
        encoding="utf-8"
    )
    edited = source.replace("implemented: false", "implemented: true", 1)
    path = tmp_path / "power_law.yaml"
    path.write_text(edited, encoding="utf-8")
    with pytest.raises(SimulatorError, match="implemented"):
        load_power_law_config(path)


def test_config_cannot_enable_collisions_for_this_generator(tmp_path: Path) -> None:
    source = (find_repo_root() / "configs" / "simulator" / "power_law.yaml").read_text(
        encoding="utf-8"
    )
    edited = source.replace("enabled: false", "enabled: true", 1)
    path = tmp_path / "power_law.yaml"
    path.write_text(edited, encoding="utf-8")
    with pytest.raises(SimulatorError, match="collisions"):
        load_power_law_config(path)


def test_generated_bodies_respect_minimum_separation() -> None:
    config = load_power_law_config()
    for seed in range(15):
        observation = generate_universe(seed, difficulty=2).observe()
        bodies = observation.bodies
        assert 2 <= len(bodies) <= 4
        for i, left in enumerate(bodies):
            for right in bodies[i + 1 :]:
                distance = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
                assert distance >= config.minimum_center_distance
            assert left.mass > 0.0
            assert left.radius >= 0.0
