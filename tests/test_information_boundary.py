"""Public records must not carry hidden laws, seeds, or parameter names."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kynovar.evaluation.ground_truth import ground_truth_record
from kynovar.laboratory import Experiment, Laboratory
from kynovar.laboratory.experiment import bodies_from_observation
from kynovar.simulator.boundary import (
    FORBIDDEN_PUBLIC_KEYS,
    InformationLeakError,
    assert_public_record,
    trajectory_to_dict,
)
from kynovar.simulator.forces import PairwisePowerLaw
from kynovar.simulator.state import BODY_OBSERVATION_FIELDS, BodyInit
from kynovar.simulator.universe import generate_universe
from kynovar.simulator.world import InternalState, World


def _secret_world() -> World:
    return World(
        [
            BodyInit(0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1),
            BodyInit(1, 2.0, 0.0, 0.0, 0.0, 1.0, 0.1),
        ],
        [PairwisePowerLaw(k=847261.13579, p=6.54321, softening=1e-8)],
        dt=0.01,
        universe_id="boundary-world",
        seed=99,
        difficulty=2,
        difficulty_description="UNIQUE_HIDDEN_DESCRIPTION_4f91c0",
    )


def test_observation_schema_is_kinematics_only() -> None:
    world = _secret_world()
    trajectory = Laboratory(world).run(
        Experiment(
            objects=bodies_from_observation(world.observe()),
            duration=0.02,
            dt=0.01,
            experiment_id="E-boundary",
        )
    )
    payload = trajectory_to_dict(trajectory)
    assert_public_record(payload)
    body = payload["observations"][0]["bodies"][0]
    assert set(body) == set(BODY_OBSERVATION_FIELDS)
    assert set(payload) == {
        "record_type",
        "schema_version",
        "universe_id",
        "experiment_id",
        "dt",
        "observations",
    }
    rendered = json.dumps(payload)
    assert "UNIQUE_HIDDEN_DESCRIPTION_4f91c0" not in rendered
    assert "pairwise_power_law" not in rendered
    assert "semi_implicit_euler" not in rendered
    assert '"seed"' not in rendered
    state = world.internal_state()
    coefficient = repr(dict(state.forces[0][1])["k"])
    assert coefficient in repr(state)
    assert coefficient not in repr(world)
    assert coefficient not in rendered


def test_public_serializer_rejects_internal_state() -> None:
    state = _secret_world().internal_state()
    assert isinstance(state, InternalState)
    with pytest.raises(InformationLeakError):
        trajectory_to_dict(state)


def test_ground_truth_is_not_a_public_record() -> None:
    record = ground_truth_record(_secret_world())
    assert record["record_type"] == "ground_truth"
    assert record["seed"] == 99
    assert record["forces"][0]["law"] == "pairwise_power_law"
    assert record["forces"][0]["parameters"]["k"] == pytest.approx(847261.13579)
    with pytest.raises(InformationLeakError):
        assert_public_record(record)


def test_forbidden_keys_cover_the_hidden_vocabulary() -> None:
    assert {"k", "p", "seed", "restitution", "softening", "ground_truth"} <= FORBIDDEN_PUBLIC_KEYS


def test_laboratory_source_does_not_return_hidden_state() -> None:
    source = Path(inspect.getfile(Laboratory)).read_text(encoding="utf-8")
    assert "internal_state" not in source
    assert "hidden_parameters" not in source
    assert "kynovar.evaluation" not in source
    assert "ground_truth" not in source


def test_generated_trajectory_omits_the_difficulty_description() -> None:
    world = generate_universe(5, difficulty=2)
    description = world.internal_state().difficulty_description
    assert description
    trajectory = Laboratory(world).run(
        Experiment(
            objects=bodies_from_observation(world.observe()),
            duration=0.05,
            dt=0.01,
        )
    )
    rendered = json.dumps(trajectory_to_dict(trajectory))
    assert description not in rendered
    assert_public_record(json.loads(rendered))
