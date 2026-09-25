"""The laboratory returns measurements and cannot edit hidden laws."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kynovar.laboratory import Experiment, Laboratory, bodies_from_observation, experiment_from_mapping
from kynovar.laboratory.experiment import step_count
from kynovar.simulator.forces import PairwisePowerLaw
from kynovar.simulator.state import BodyInit, SimulatorError
from kynovar.simulator.universe import generate_universe
from kynovar.simulator.world import World


def _world() -> World:
    return World(
        [
            BodyInit(0, 0.0, 0.0, 0.1, 0.0, 2.0, 0.1),
            BodyInit(1, 2.0, 0.0, -0.1, 0.2, 3.0, 0.1),
        ],
        [PairwisePowerLaw(k=4.0, p=3.0)],
        dt=0.01,
        universe_id="lab-world",
        seed=7,
        difficulty=2,
        difficulty_description="hidden from the laboratory return value",
    )


def test_duration_must_be_an_integer_multiple_of_dt() -> None:
    assert step_count(5.0, 0.01) == 500
    assert step_count(0.0, 0.01) == 0
    with pytest.raises(SimulatorError):
        step_count(0.015, 0.01)
    with pytest.raises(SimulatorError):
        Experiment(objects=(BodyInit(0, 0, 0, 0, 0, 1, 0.1),), duration=0.015, dt=0.01)


def test_run_returns_measurements_and_leaves_the_prototype_unchanged() -> None:
    world = _world()
    before = world.observe()
    hidden = world.internal_state()
    trajectory = Laboratory(world).run(
        Experiment(
            objects=(
                BodyInit(0, -1.0, 0.2, 0.0, 0.0, 1.5, 0.05),
                BodyInit(1, 1.0, -0.2, 0.0, 0.0, 1.5, 0.05),
            ),
            duration=0.05,
            dt=0.01,
        )
    )
    assert world.observe() == before
    assert world.internal_state().forces == hidden.forces
    assert world.internal_state().time == 0.0
    assert trajectory.experiment_id == "E-0001"
    assert trajectory.universe_id == "lab-world"
    assert len(trajectory.observations) == 6
    assert trajectory.positions().shape == (6, 2, 2)
    assert trajectory.observations[0].time == 0.0
    assert trajectory.observations[-1].step_index == 5


def test_successive_experiments_do_not_carry_state() -> None:
    laboratory = Laboratory(_world())
    first = laboratory.run(
        Experiment(
            objects=(BodyInit(0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.1),),
            duration=0.02,
            dt=0.01,
            experiment_id="E-CUSTOM",
        )
    )
    second = laboratory.run(
        Experiment(
            objects=(BodyInit(0, 3.0, -1.0, 0.0, 2.0, 2.0, 0.2),),
            duration=0.0,
            dt=0.01,
        )
    )
    assert first.experiment_id == "E-CUSTOM"
    assert second.experiment_id == "E-0002"
    assert second.observations[0].bodies[0].x == pytest.approx(3.0)
    assert second.observations[0].bodies[0].vy == pytest.approx(2.0)
    assert len(second.observations) == 1


def test_identical_experiments_match() -> None:
    experiment = Experiment(
        objects=(
            BodyInit(0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.1),
            BodyInit(1, 1.5, 0.4, 0.2, -0.1, 1.0, 0.1),
        ),
        duration=0.2,
        dt=0.01,
    )
    left = Laboratory(_world()).run(experiment)
    right = Laboratory(_world()).run(experiment)
    assert left.observations == right.observations


def test_observation_is_sufficient_to_rebuild_an_experiment() -> None:
    world = generate_universe(11, difficulty=2)
    experiment = Experiment(
        objects=bodies_from_observation(world.observe()),
        duration=0.2,
        dt=0.01,
        experiment_id="from-observation",
    )
    trajectory = Laboratory(world).run(experiment)
    clone = generate_universe(11, difficulty=2)
    assert world.time == 0.0
    for frame in trajectory.observations[1:]:
        clone.step()
        assert clone.observe() == frame


def test_mapping_rejects_hidden_controls() -> None:
    payload = {
        "objects": [
            {"id": 0, "x": 0, "y": 0, "vx": 0, "vy": 0, "mass": 1, "radius": 0.1},
        ],
        "duration": 0.1,
        "dt": 0.1,
    }
    experiment = experiment_from_mapping(payload)
    assert experiment.objects[0].mass == 1.0
    for extra in ({"k": 4.0}, {"p": 3}, {"restitution": 1}, {"seed": 1}, {"forces": []}):
        with pytest.raises(SimulatorError, match="cannot set"):
            experiment_from_mapping({**payload, **extra})
    sneaky_object = dict(payload["objects"][0])
    sneaky_object["ax"] = 0.0
    with pytest.raises(SimulatorError, match="cannot set"):
        experiment_from_mapping({**payload, "objects": [sneaky_object]})


def test_subclass_cannot_smuggle_a_parameter() -> None:
    @dataclass(frozen=True)
    class Sneaky(Experiment):
        k: float = 4.0

    experiment = Sneaky(
        objects=(BodyInit(0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1),),
        duration=0.1,
        dt=0.1,
        k=4.0,
    )
    with pytest.raises(TypeError, match="subclass"):
        Laboratory(_world()).run(experiment)


def test_laboratory_repr_has_no_parameter_values() -> None:
    world = _world()
    text = repr(Laboratory(world))
    assert "lab-world" in text
    assert "4.0" not in text
    assert "hidden from the laboratory" not in text
