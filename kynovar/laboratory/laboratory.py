"""Run experiments against a hidden world.

`run` accepts initial conditions, duration, and timestep. It returns measured
trajectories. It does not accept or return hidden laws.
"""

from __future__ import annotations

from kynovar.laboratory.experiment import Experiment, Trajectory, step_count
from kynovar.simulator.boundary import trajectory_to_dict
from kynovar.simulator.state import OBSERVATION_SCHEMA_VERSION
from kynovar.simulator.world import World


class Laboratory:
    def __init__(self, world: World) -> None:
        if not isinstance(world, World):
            raise TypeError(f"Laboratory requires a World, got {type(world).__name__}.")
        self._world = world
        self._experiment_count = 0

    def __repr__(self) -> str:
        return (
            f"Laboratory(universe_id={self._world.universe_id!r}, "
            f"experiments_run={self._experiment_count})"
        )

    @property
    def universe_id(self) -> str:
        return self._world.universe_id

    def run(self, experiment: Experiment) -> Trajectory:
        if type(experiment) is not Experiment:
            raise TypeError(
                "Laboratory.run accepts Experiment only. "
                "A subclass cannot add hidden-law controls."
            )
        guard = self._world._capture_guard()
        simulation = self._world.fork(experiment.objects, dt=experiment.dt)
        frames = [simulation.observe()]
        for _ in range(step_count(experiment.duration, experiment.dt)):
            simulation.step()
            frames.append(simulation.observe())
        self._world._assert_same_guard(guard)
        self._experiment_count += 1
        experiment_id = experiment.experiment_id or f"E-{self._experiment_count:04d}"
        trajectory = Trajectory(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            universe_id=simulation.universe_id,
            experiment_id=experiment_id,
            dt=experiment.dt,
            observations=tuple(frames),
        )
        # Validate the public encoding before handing it back.
        trajectory_to_dict(trajectory)
        return trajectory
