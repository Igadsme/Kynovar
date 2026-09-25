"""Seeded worlds and trajectories are bitwise stable across repeats."""

from __future__ import annotations

from kynovar.laboratory import Experiment, Laboratory, bodies_from_observation
from kynovar.simulator.forces import PairwisePowerLaw
from kynovar.simulator.state import BodyInit
from kynovar.simulator.universe import generate_universe
from kynovar.simulator.world import World


def test_two_processes_of_logic_match_for_the_target_law() -> None:
    def rollout() -> list[tuple[float, float, float, float]]:
        world = World(
            [
                BodyInit(0, 0.2, -0.4, 0.1, 0.05, 2.0, 0.1),
                BodyInit(1, 1.8, 0.3, -0.05, 0.02, 3.0, 0.1),
            ],
            [PairwisePowerLaw(k=4.0, p=3.0)],
            dt=0.01,
            universe_id="target",
            seed=4,
        )
        frames = []
        for _ in range(50):
            world.step()
            body = world.observe().bodies[0]
            frames.append((body.x, body.y, body.vx, body.vy))
        return frames

    assert rollout() == rollout()


def test_generated_universe_laboratory_rollout_matches() -> None:
    def rollout(seed: int) -> tuple[tuple[float, ...], ...]:
        world = generate_universe(seed, difficulty=2)
        trajectory = Laboratory(world).run(
            Experiment(
                objects=bodies_from_observation(world.observe()),
                duration=0.3,
                dt=0.01,
            )
        )
        frames = []
        for observation in trajectory.observations:
            for body in observation.bodies:
                frames.append((body.x, body.y, body.vx, body.vy, body.ax, body.ay))
        return tuple(frames)

    assert rollout(42) == rollout(42)
    assert rollout(42) != rollout(43)
