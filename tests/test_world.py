"""World pipeline: forces, integration, collisions, and measurements."""

from __future__ import annotations

import numpy as np
import pytest

from kynovar.simulator.collisions import CollisionSettings
from kynovar.simulator.forces import ConstantGravity, LinearDrag, PairwisePowerLaw, QuadraticDrag
from kynovar.simulator.state import BodyInit, SimulatorError
from kynovar.simulator.world import World


def _body(**overrides: float) -> BodyInit:
    values: dict[str, float] = {
        "id": 0,
        "x": 0.0,
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "mass": 1.0,
        "radius": 0.1,
    }
    values.update(overrides)
    return BodyInit(
        id=int(values["id"]),
        x=values["x"],
        y=values["y"],
        vx=values["vx"],
        vy=values["vy"],
        mass=values["mass"],
        radius=values["radius"],
    )


def test_free_particle_through_the_world() -> None:
    world = World([_body(x=1.0, y=-2.0, vx=0.5, vy=-0.25)], [], dt=0.1, universe_id="free")
    for _ in range(15):
        world.step()
    observed = world.observe()
    body = observed.bodies[0]
    assert body.x == pytest.approx(1.0 + 15 * 0.1 * 0.5)
    assert body.y == pytest.approx(-2.0 + 15 * 0.1 * -0.25)
    assert body.ax == 0.0 and body.ay == 0.0
    assert observed.time == pytest.approx(1.5)
    assert observed.step_index == 15


def test_constant_gravity_matches_discrete_and_continuous_gap() -> None:
    gy = -9.81
    dt = 0.01
    steps = 100
    t_final = steps * dt
    vy0 = 2.0
    y0 = -2.0
    world = World(
        [_body(y=y0, vy=vy0, vx=0.5, mass=4.0)],
        [ConstantGravity(0.0, gy)],
        dt=dt,
        universe_id="gravity",
    )
    for _ in range(steps):
        world.step()
    body = world.observe().bodies[0]
    discrete_y = y0 + steps * dt * vy0 + gy * dt**2 * steps * (steps + 1) / 2
    continuous_y = y0 + vy0 * t_final + 0.5 * gy * t_final**2
    assert body.y == pytest.approx(discrete_y, abs=1e-12)
    assert body.vy == pytest.approx(vy0 + steps * gy * dt, abs=1e-12)
    assert body.x == pytest.approx(steps * dt * 0.5, abs=1e-12)
    assert body.ay == pytest.approx(gy)
    assert body.y - continuous_y == pytest.approx(0.5 * gy * dt * t_final, abs=1e-9)


def test_gravity_acceleration_does_not_depend_on_mass() -> None:
    def drop(mass: float) -> float:
        world = World(
            [_body(mass=mass)],
            [ConstantGravity(0.0, -9.81)],
            dt=0.05,
            universe_id="drop",
        )
        world.step()
        return world.observe().bodies[0].y

    assert drop(0.2) == pytest.approx(drop(20.0))


def test_target_power_law_one_step() -> None:
    """From rest, F = 3, so a = (1.5, -1) on the two masses. dt = 0.1."""
    world = World(
        [
            BodyInit(0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.1),
            BodyInit(1, 2.0, 0.0, 0.0, 0.0, 3.0, 0.1),
        ],
        [PairwisePowerLaw(k=4.0, p=3.0)],
        dt=0.1,
        universe_id="target-law",
    )
    initial = world.observe()
    assert initial.bodies[0].ax == pytest.approx(1.5)
    assert initial.bodies[1].ax == pytest.approx(-1.0)
    assert initial.bodies[0].ay == pytest.approx(0.0)
    world.step()
    moved = world.observe()
    assert moved.bodies[0].vx == pytest.approx(0.15)
    assert moved.bodies[0].x == pytest.approx(0.015)
    assert moved.bodies[1].vx == pytest.approx(-0.1)
    assert moved.bodies[1].x == pytest.approx(1.99)


def test_one_step_drag() -> None:
    linear = World(
        [_body(vx=4.0, mass=2.0)],
        [LinearDrag(0.5)],
        dt=0.1,
        universe_id="linear-drag",
    )
    linear.step()
    body = linear.observe().bodies[0]
    assert body.vx == pytest.approx(3.9)
    assert body.x == pytest.approx(0.39)

    quadratic = World(
        [_body(vx=3.0, vy=4.0, mass=1.0)],
        [QuadraticDrag(0.2)],
        dt=0.1,
        universe_id="quadratic-drag",
    )
    before = quadratic.observe()
    assert before.bodies[0].ax == pytest.approx(-3.0)
    assert before.bodies[0].ay == pytest.approx(-4.0)
    quadratic.step()
    after = quadratic.observe().bodies[0]
    assert after.vx == pytest.approx(3.0 + -3.0 * 0.1)
    assert after.vy == pytest.approx(4.0 + -4.0 * 0.1)


def test_superposed_gravity_and_drag() -> None:
    world = World(
        [_body(vx=4.0, mass=2.0)],
        [ConstantGravity(0.0, -10.0), LinearDrag(0.5)],
        dt=0.1,
        universe_id="superposition",
    )
    observed = world.observe().bodies[0]
    # F_drag = (-2, 0), F_gravity = (0, -20), a = (-1, -10).
    assert observed.ax == pytest.approx(-1.0)
    assert observed.ay == pytest.approx(-10.0)


def test_pairwise_momentum_is_conserved() -> None:
    rng = np.random.default_rng(0)
    bodies = [
        BodyInit(
            id=index,
            x=float(index * 1.5),
            y=float(rng.normal()),
            vx=float(rng.normal()),
            vy=float(rng.normal()),
            mass=float(1 + index),
            radius=0.05,
        )
        for index in range(3)
    ]
    world = World(
        bodies,
        [PairwisePowerLaw(k=4.0, p=3.0)],
        dt=0.005,
        universe_id="momentum",
    )

    def momentum(observation: object) -> tuple[float, float]:
        bodies_now = observation.bodies  # type: ignore[attr-defined]
        return (
            sum(body.mass * body.vx for body in bodies_now),
            sum(body.mass * body.vy for body in bodies_now),
        )

    start = momentum(world.observe())
    for _ in range(400):
        world.step()
    end = momentum(world.observe())
    assert end[0] == pytest.approx(start[0], abs=1e-8)
    assert end[1] == pytest.approx(start[1], abs=1e-8)


def test_collision_is_applied_after_the_drift() -> None:
    world = World(
        [
            BodyInit(0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.6),
            BodyInit(1, 1.0, 0.0, -1.0, 0.0, 1.0, 0.6),
        ],
        [],
        dt=0.01,
        universe_id="bounce",
        collision=CollisionSettings(enabled=True, restitution=1.0),
    )
    world.step()
    observed = world.observe()
    assert observed.bodies[0].x == pytest.approx(-0.10)
    assert observed.bodies[1].x == pytest.approx(1.10)
    assert observed.bodies[0].vx == pytest.approx(-1.0)
    assert observed.bodies[1].vx == pytest.approx(1.0)

    coasting = World(
        [
            BodyInit(0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.6),
            BodyInit(1, 1.0, 0.0, -1.0, 0.0, 1.0, 0.6),
        ],
        [],
        dt=0.01,
        universe_id="coast",
        collision=CollisionSettings(enabled=False),
    )
    coasting.step()
    assert coasting.observe().bodies[0].vx == pytest.approx(1.0)
    assert coasting.observe().bodies[0].x == pytest.approx(0.01)


def test_observation_does_not_advance_time_and_repeated_calls_match() -> None:
    world = World([_body(vx=1.0)], [ConstantGravity(0.0, -1.0)], dt=0.2, universe_id="still")
    first = world.observe()
    second = world.observe()
    assert first == second
    assert world.time == 0.0


def test_invalid_worlds_are_rejected() -> None:
    with pytest.raises(SimulatorError):
        BodyInit(0, 0.0, 0.0, 0.0, 0.0, mass=0.0, radius=1.0)
    with pytest.raises(SimulatorError):
        BodyInit(0, 0.0, 0.0, 0.0, 0.0, mass=1.0, radius=-0.1)
    with pytest.raises(SimulatorError):
        World([_body(id=1), _body(id=1)], [], dt=0.1, universe_id="dup")
    with pytest.raises(SimulatorError):
        World([_body()], [], dt=0.0, universe_id="dt")
    with pytest.raises(SimulatorError):
        World([_body()], [], dt=0.1, universe_id="flat", dimension=3)
    with pytest.raises(SimulatorError):
        World([], [], dt=0.1, universe_id="empty")
