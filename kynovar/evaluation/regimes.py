"""Evaluation-only diagnostics of simulator regimes.

This module reads hidden law parameters in order to explain which hidden laws
and initial conditions produce unstable trajectories. It is scoring tooling.
Discovery and training code must not import it.
"""

from __future__ import annotations

import numpy as np

from kynovar.data.generate import mix_seed, trajectory_arrays
from kynovar.data.initial_conditions import sample_bodies
from kynovar.evaluation.ground_truth import ground_truth_record
from kynovar.laboratory import Experiment, Laboratory
from kynovar.simulator.forces import PairwisePowerLaw
from kynovar.simulator.integrator import RungeKutta4, SemiImplicitEuler
from kynovar.simulator.state import BodyInit
from kynovar.simulator.universe import UniverseConfig, generate_universe
from kynovar.simulator.world import World

OPERATING_POSITION = 8.0
OPERATING_SPEED = 8.0
OPERATING_ACCELERATION = 80.0
NEAR_SINGULAR_DISTANCE = 0.1


def trajectory_statistics(states: np.ndarray) -> dict[str, float | int]:
    """Summaries of one observed trajectory with shape (frames, bodies, 8)."""
    position = states[:, :, 0:2].astype(np.float64)
    velocity = states[:, :, 2:4].astype(np.float64)
    acceleration = states[:, :, 4:6].astype(np.float64)
    mass = states[:, :, 6].astype(np.float64)
    radius = states[0, :, 7].astype(np.float64)
    finite = np.isfinite(states)
    nonfinite = int((~finite).sum())
    pos_norm = np.linalg.norm(position, axis=-1)
    speed = np.linalg.norm(velocity, axis=-1)
    acc_norm = np.linalg.norm(acceleration, axis=-1)
    force = acc_norm * mass
    frames, bodies = states.shape[0], states.shape[1]
    min_distance = np.full(frames, np.inf)
    overlaps = 0
    for i in range(bodies):
        for j in range(i + 1, bodies):
            distance = np.linalg.norm(position[:, j] - position[:, i], axis=-1)
            min_distance = np.minimum(min_distance, distance)
            overlaps += int((distance < radius[i] + radius[j]).sum())
    outside = (
        (pos_norm > OPERATING_POSITION) | (speed > OPERATING_SPEED) | (acc_norm > OPERATING_ACCELERATION)
    )
    return {
        "max_position": float(np.nanmax(pos_norm)),
        "max_speed": float(np.nanmax(speed)),
        "max_acceleration": float(np.nanmax(acc_norm)),
        "max_force": float(np.nanmax(force)),
        "min_distance": float(np.nanmin(min_distance)) if bodies > 1 else float("inf"),
        "overlap_frames": overlaps,
        "near_singular_frames": int((min_distance < NEAR_SINGULAR_DISTANCE).sum()) if bodies > 1 else 0,
        "fraction_outside": float(outside.mean()),
        "nonfinite": nonfinite,
        "bodies": int(bodies),
    }


def diagnose_prior(
    config: UniverseConfig,
    universes: int,
    experiments_per_universe: int,
    duration: float,
    dt: float,
    dataset_seed: int,
) -> list[dict[str, object]]:
    """Sample universes from a prior and summarize every trajectory with its hidden law."""
    rows: list[dict[str, object]] = []
    for index in range(universes):
        world = generate_universe(mix_seed(dataset_seed, index, 1), difficulty=2, config=config)
        law = ground_truth_record(world)["forces"][0]["parameters"]
        laboratory = Laboratory(world)
        for experiment_index in range(experiments_per_universe):
            rng = np.random.Generator(np.random.PCG64(mix_seed(dataset_seed, index, experiment_index, 2)))
            bodies = sample_bodies(rng, config)
            trajectory = laboratory.run(Experiment(objects=bodies, duration=duration, dt=dt))
            states, _ids = trajectory_arrays(trajectory)
            stats = trajectory_statistics(states)
            masses = np.array([body.mass for body in bodies])
            positions = np.array([[body.x, body.y] for body in bodies])
            separations = [
                float(np.linalg.norm(positions[a] - positions[b]))
                for a in range(len(bodies))
                for b in range(a + 1, len(bodies))
            ]
            speeds = [float(np.hypot(body.vx, body.vy)) for body in bodies]
            rows.append(
                {
                    "universe": world.universe_id,
                    "k": float(law["k"]),
                    "p": float(law["p"]),
                    "max_mass": float(masses.max()),
                    "initial_min_separation": min(separations),
                    "initial_max_speed": max(speeds),
                    "initial_max_acceleration": float(np.linalg.norm(states[0, :, 4:6], axis=-1).max()),
                    **stats,
                }
            )
    return rows


def power_law_world(k: float, p: float, bodies: tuple[BodyInit, ...], dt: float, integrator: str) -> World:
    chosen = RungeKutta4() if integrator == "rk4" else SemiImplicitEuler()
    return World(
        bodies=bodies,
        force_laws=(PairwisePowerLaw(k=k, p=p),),
        dt=dt,
        universe_id="K-EVAL",
        integrator=chosen,
    )


def integrate_positions(k: float, p: float, bodies: tuple[BodyInit, ...], dt: float, duration: float, integrator: str) -> np.ndarray:
    world = power_law_world(k, p, bodies, dt, integrator)
    steps = int(round(duration / dt))
    frames = [np.array([[body.x, body.y] for body in world.observe().bodies])]
    for _ in range(steps):
        world.step()
        frames.append(np.array([[body.x, body.y] for body in world.observe().bodies]))
    return np.asarray(frames)


def timestep_study(
    k: float,
    p: float,
    bodies: tuple[BodyInit, ...],
    duration: float,
    timesteps: tuple[float, ...] = (0.02, 0.01, 0.005),
    reference_dt: float = 0.0005,
) -> dict[str, object]:
    """Final and path position error of semi-implicit Euler against an RK4 reference."""
    reference = integrate_positions(k, p, bodies, reference_dt, duration, "rk4")
    result: dict[str, object] = {"k": k, "p": p, "duration": duration, "reference_dt": reference_dt, "rows": []}
    for dt in timesteps:
        ratio = int(round(dt / reference_dt))
        for name in ("euler", "rk4"):
            path = integrate_positions(k, p, bodies, dt, duration, name)
            aligned = reference[::ratio][: path.shape[0]]
            error = np.linalg.norm(path - aligned, axis=-1)
            result["rows"].append(
                {
                    "dt": dt,
                    "integrator": "rk4" if name == "rk4" else "semi-implicit Euler",
                    "final_position_error": float(error[-1].max()),
                    "max_position_error": float(error.max()),
                    "mean_position_error": float(error.mean()),
                }
            )
    return result
