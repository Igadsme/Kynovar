"""Sample experimenter-visible initial conditions from public ranges."""

from __future__ import annotations

import numpy as np

from kynovar.simulator.state import BodyInit, SimulatorError
from kynovar.simulator.universe import UniverseConfig


def sample_bodies(rng: np.random.Generator, config: UniverseConfig) -> tuple[BodyInit, ...]:
    """Draw masses, radii, velocities, and non-overlapping positions.

    The draw uses the published ranges. It does not read a realized force law.
    """
    count = int(rng.integers(config.count_range[0], config.count_range[1] + 1))
    masses = rng.uniform(config.mass_range[0], config.mass_range[1], size=count)
    radii = rng.uniform(config.radius_range[0], config.radius_range[1], size=count)
    velocities = rng.uniform(config.velocity_range[0], config.velocity_range[1], size=(count, 2))
    placed: list[np.ndarray] = []
    minimum = config.minimum_center_distance
    for index in range(count):
        accepted: np.ndarray | None = None
        for _attempt in range(config.placement_attempts):
            candidate = rng.uniform(config.position_range[0], config.position_range[1], size=2)
            if all(float(np.linalg.norm(candidate - other)) >= minimum for other in placed):
                accepted = candidate
                break
        if accepted is None:
            raise SimulatorError(
                f"Failed to place body {index} after {config.placement_attempts} attempts."
            )
        placed.append(accepted)
    positions = np.vstack(placed)
    return tuple(
        BodyInit(
            id=index,
            x=float(positions[index, 0]),
            y=float(positions[index, 1]),
            vx=float(velocities[index, 0]),
            vy=float(velocities[index, 1]),
            mass=float(masses[index]),
            radius=float(radii[index]),
        )
        for index in range(count)
    )
