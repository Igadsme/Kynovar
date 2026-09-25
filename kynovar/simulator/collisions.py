"""Impulsive collision response with a coefficient of restitution.

Resolution order is deterministic: pairs are visited in increasing (i, j),
which is the body-array order. Overlap is removed along the contact normal,
weighted by inverse mass, and an impulse is applied when the bodies are
approaching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kynovar.simulator.state import SimulatorError


@dataclass(frozen=True)
class CollisionSettings:
    enabled: bool = False
    restitution: float | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.restitution is None:
            raise SimulatorError("enabled collisions require a restitution coefficient.")
        coefficient = float(self.restitution)
        if not np.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
            raise SimulatorError(
                f"restitution must be in [0, 1], got {self.restitution!r}."
            )
        object.__setattr__(self, "restitution", coefficient)


@dataclass(frozen=True)
class CollisionEvent:
    i: int
    j: int
    overlap: float
    impulse: float


def resolve_collisions(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    radii: np.ndarray,
    restitution: float,
) -> tuple[np.ndarray, np.ndarray, tuple[CollisionEvent, ...]]:
    """Return updated positions, velocities, and the events that fired.

    The impulse along the contact normal n (from i toward j) is

        J = -(1 + e) * ((v_j - v_i) · n) / (1/m_i + 1/m_j)

    applied only when the relative velocity along n is negative.
    """
    if not 0.0 <= restitution <= 1.0:
        raise SimulatorError(f"restitution must be in [0, 1], got {restitution}.")
    new_positions = np.array(positions, dtype=np.float64, copy=True)
    new_velocities = np.array(velocities, dtype=np.float64, copy=True)
    events: list[CollisionEvent] = []
    count = new_positions.shape[0]
    for i in range(count):
        for j in range(i + 1, count):
            delta = new_positions[j] - new_positions[i]
            distance = float(np.linalg.norm(delta))
            contact = float(radii[i] + radii[j])
            if distance >= contact:
                continue
            if distance == 0.0:
                normal = np.zeros(new_positions.shape[1], dtype=np.float64)
                normal[0] = 1.0
            else:
                normal = delta / distance
            overlap = contact - distance
            inverse_i = 1.0 / float(masses[i])
            inverse_j = 1.0 / float(masses[j])
            inverse_sum = inverse_i + inverse_j
            share_i = inverse_i / inverse_sum
            new_positions[i] -= share_i * overlap * normal
            new_positions[j] += (1.0 - share_i) * overlap * normal
            relative_velocity = float((new_velocities[j] - new_velocities[i]) @ normal)
            impulse = 0.0
            if relative_velocity < 0.0:
                impulse = -(1.0 + restitution) * relative_velocity / inverse_sum
                new_velocities[i] -= (impulse * inverse_i) * normal
                new_velocities[j] += (impulse * inverse_j) * normal
            events.append(CollisionEvent(i=i, j=j, overlap=overlap, impulse=impulse))
    return new_positions, new_velocities, tuple(events)
