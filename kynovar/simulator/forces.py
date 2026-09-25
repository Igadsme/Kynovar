"""Continuous force laws.

Each law maps a mechanical state to pairwise-consistent forces in the same
units as mass * acceleration. Positive pairwise coefficients attract.
Hidden parameter values are exposed only by `hidden_parameters`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from kynovar.simulator.state import SimulatorError


class ForceLaw(ABC):
    name: ClassVar[str]

    @abstractmethod
    def forces(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Return an (N, D) force array. Do not mutate the inputs."""

    @abstractmethod
    def hidden_parameters(self) -> dict[str, float]:
        """Parameters withheld from `World.observe`."""


def _pairwise_delta(
    positions: np.ndarray,
) -> list[tuple[int, int, np.ndarray, float]]:
    pairs: list[tuple[int, int, np.ndarray, float]] = []
    count = positions.shape[0]
    for i in range(count):
        for j in range(i + 1, count):
            delta = positions[j] - positions[i]
            distance = float(np.linalg.norm(delta))
            pairs.append((i, j, delta, distance))
    return pairs


class ConstantGravity(ForceLaw):
    """F_i = m_i * g. `g` is the gravitational acceleration vector."""

    name: ClassVar[str] = "constant_gravity"

    def __init__(self, gx: float, gy: float) -> None:
        self._g = np.array([float(gx), float(gy)], dtype=np.float64)
        if not np.isfinite(self._g).all():
            raise SimulatorError("gravity components must be finite.")

    def forces(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
        if positions.shape[1] != self._g.shape[0]:
            raise SimulatorError(
                f"constant gravity is {self._g.shape[0]}D but positions are {positions.shape[1]}D."
            )
        return masses[:, None] * self._g

    def hidden_parameters(self) -> dict[str, float]:
        return {"gx": float(self._g[0]), "gy": float(self._g[1])}


class PairwisePowerLaw(ForceLaw):
    """F_ij = k * m_i * m_j / r_eff^p, directed from i toward j when k > 0.

    r_eff = max(r, softening). Separations at or above `softening` use the
    written formula exactly. Coincident bodies (r = 0) contribute no force.
    """

    name: ClassVar[str] = "pairwise_power_law"

    def __init__(self, k: float, p: float, softening: float = 1e-8) -> None:
        self._k = float(k)
        self._p = float(p)
        self._softening = float(softening)
        if not np.isfinite(self._k) or not np.isfinite(self._p):
            raise SimulatorError("power-law parameters k and p must be finite.")
        if not np.isfinite(self._softening) or self._softening < 0.0:
            raise SimulatorError("softening must be finite and non-negative.")

    def forces(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
        forces = np.zeros_like(positions, dtype=np.float64)
        for i, j, delta, distance in _pairwise_delta(positions):
            if distance == 0.0:
                continue
            r_eff = max(distance, self._softening)
            magnitude = self._k * float(masses[i]) * float(masses[j]) / (r_eff**self._p)
            direction = delta / distance
            contribution = magnitude * direction
            forces[i] += contribution
            forces[j] -= contribution
        return forces

    def hidden_parameters(self) -> dict[str, float]:
        return {"k": self._k, "p": self._p, "softening": self._softening}


class LinearDrag(ForceLaw):
    """F = -coefficient * v."""

    name: ClassVar[str] = "linear_drag"

    def __init__(self, coefficient: float) -> None:
        self._coefficient = float(coefficient)
        if not np.isfinite(self._coefficient):
            raise SimulatorError("linear drag coefficient must be finite.")

    def forces(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
        return -self._coefficient * velocities

    def hidden_parameters(self) -> dict[str, float]:
        return {"coefficient": self._coefficient}


class QuadraticDrag(ForceLaw):
    """F = -coefficient * |v| * v."""

    name: ClassVar[str] = "quadratic_drag"

    def __init__(self, coefficient: float) -> None:
        self._coefficient = float(coefficient)
        if not np.isfinite(self._coefficient):
            raise SimulatorError("quadratic drag coefficient must be finite.")

    def forces(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocities, axis=1, keepdims=True)
        return -self._coefficient * speed * velocities

    def hidden_parameters(self) -> dict[str, float]:
        return {"coefficient": self._coefficient}


class SpringForce(ForceLaw):
    """Pairwise Hooke force. F_ij magnitude = stiffness * (r - rest_length).

    Stretched pairs attract and compressed pairs repel. Every unique pair
    shares the same stiffness and rest length.
    """

    name: ClassVar[str] = "spring"

    def __init__(self, stiffness: float, rest_length: float) -> None:
        self._stiffness = float(stiffness)
        self._rest_length = float(rest_length)
        if not np.isfinite(self._stiffness) or not np.isfinite(self._rest_length):
            raise SimulatorError("spring parameters must be finite.")
        if self._rest_length < 0.0:
            raise SimulatorError("spring rest length must be non-negative.")

    def forces(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
        forces = np.zeros_like(positions, dtype=np.float64)
        for i, j, delta, distance in _pairwise_delta(positions):
            if distance == 0.0:
                continue
            magnitude = self._stiffness * (distance - self._rest_length)
            direction = delta / distance
            contribution = magnitude * direction
            forces[i] += contribution
            forces[j] -= contribution
        return forces

    def hidden_parameters(self) -> dict[str, float]:
        return {"stiffness": self._stiffness, "rest_length": self._rest_length}


CONTINUOUS_FORCE_LAWS: dict[str, type[ForceLaw]] = {
    ConstantGravity.name: ConstantGravity,
    PairwisePowerLaw.name: PairwisePowerLaw,
    LinearDrag.name: LinearDrag,
    QuadraticDrag.name: QuadraticDrag,
    SpringForce.name: SpringForce,
}
