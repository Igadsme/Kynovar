"""The scientist's side of the laboratory.

An `ExperimentClient` wraps a `Laboratory` and exposes only what an
experimenter controls (masses, positions, velocities, duration, timestep) and
what an instrument measures. It keeps a budget ledger so planners can be
compared by experiments, observations, simulation steps, and wall-clock time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from kynovar.discovery.evidence import state_layout
from kynovar.laboratory import Experiment, Laboratory
from kynovar.simulator.state import BodyInit
from kynovar.simulator.world3d import BodyInit3D, Experiment3D, observation_array_3d

DEFAULT_RADIUS = 0.05


@dataclass(frozen=True)
class ExperimentDesign:
    """Observable controls for one experiment. Arrays are per body."""

    masses: tuple[float, ...]
    positions: tuple[tuple[float, float], ...]
    velocities: tuple[tuple[float, float], ...]
    duration: float = 1.0
    dt: float = 0.01
    label: str = ""

    @property
    def dimension(self) -> int:
        return len(self.positions[0])

    def bodies(self):
        if self.dimension == 3:
            return tuple(
                BodyInit3D(id=index, x=float(p[0]), y=float(p[1]), z=float(p[2]), vx=float(v[0]), vy=float(v[1]), vz=float(v[2]), mass=float(m), radius=DEFAULT_RADIUS)
                for index, (m, p, v) in enumerate(zip(self.masses, self.positions, self.velocities, strict=True))
            )
        return tuple(
            BodyInit(id=index, x=float(p[0]), y=float(p[1]), vx=float(v[0]), vy=float(v[1]), mass=float(m), radius=DEFAULT_RADIUS)
            for index, (m, p, v) in enumerate(zip(self.masses, self.positions, self.velocities, strict=True))
        )

    def vector(self) -> np.ndarray:
        """A flat numeric description used for duplicate detection."""
        return np.concatenate([np.asarray(self.masses), np.asarray(self.positions).ravel(), np.asarray(self.velocities).ravel(), [self.duration]])

    def to_dict(self) -> dict:
        return {
            "masses": list(self.masses),
            "positions": [list(p) for p in self.positions],
            "velocities": [list(v) for v in self.velocities],
            "duration": self.duration,
            "dt": self.dt,
            "label": self.label,
        }

    @staticmethod
    def two_body(m1: float, m2: float, separation: float, v1: tuple[float, float] = (0.0, 0.0), v2: tuple[float, float] = (0.0, 0.0), duration: float = 1.0, dt: float = 0.01, label: str = "") -> "ExperimentDesign":
        half = separation / 2.0
        return ExperimentDesign((m1, m2), ((-half, 0.0), (half, 0.0)), (tuple(v1), tuple(v2)), duration, dt, label)

    @staticmethod
    def single(mass: float, speed: float, angle: float = 0.0, duration: float = 1.0, dt: float = 0.01, label: str = "") -> "ExperimentDesign":
        velocity = (speed * float(np.cos(angle)), speed * float(np.sin(angle)))
        return ExperimentDesign((mass,), ((0.0, 0.0),), (velocity,), duration, dt, label)


def observation_array(trajectory) -> np.ndarray:
    """(frames, bodies, 8) float64: x, y, vx, vy, ax, ay, mass, radius."""
    return np.asarray(
        [[[b.x, b.y, b.vx, b.vy, b.ax, b.ay, b.mass, b.radius] for b in frame.bodies] for frame in trajectory.observations],
        dtype=np.float64,
    )


@dataclass
class Ledger:
    experiments: int = 0
    observations: int = 0
    steps: int = 0
    seconds: float = 0.0
    designs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"experiments": self.experiments, "observations": self.observations, "steps": self.steps, "seconds": self.seconds}


@dataclass(frozen=True)
class Instrument:
    """Measurement model applied to every observed frame.

    Noise is Gaussian with the given standard deviations. Acceleration noise
    has an absolute part and a part proportional to |a|. `drop_fraction`
    marks random frames as missing (NaN), as a real tracker would.
    """

    position_noise: float = 0.0
    velocity_noise: float = 0.0
    acceleration_noise: float = 0.0
    relative_acceleration_noise: float = 0.0
    drop_fraction: float = 0.0
    seed: int = 0

    @property
    def ideal(self) -> bool:
        return not any((self.position_noise, self.velocity_noise, self.acceleration_noise, self.relative_acceleration_noise, self.drop_fraction))

    def measure(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.ideal:
            return states
        measured = states.copy()
        layout = state_layout(states)
        shape = states[:, :, layout["pos"]].shape
        measured[:, :, layout["pos"]] += rng.normal(0.0, self.position_noise, shape) if self.position_noise else 0.0
        measured[:, :, layout["vel"]] += rng.normal(0.0, self.velocity_noise, shape) if self.velocity_noise else 0.0
        magnitude = np.linalg.norm(states[:, :, layout["acc"]], axis=-1, keepdims=True)
        sigma = self.acceleration_noise + self.relative_acceleration_noise * magnitude
        measured[:, :, layout["acc"]] += rng.normal(0.0, 1.0, shape) * sigma
        if self.drop_fraction:
            dropped = rng.random(states.shape[0]) < self.drop_fraction
            dropped[0] = False
            measured[dropped, :, 0 : layout["acc"].stop] = np.nan
        return measured

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class ExperimentClient:
    def __init__(self, laboratory: Laboratory, instrument: Instrument | None = None) -> None:
        self._laboratory = laboratory
        self.instrument = instrument or Instrument()
        self._noise = np.random.default_rng(self.instrument.seed)
        self.ledger = Ledger()
        self.last_states: np.ndarray | None = None

    @property
    def universe_id(self) -> str:
        return self._laboratory.universe_id

    def run(self, design: ExperimentDesign) -> np.ndarray:
        started = time.perf_counter()
        if design.dimension == 3:
            frames = self._laboratory.run(Experiment3D(objects=design.bodies(), duration=design.duration, dt=design.dt))
            raw = observation_array_3d(frames)
        else:
            raw = observation_array(self._laboratory.run(Experiment(objects=design.bodies(), duration=design.duration, dt=design.dt)))
        states = self.instrument.measure(raw, self._noise)
        self.last_states = states
        self.ledger.experiments += 1
        self.ledger.observations += int(states.shape[0] * states.shape[1])
        self.ledger.steps += int(states.shape[0] - 1)
        self.ledger.seconds += time.perf_counter() - started
        self.ledger.designs.append(design.to_dict())
        return states


def observable_trouble(states: np.ndarray, max_acceleration: float = 60.0, min_separation: float = 0.25, max_position: float = 12.0) -> str | None:
    """Why an observed trajectory is unusable, judged from observations only.

    Frames marked missing by the instrument (NaN in every body) are ignored.
    """
    layout = state_layout(states)
    kinematic = slice(0, layout["acc"].stop)
    present = np.isfinite(states[:, :, kinematic]).all(axis=(1, 2))
    if present.sum() < 3:
        return "too few valid frames"
    states = states[present]
    if not np.isfinite(states).all():
        return "non-finite observation"
    acceleration = np.linalg.norm(states[:, :, layout["acc"]], axis=-1)
    if acceleration.max() > max_acceleration:
        return "acceleration above instrument range"
    if np.abs(states[:, :, layout["pos"]]).max() > max_position:
        return "body left the observation region"
    bodies = states.shape[1]
    for i in range(bodies):
        for j in range(i + 1, bodies):
            if np.linalg.norm(states[:, i, layout["pos"]] - states[:, j, layout["pos"]], axis=-1).min() < min_separation:
                return "close approach below minimum separation"
    return None


@dataclass(frozen=True)
class DesignRanges:
    """Observable control ranges for randomly sampled experiments."""

    mass: tuple[float, float] = (0.5, 2.0)
    separation: tuple[float, float] = (0.8, 2.5)
    speed: tuple[float, float] = (0.0, 0.4)
    single_speed: tuple[float, float] = (0.5, 3.0)
    duration: float = 1.0
    dt: float = 0.01


def random_two_body(rng: np.random.Generator, ranges: DesignRanges, label: str = "") -> ExperimentDesign:
    speeds = rng.uniform(*ranges.speed, size=2)
    angles = rng.uniform(0.0, 2 * np.pi, size=2)
    return ExperimentDesign.two_body(
        float(rng.uniform(*ranges.mass)),
        float(rng.uniform(*ranges.mass)),
        float(rng.uniform(*ranges.separation)),
        (float(speeds[0] * np.cos(angles[0])), float(speeds[0] * np.sin(angles[0]))),
        (float(speeds[1] * np.cos(angles[1])), float(speeds[1] * np.sin(angles[1]))),
        ranges.duration,
        ranges.dt,
        label,
    )


def random_two_body_3d(rng: np.random.Generator, ranges: DesignRanges, label: str = "") -> ExperimentDesign:
    """Two bodies separated along a random 3D direction, with random 3D velocities."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    separation = float(rng.uniform(*ranges.separation))
    half = direction * separation / 2.0
    velocities = []
    for _ in range(2):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * float(rng.uniform(*ranges.speed))
        velocities.append(tuple(float(c) for c in v))
    return ExperimentDesign(
        (float(rng.uniform(*ranges.mass)), float(rng.uniform(*ranges.mass))),
        (tuple(float(c) for c in -half), tuple(float(c) for c in half)),
        tuple(velocities),
        ranges.duration,
        ranges.dt,
        label,
    )


def random_single(rng: np.random.Generator, ranges: DesignRanges, label: str = "") -> ExperimentDesign:
    return ExperimentDesign.single(
        float(rng.uniform(*ranges.mass)),
        float(rng.uniform(*ranges.single_speed)),
        float(rng.uniform(0.0, 2 * np.pi)),
        ranges.duration,
        ranges.dt,
        label,
    )


def collect(client: ExperimentClient, designs: list[ExperimentDesign], **trouble_kwargs) -> tuple[list[np.ndarray], list[dict]]:
    """Run designs and keep usable trajectories. Rejected runs still cost budget."""
    kept, rejected = [], []
    for design in designs:
        states = client.run(design)
        reason = observable_trouble(states, **trouble_kwargs)
        if reason is None:
            kept.append(states)
        else:
            rejected.append({"design": design.to_dict(), "reason": reason})
    return kept, rejected
