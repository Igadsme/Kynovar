"""Turn observed trajectories into regression evidence.

Only observable quantities are used: positions, velocities, accelerations,
masses. Hidden law parameters never enter these tables.

Two-body experiments give a central interaction force per frame. For body i
with partner j and unit vector u_ij from i toward j,

    F_ij = m_i (a_i . u_ij)

is the signed force along the line of centers, positive when attractive.
Single-body experiments give the force along the direction of motion,

    F_par = m (a . v / |v|).

Accelerations can come from the recorded measurements or, for noisy or
visual data, from finite differences of observed positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAIRWISE_VARIABLES = ("m1", "m2", "r")


def state_layout(states: np.ndarray) -> dict[str, slice | int]:
    """Column layout of an observation array: 8 columns in 2D, 11 in 3D."""
    width = states.shape[-1]
    if width == 8:
        return {"pos": slice(0, 2), "vel": slice(2, 4), "acc": slice(4, 6), "mass": 6, "dim": 2}
    if width == 11:
        return {"pos": slice(0, 3), "vel": slice(3, 6), "acc": slice(6, 9), "mass": 9, "dim": 3}
    raise ValueError(f"Unknown observation layout with {width} columns.")


def _perpendicular(vector: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """Signed perpendicular component in 2D; its magnitude in 3D."""
    if vector.shape[-1] == 2:
        return vector[:, 0] * unit[:, 1] - vector[:, 1] * unit[:, 0]
    along = (vector * unit).sum(axis=-1, keepdims=True)
    return np.linalg.norm(vector - along * unit, axis=-1)
PAIRWISE_WITH_VELOCITY = ("m1", "m2", "r", "vr")
SINGLE_VARIABLES = ("m", "s")


@dataclass
class Evidence:
    kind: str
    variables: tuple[str, ...]
    data: dict[str, np.ndarray]
    target: np.ndarray
    transverse: np.ndarray
    experiments: int
    frames: int
    groups: np.ndarray | None = None

    def subset(self, mask: np.ndarray) -> "Evidence":
        return Evidence(
            self.kind,
            self.variables,
            {key: value[mask] for key, value in self.data.items()},
            self.target[mask],
            self.transverse[mask],
            self.experiments,
            int(mask.sum()),
            None if self.groups is None else self.groups[mask],
        )

    def __len__(self) -> int:
        return int(self.target.shape[0])


def finite_difference_states(states: np.ndarray, dt: float) -> np.ndarray:
    """Recompute velocities and accelerations from positions only.

    v(t) is the backward difference (x(t) - x(t-1)) / dt, which is exactly the
    simulator's semi-implicit velocity when positions are exact. a(t) is
    (v(t+1) - v(t)) / dt. The first and last frames are dropped.
    """
    layout = state_layout(states)
    position = states[:, :, layout["pos"]].astype(np.float64)
    velocity = np.diff(position, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    out = states[1:-1].astype(np.float64).copy()
    out[:, :, layout["vel"]] = velocity[:-1]
    out[:, :, layout["acc"]] = acceleration
    return out


def pairwise_evidence(trajectories: list[np.ndarray], include_velocity: bool = False, max_frames: int | None = None, seed: int = 0) -> Evidence:
    rows: dict[str, list[np.ndarray]] = {"m1": [], "m2": [], "r": [], "vr": []}
    target, transverse, groups = [], [], []
    used = 0
    for states in trajectories:
        if states.shape[1] != 2:
            continue
        used += 1
        layout = state_layout(states)
        for i, j in ((0, 1), (1, 0)):
            delta = states[:, j, layout["pos"]] - states[:, i, layout["pos"]]
            distance = np.linalg.norm(delta, axis=-1)
            valid = distance > 1e-9
            unit = delta[valid] / distance[valid, None]
            mass_i = states[valid, i, layout["mass"]]
            acceleration = states[valid, i, layout["acc"]]
            velocity = states[valid, i, layout["vel"]]
            rows["m1"].append(mass_i)
            rows["m2"].append(states[valid, j, layout["mass"]])
            rows["r"].append(distance[valid])
            rows["vr"].append((velocity * unit).sum(axis=-1))
            target.append(mass_i * (acceleration * unit).sum(axis=-1))
            transverse.append(mass_i * _perpendicular(acceleration, unit))
            groups.append(np.full(int(valid.sum()), used - 1))
    if not target:
        raise ValueError("Pairwise evidence needs at least one two-body trajectory.")
    variables = PAIRWISE_WITH_VELOCITY if include_velocity else PAIRWISE_VARIABLES
    data = {name: np.concatenate(rows[name]).astype(np.float64) for name in variables}
    evidence = Evidence("pairwise", variables, data, np.concatenate(target), np.concatenate(transverse), used, 0, np.concatenate(groups))
    evidence.frames = len(evidence)
    return _thin(evidence, max_frames, seed)


def single_body_evidence(trajectories: list[np.ndarray], max_frames: int | None = None, seed: int = 0) -> Evidence:
    masses, speeds, target, transverse, groups = [], [], [], [], []
    used = 0
    for states in trajectories:
        if states.shape[1] != 1:
            continue
        used += 1
        layout = state_layout(states)
        velocity = states[:, 0, layout["vel"]]
        speed = np.linalg.norm(velocity, axis=-1)
        valid = speed > 1e-6
        direction = velocity[valid] / speed[valid, None]
        acceleration = states[valid, 0, layout["acc"]]
        mass = states[valid, 0, layout["mass"]]
        masses.append(mass)
        speeds.append(speed[valid])
        target.append(mass * (acceleration * direction).sum(axis=-1))
        transverse.append(mass * _perpendicular(acceleration, direction))
        groups.append(np.full(int(valid.sum()), used - 1))
    if not target:
        raise ValueError("Single-body evidence needs at least one one-body trajectory.")
    evidence = Evidence(
        "single",
        SINGLE_VARIABLES,
        {"m": np.concatenate(masses), "s": np.concatenate(speeds)},
        np.concatenate(target),
        np.concatenate(transverse),
        used,
        0,
        np.concatenate(groups),
    )
    evidence.frames = len(evidence)
    return _thin(evidence, max_frames, seed)


def _thin(evidence: Evidence, max_frames: int | None, seed: int) -> Evidence:
    if max_frames is None or len(evidence) <= max_frames:
        return evidence
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(evidence), dtype=bool)
    mask[rng.choice(len(evidence), size=max_frames, replace=False)] = True
    return evidence.subset(mask)
