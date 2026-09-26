"""Observable design space for two-body experiments and feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kynovar.discovery.lab import DEFAULT_RADIUS, DesignRanges, ExperimentDesign, random_two_body
from kynovar.planning.forward import simulate_pair


@dataclass(frozen=True)
class Feasibility:
    min_initial_gap: float = 0.2
    max_acceleration: float = 60.0
    min_separation: float = 0.25
    max_position: float = 12.0
    duplicate_distance: float = 0.05


def in_range(design: ExperimentDesign, ranges: DesignRanges) -> bool:
    separation = float(np.linalg.norm(np.subtract(design.positions[1], design.positions[0])))
    speeds = [float(np.linalg.norm(v)) for v in design.velocities]
    return (
        all(ranges.mass[0] <= m <= ranges.mass[1] for m in design.masses)
        and ranges.separation[0] - 1e-9 <= separation <= ranges.separation[1] + 1e-9
        and all(ranges.speed[0] - 1e-9 <= s <= ranges.speed[1] + 1e-9 for s in speeds)
    )


def normalized_vector(design: ExperimentDesign, ranges: DesignRanges) -> np.ndarray:
    separation = float(np.linalg.norm(np.subtract(design.positions[1], design.positions[0])))
    span = lambda bounds: (bounds[1] - bounds[0]) or 1.0  # noqa: E731
    parts = [(m - ranges.mass[0]) / span(ranges.mass) for m in design.masses]
    parts.append((separation - ranges.separation[0]) / span(ranges.separation))
    parts += [c / (span(ranges.speed) or 1.0) for v in design.velocities for c in v]
    return np.asarray(parts)


def feasibility_reason(design: ExperimentDesign, ranges: DesignRanges, models, previous: list[ExperimentDesign], rules: Feasibility) -> str | None:
    """Why a design is infeasible, or None. Uses only observable controls and the scientist's own models."""
    if not in_range(design, ranges):
        return "out of range"
    separation = float(np.linalg.norm(np.subtract(design.positions[1], design.positions[0])))
    if separation < 2 * DEFAULT_RADIUS + rules.min_initial_gap:
        return "initial overlap"
    vector = normalized_vector(design, ranges)
    for earlier in previous:
        if np.linalg.norm(vector - normalized_vector(earlier, ranges)) < rules.duplicate_distance:
            return "duplicate"
    for model in models:
        path = simulate_pair(model, design)
        if path is None:
            return "hypothesis prediction invalid"
        if path["max_acceleration"] > rules.max_acceleration or path["min_separation"] < rules.min_separation or path["max_position"] > rules.max_position:
            return "predicted unstable"
    return None


def candidate_pool(rng: np.random.Generator, ranges: DesignRanges, count: int, sampler=random_two_body) -> list[ExperimentDesign]:
    return [sampler(rng, ranges, label=f"candidate-{i}") for i in range(count)]
