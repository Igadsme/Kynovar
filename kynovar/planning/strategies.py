"""Experiment selection strategies with a common interface.

  * passive: observe configurations from a fixed default distribution, the
    way an observer without control would see "natural" events;
  * random: uniform over the full allowed control ranges;
  * grid: a fixed space-filling sweep over separation and mass ratio;
  * active: among feasible random candidates, pick the one with the largest
    standardized disagreement between current hypotheses.
"""

from __future__ import annotations

import numpy as np

from kynovar.discovery.lab import DesignRanges, ExperimentDesign, random_two_body
from kynovar.planning.acquisition import standardized_disagreement
from kynovar.planning.design_space import Feasibility, candidate_pool, feasibility_reason


class Strategy:
    name = "base"

    def __init__(self, ranges: DesignRanges, seed: int = 0, rules: Feasibility | None = None) -> None:
        self.ranges = ranges
        self.rng = np.random.default_rng(seed)
        self.rules = rules or Feasibility()
        self.rejections: dict[str, int] = {}

    def _feasible(self, design, models, previous) -> bool:
        reason = feasibility_reason(design, self.ranges, models, previous, self.rules)
        if reason is not None:
            self.rejections[reason] = self.rejections.get(reason, 0) + 1
        return reason is None

    def propose(self, models, previous: list[ExperimentDesign], step: int) -> tuple[ExperimentDesign, dict]:
        raise NotImplementedError


class PassiveStrategy(Strategy):
    name = "passive"

    def __init__(self, ranges: DesignRanges, seed: int = 0, rules: Feasibility | None = None, natural: DesignRanges | None = None) -> None:
        super().__init__(ranges, seed, rules)
        self.natural = natural or DesignRanges(mass=ranges.mass, separation=(1.3, 1.7), speed=(0.0, 0.1), duration=ranges.duration, dt=ranges.dt)

    def propose(self, models, previous, step):
        return random_two_body(self.rng, self.natural, label=f"passive-{step}"), {}


class RandomStrategy(Strategy):
    name = "random"

    def propose(self, models, previous, step):
        for _ in range(200):
            design = random_two_body(self.rng, self.ranges, label=f"random-{step}")
            if self._feasible(design, models, previous):
                return design, {}
        return design, {"note": "no feasible random design in 200 draws"}


class GridStrategy(Strategy):
    name = "grid"

    def __init__(self, ranges: DesignRanges, seed: int = 0, rules: Feasibility | None = None, points: int = 16) -> None:
        super().__init__(ranges, seed, rules)
        separations = np.linspace(*ranges.separation, 4)
        ratios = np.linspace(0.0, 1.0, 4)
        # Van der Corput-like interleaving so early grid points already span the space.
        order = [0, 15, 5, 10, 3, 12, 6, 9, 1, 14, 4, 11, 2, 13, 7, 8]
        cells = [(s, q) for s in separations for q in ratios]
        self.queue = [cells[i] for i in order[:points]]

    def propose(self, models, previous, step):
        while self.queue:
            separation, ratio = self.queue.pop(0)
            m1 = self.ranges.mass[0] + ratio * (self.ranges.mass[1] - self.ranges.mass[0])
            m2 = self.ranges.mass[1] - ratio * (self.ranges.mass[1] - self.ranges.mass[0]) * 0.5
            design = ExperimentDesign.two_body(float(m1), float(m2), float(separation), duration=self.ranges.duration, dt=self.ranges.dt, label=f"grid-{step}")
            if self._feasible(design, models, previous):
                return design, {}
        return RandomStrategy.propose(self, models, previous, step)


class ActiveStrategy(Strategy):
    name = "active"

    def __init__(self, ranges: DesignRanges, seed: int = 0, rules: Feasibility | None = None, candidates: int = 48) -> None:
        super().__init__(ranges, seed, rules)
        self.candidates = candidates

    def propose(self, models, previous, step):
        if len(models) < 2:
            return RandomStrategy.propose(self, models, previous, step)
        scored = []
        for design in candidate_pool(self.rng, self.ranges, self.candidates):
            if not self._feasible(design, models, previous):
                continue
            scored.append((standardized_disagreement(models, design), design))
        if not scored:
            return RandomStrategy.propose(self, models, previous, step)
        score, design = max(scored, key=lambda item: item[0])
        return ExperimentDesign(design.masses, design.positions, design.velocities, design.duration, design.dt, f"active-{step}"), {"disagreement": score, "feasible_candidates": len(scored)}


STRATEGIES = {cls.name: cls for cls in (PassiveStrategy, RandomStrategy, GridStrategy, ActiveStrategy)}
