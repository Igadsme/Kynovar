"""Search for experiments most likely to break the current theory.

Four criteria, each computed from the scientist's own models and the
evidence already collected:

  * uncertainty: leader's predictive std relative to |prediction| along the
    predicted path;
  * disagreement: standardized disagreement between active hypotheses;
  * extrapolation: how far the predicted path leaves the observed box of
    (log m1, log m2, log r), in units of the observed spread;
  * weakly_observed: mean nearest-neighbour distance from path points to
    observed points, restricted to points inside the observed box.

Every challenge stores its prediction and interval before the experiment
runs, then compares them with the outcome. A contradicting outcome becomes a
permanent counterexample.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kynovar.discovery.evidence import Evidence, pairwise_evidence
from kynovar.discovery.lab import DesignRanges, ExperimentClient, ExperimentDesign, observable_trouble
from kynovar.planning.acquisition import standardized_disagreement
from kynovar.planning.design_space import Feasibility, candidate_pool, feasibility_reason
from kynovar.planning.forward import simulate_pair
from kynovar.theory.hypothesis import judge

CRITERIA = ("uncertainty", "disagreement", "extrapolation", "weakly_observed")
FEATURES = ("m1", "m2", "r")


def _log_features(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.log(np.maximum(data[name], 1e-12)) for name in FEATURES])


class ObservedRegion:
    def __init__(self, evidence: Evidence, max_points: int = 1500, seed: int = 0) -> None:
        points = _log_features(evidence.data)
        if len(points) > max_points:
            points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
        self.points = points
        self.low, self.high = points.min(axis=0), points.max(axis=0)
        self.scale = np.maximum(points.std(axis=0), 1e-6)

    def extrapolation(self, data) -> float:
        query = _log_features(data)
        outside = np.maximum(self.low - query, 0) + np.maximum(query - self.high, 0)
        return float(np.mean(np.linalg.norm(outside / self.scale, axis=1)))

    def weak_coverage(self, data) -> float:
        query = _log_features(data)
        inside = np.all((query >= self.low) & (query <= self.high), axis=1)
        if not inside.any():
            return 0.0
        q = query[inside] / self.scale
        p = self.points / self.scale
        distances = np.sqrt(((q[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return float(distances.mean())


@dataclass
class Challenge:
    challenge_id: str
    criterion: str
    target: str
    design: dict
    criterion_score: float
    predicted_force_mean: float
    predicted_interval_95: list[float]
    created_at: float = field(default_factory=time.time)
    outcome: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Counterexample:
    counterexample_id: str
    hypothesis_id: str
    equation: str
    challenge_id: str
    criterion: str
    design: dict
    inside_95: float
    median_z: float
    max_z: float
    observations: int
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class CounterexampleStore:
    """Append-only. Counterexamples are never edited or removed."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.items: list[Counterexample] = []
        if path is not None and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.items.append(Counterexample(**json.loads(line)))

    def add(self, item: Counterexample) -> None:
        self.items.append(item)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item.to_dict()) + "\n")

    def against(self, hypothesis_id: str) -> list[Counterexample]:
        return [item for item in self.items if item.hypothesis_id == hypothesis_id]


class Challenger:
    def __init__(self, ranges: DesignRanges, candidates: int = 64, seed: int = 0, rules: Feasibility | None = None, sampler=None) -> None:
        self.ranges = ranges
        self.sampler = sampler
        self.candidates = candidates
        self.rng = np.random.default_rng(seed)
        self.rules = rules or Feasibility()

    def score(self, design: ExperimentDesign, leader, others, region: ObservedRegion) -> dict[str, float] | None:
        path = simulate_pair(leader.model, design)
        if path is None or not len(path["data"]["r"]):
            return None
        data = {name: path["data"][name] for name in leader.variables}
        mean, std = leader.model.predict(data)
        scores = {
            "uncertainty": float(np.mean(std / (np.abs(mean) + 1e-3))),
            "extrapolation": region.extrapolation(path["data"]),
            "weakly_observed": region.weak_coverage(path["data"]),
            "disagreement": standardized_disagreement([leader.model] + [h.model for h in others], design) if others else 0.0,
        }
        return scores

    def propose(self, leader, others, evidence: Evidence, previous: list[ExperimentDesign]) -> dict[str, tuple[ExperimentDesign, float]]:
        """Best feasible design for each criterion."""
        region = ObservedRegion(evidence)
        models = [leader.model] + [h.model for h in others]
        best: dict[str, tuple[ExperimentDesign, float]] = {}
        pool = candidate_pool(self.rng, self.ranges, self.candidates, self.sampler) if self.sampler else candidate_pool(self.rng, self.ranges, self.candidates)
        for design in pool:
            if feasibility_reason(design, self.ranges, models, previous, self.rules) is not None:
                continue
            scores = self.score(design, leader, others, region)
            if scores is None:
                continue
            for criterion, value in scores.items():
                if np.isfinite(value) and (criterion not in best or value > best[criterion][1]):
                    best[criterion] = (design, value)
        return best

    def challenge(self, client: ExperimentClient, manager, leader, design: ExperimentDesign, criterion: str, score: float, store: CounterexampleStore) -> tuple[Challenge, Evidence | None]:
        """Predict, run, compare, update the theory, and keep any counterexample.

        The prediction and interval are recorded before the experiment runs.
        Returns the challenge record and the new evidence (None if unusable).
        """
        path = simulate_pair(leader.model, design)
        data = {name: path["data"][name] for name in leader.variables}
        mean, low, high = leader.model.interval(data)
        record = Challenge(
            challenge_id=f"C-{uuid.uuid4().hex[:8]}",
            criterion=criterion,
            target=leader.id,
            design=design.to_dict(),
            criterion_score=float(score),
            predicted_force_mean=float(np.mean(mean)),
            predicted_interval_95=[float(np.mean(low)), float(np.mean(high))],
        )
        states = client.run(design)
        reason = observable_trouble(states)
        if reason is not None:
            record.outcome = {"usable": False, "reason": reason}
            manager.notebook.record("challenge", [leader.id], experiment_id=record.challenge_id, target=leader.id, summary=f"unusable outcome ({reason})")
            return record, None
        evidence = pairwise_evidence([states], include_velocity=True)
        verdict, inside, median_z, _ = judge(leader.model, {n: evidence.data[n] for n in leader.variables}, evidence.target)
        observed_mean, predicted_mean = float(np.mean(evidence.target)), float(np.mean(leader.model.predict({n: evidence.data[n] for n in leader.variables})[0]))
        mean_obs, std_obs = leader.model.predict({n: evidence.data[n] for n in leader.variables})
        max_z = float(np.max(np.abs(evidence.target - mean_obs) / np.maximum(std_obs, 1e-12)))
        record.outcome = {"usable": True, "verdict": verdict, "inside_95": inside, "median_z": median_z, "max_z": max_z, "observed_force_mean": observed_mean, "predicted_force_mean_on_observed_path": predicted_mean, "observations": len(evidence)}
        manager.notebook.record("challenge", [leader.id], experiment_id=record.challenge_id, target=f"{leader.id} via {criterion}", summary=f"{verdict}; {inside:.0%} inside the 95% interval")
        manager.test(record.challenge_id, evidence)
        if verdict == "contradicts":
            counterexample = Counterexample(
                counterexample_id=f"X-{len(store.items) + 1:04d}",
                hypothesis_id=leader.id,
                equation=leader.equation,
                challenge_id=record.challenge_id,
                criterion=criterion,
                design=design.to_dict(),
                inside_95=inside,
                median_z=median_z,
                max_z=max_z,
                observations=len(evidence),
            )
            store.add(counterexample)
            manager.notebook.record("counterexample", [leader.id], counterexample_id=counterexample.counterexample_id, hypothesis_id=leader.id, summary=f"{inside:.0%} inside the 95% interval, median |z| {median_z:.1f}")
        self._record_other_counterexamples(manager, leader, record, design, criterion, store)
        return record, evidence

    def _record_other_counterexamples(self, manager, leader, record, design, criterion, store) -> None:
        """A challenge aimed at the leader can also contradict rival hypotheses."""
        for hypothesis in manager.hypotheses.values():
            if hypothesis is leader or not hypothesis.verdicts:
                continue
            verdict = hypothesis.verdicts[-1]
            if verdict.experiment_id != record.challenge_id or verdict.verdict != "contradicts":
                continue
            counterexample = Counterexample(
                counterexample_id=f"X-{len(store.items) + 1:04d}",
                hypothesis_id=hypothesis.id,
                equation=hypothesis.equation,
                challenge_id=record.challenge_id,
                criterion=criterion,
                design=design.to_dict(),
                inside_95=verdict.inside_95,
                median_z=verdict.median_z,
                max_z=float("nan"),
                observations=verdict.observations,
            )
            store.add(counterexample)
            manager.notebook.record("counterexample", [hypothesis.id], counterexample_id=counterexample.counterexample_id, hypothesis_id=hypothesis.id, summary=f"{verdict.inside_95:.0%} inside the 95% interval, median |z| {verdict.median_z:.1f}")
