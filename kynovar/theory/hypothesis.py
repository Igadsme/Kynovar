"""Hypothesis records.

`score` is the mean log predictive density (nats per observation) of all
evidence the hypothesis has been tested on, under its own predictive
intervals, minus a complexity penalty. It is a comparative figure of merit.
It is NOT a probability and must never be presented as one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from kynovar.discovery.expression import latex, pretty, structure_key
from kynovar.uncertainty.parametric import UncertainLaw

SCORE_SEMANTICS = (
    "Penalized mean log predictive density in nats per observation: mean log N(y | f(x), sigma(x)) over every tested "
    "observation, minus complexity_penalty * complexity. Higher is better. Comparative only; not a probability."
)


class Status(str, Enum):
    PROPOSED = "PROPOSED"
    SUPPORTED = "SUPPORTED"
    CHALLENGED = "CHALLENGED"
    CONTRADICTED = "CONTRADICTED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


ACTIVE = {Status.PROPOSED, Status.SUPPORTED, Status.CHALLENGED}


@dataclass
class ExperimentVerdict:
    experiment_id: str
    verdict: str  # "supports", "contradicts", "inconclusive"
    inside_95: float
    median_z: float
    log_density: float
    observations: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Hypothesis:
    id: str
    model: UncertainLaw
    variables: tuple[str, ...]
    complexity: int
    source: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: Status = Status.PROPOSED
    verdicts: list[ExperimentVerdict] = field(default_factory=list)
    log_density_sum: float = 0.0
    observations: int = 0
    history: list[dict] = field(default_factory=list)
    merged_from: list[str] = field(default_factory=list)
    parent: str | None = None
    complexity_penalty: float = 0.01

    @property
    def expression(self):
        return self.model.law.expression()

    @property
    def equation(self) -> str:
        return pretty(self.expression)

    @property
    def latex(self) -> str:
        return latex(self.expression)

    @property
    def structure(self):
        return structure_key(self.expression)

    @property
    def fit_error(self) -> float:
        return float(self.model.nmse)

    @property
    def supporting(self) -> list[str]:
        return [v.experiment_id for v in self.verdicts if v.verdict == "supports"]

    @property
    def contradicting(self) -> list[str]:
        return [v.experiment_id for v in self.verdicts if v.verdict == "contradicts"]

    @property
    def score(self) -> float | None:
        if self.observations == 0:
            return None
        return self.log_density_sum / self.observations - self.complexity_penalty * self.complexity

    def set_status(self, status: Status, reason: str) -> Status:
        old = self.status
        self.status = status
        self.updated_at = time.time()
        self.history.append({"time": self.updated_at, "event": "status", "old": old.value, "new": status.value, "reason": reason})
        return old

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "equation": self.equation,
            "latex": self.latex,
            "variables": list(self.variables),
            "parameters": self.model.parameter_summary(),
            "parameter_uncertainty_method": f"experiment-level bootstrap ({self.model.bootstrap} resamples)",
            "fit_error": self.fit_error,
            "fit_error_definition": "weighted NMSE on fitting evidence, weights 1/(|y| + median|y|)",
            "complexity": self.complexity,
            "source": self.source,
            "supporting_experiments": self.supporting,
            "contradicting_experiments": self.contradicting,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "score": self.score,
            "score_semantics": SCORE_SEMANTICS,
            "observations_tested": self.observations,
            "merged_from": self.merged_from,
            "parent": self.parent,
            "history": self.history,
        }


def judge(model: UncertainLaw, data, y, contradict_below: float = 0.5, support_above: float = 0.8) -> tuple[str, float, float, np.ndarray]:
    """Compare predictions and intervals with outcomes for one experiment.

    Returns verdict, fraction inside the 95% interval, median |z|, and
    per-observation log densities. Thresholds are fixed in advance:
    fewer than half inside the 95% interval contradicts; at least 80% supports.
    """
    mean, std = model.predict(data)
    std = np.maximum(std, 1e-12)
    z = np.abs(y - mean) / std
    inside = float(np.mean(z <= 1.959964))
    log_density = model.log_density(data, y)
    if inside < contradict_below:
        verdict = "contradicts"
    elif inside >= support_above:
        verdict = "supports"
    else:
        verdict = "inconclusive"
    return verdict, inside, float(np.median(z)), log_density
