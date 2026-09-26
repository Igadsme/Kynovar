"""A sequential experimentation campaign.

Each step: the strategy proposes a design from observable controls, the
laboratory runs it, and the scientist refits a candidate set (generic
families plus a monomial-sum search) on all evidence so far. The leader is
chosen by leave-one-experiment-out predictive log density minus a
complexity penalty. The campaign never sees the hidden law; judging whether
the leader is correct is the caller's (evaluation) job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from kynovar.discovery.evidence import Evidence, pairwise_evidence
from kynovar.discovery.expression import canonicalize, pretty, prune_terms
from kynovar.discovery.lab import ExperimentClient, ExperimentDesign, observable_trouble
from kynovar.discovery.power_sum import power_sum_search
from kynovar.theory.families import PAIRWISE_FAMILIES
from kynovar.uncertainty.parametric import ParametricLaw, fit_uncertain


@dataclass
class Candidate:
    name: str
    model: object
    complexity: int
    cv_score: float


@dataclass
class StepRecord:
    step: int
    design: dict
    accepted: bool
    reason: str | None
    leader: str | None
    leader_expression: object
    cv_scores: dict
    experiments: int
    observations: int
    simulation_steps: int
    seconds: float
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        record = dict(self.__dict__)
        record["leader_expression"] = pretty(self.leader_expression) if self.leader_expression is not None else None
        return record


def _complexity(expression) -> int:
    import sympy

    return int(sum(1 for _ in sympy.preorder_traversal(expression)))


def cross_validated_score(expression, evidence: Evidence, folds: int = 5) -> float:
    """Mean held-out log density per observation over experiment folds."""
    groups = np.unique(evidence.groups)
    if len(groups) < 2:
        return float("-inf")
    chosen = groups if len(groups) <= folds else np.array_split(groups, folds)
    total, count = 0.0, 0
    for held in chosen:
        held = np.atleast_1d(held)
        test = np.isin(evidence.groups, held)
        train = ~test
        law = ParametricLaw.from_expression(expression, evidence.variables)
        model = fit_uncertain(law, {k: v[train] for k, v in evidence.data.items()}, evidence.target[train], None, bootstrap=0)
        density = model.log_density({k: v[test] for k, v in evidence.data.items()}, evidence.target[test])
        density = np.where(np.isfinite(density), density, -1e3)
        total += float(density.sum())
        count += int(test.sum())
    return total / max(count, 1)


def candidate_set(evidence: Evidence, complexity_penalty: float = 0.02, bootstrap: int = 10, seed: int = 0) -> list[Candidate]:
    expressions = dict(PAIRWISE_FAMILIES)
    groups = np.unique(evidence.groups)
    if len(groups) >= 2:
        rng = np.random.default_rng(seed)
        held = rng.choice(groups, size=max(1, len(groups) // 4), replace=False)
        mask = ~np.isin(evidence.groups, held)
        found = power_sum_search(
            {k: v[mask] for k, v in evidence.data.items()}, evidence.target[mask],
            {k: v[~mask] for k, v in evidence.data.items()}, evidence.target[~mask],
            evidence.variables,
        )
        if found.get("expression") is not None:
            expressions["monomial_sum"] = prune_terms(found["expression"], evidence.data, evidence.variables)
    candidates = []
    for name, expression in expressions.items():
        expression = canonicalize(expression)
        law = ParametricLaw.from_expression(expression, evidence.variables)
        model = fit_uncertain(law, evidence.data, evidence.target, evidence.groups, bootstrap=bootstrap, seed=seed)
        if not np.isfinite(model.nmse):
            continue
        complexity = _complexity(model.law.expression())
        score = cross_validated_score(model.law.expression(), evidence) - complexity_penalty * complexity
        candidates.append(Candidate(name, model, complexity, score))
    candidates.sort(key=lambda c: -c.cv_score)
    return candidates


def run_campaign(client: ExperimentClient, strategy, budget: int, initial: list[ExperimentDesign], include_velocity: bool = False, top_k: int = 4, seed: int = 0) -> list[StepRecord]:
    runs: list[np.ndarray] = []
    designs: list[ExperimentDesign] = []
    records: list[StepRecord] = []
    candidates: list[Candidate] = []
    started = time.perf_counter()
    for step in range(budget):
        info = {}
        if step < len(initial):
            design = initial[step]
        else:
            design, info = strategy.propose([c.model for c in candidates[:top_k]], designs, step)
        states = client.run(design)
        designs.append(design)
        reason = observable_trouble(states)
        if reason is None:
            runs.append(states)
        leader = None
        if len(runs) >= 2:
            evidence = pairwise_evidence(runs, include_velocity=include_velocity)
            candidates = candidate_set(evidence, seed=seed + step)
            leader = candidates[0] if candidates else None
        records.append(
            StepRecord(
                step=step,
                design=design.to_dict(),
                accepted=reason is None,
                reason=reason,
                leader=leader.name if leader else None,
                leader_expression=leader.model.law.expression() if leader else None,
                cv_scores={c.name: c.cv_score for c in candidates},
                experiments=client.ledger.experiments,
                observations=client.ledger.observations,
                simulation_steps=client.ledger.steps,
                seconds=time.perf_counter() - started,
                info=info,
            )
        )
    return records
