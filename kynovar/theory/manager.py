"""TheoryManager: the lifecycle of competing hypotheses.

Status rules (fixed in advance, applied to every hypothesis alike):
  * PROPOSED -> SUPPORTED after two supporting experiments and no contradiction.
  * any active status -> CHALLENGED on its first contradicting experiment.
  * CHALLENGED -> CONTRADICTED on a second contradicting experiment, or at
    once if fewer than 20% of an experiment's observations fall inside the
    95% interval.
  * CHALLENGED -> SUPPORTED after three consecutive supporting experiments.
  * SUPERSEDED is set only when another hypothesis replaces this one.
Nothing is ever marked proven.
"""

from __future__ import annotations

import time

import numpy as np
import sympy

from kynovar.discovery.evidence import Evidence
from kynovar.discovery.expression import canonicalize, terms
from kynovar.theory.hypothesis import ACTIVE, ExperimentVerdict, Hypothesis, Status, judge
from kynovar.theory.notebook import Notebook
from kynovar.uncertainty.parametric import ParametricLaw, UncertainLaw, fit_uncertain

STRONG_CONTRADICTION = 0.2


class TheoryManager:
    def __init__(self, notebook: Notebook | None = None, bootstrap: int = 40, complexity_penalty: float = 0.01, seed: int = 0) -> None:
        self.notebook = notebook or Notebook()
        self.hypotheses: dict[str, Hypothesis] = {}
        self.bootstrap = bootstrap
        self.complexity_penalty = complexity_penalty
        self.seed = seed
        self._counter = 0

    # Creation -----------------------------------------------------------------
    def propose(self, expression: sympy.Expr, evidence: Evidence, source: str, complexity: int | None = None, parent: str | None = None) -> Hypothesis:
        law = ParametricLaw.from_expression(canonicalize(expression), evidence.variables)
        model = fit_uncertain(law, evidence.data, evidence.target, evidence.groups, bootstrap=self.bootstrap, seed=self.seed + self._counter)
        return self._register(model, evidence.variables, source, complexity, parent)

    def adopt(self, model: UncertainLaw, variables: tuple[str, ...], source: str, complexity: int | None = None, parent: str | None = None) -> Hypothesis:
        return self._register(model, variables, source, complexity, parent)

    def _register(self, model, variables, source, complexity, parent) -> Hypothesis:
        self._counter += 1
        hypothesis_id = f"H-{self._counter:04d}"
        size = complexity if complexity is not None else _complexity(model.law.expression())
        hypothesis = Hypothesis(hypothesis_id, model, tuple(variables), int(size), source, parent=parent, complexity_penalty=self.complexity_penalty)
        hypothesis.history.append({"time": hypothesis.created_at, "event": "created", "source": source, "equation": hypothesis.equation})
        self.hypotheses[hypothesis_id] = hypothesis
        self.notebook.record(
            "hypothesis_created",
            [hypothesis_id],
            hypothesis_id=hypothesis_id,
            equation=hypothesis.equation,
            fit_error=hypothesis.fit_error,
            complexity=hypothesis.complexity,
            source=source,
            parameters=model.parameter_summary(),
        )
        return hypothesis

    # Evidence -------------------------------------------------------------------
    def test(self, experiment_id: str, evidence: Evidence, hypotheses: list[str] | None = None) -> dict[str, ExperimentVerdict]:
        verdicts = {}
        for hypothesis in self._select(hypotheses):
            verdict, inside, median_z, log_density = judge(hypothesis.model, evidence.data, evidence.target)
            finite = log_density[np.isfinite(log_density)]
            record = ExperimentVerdict(experiment_id, verdict, inside, median_z, float(finite.sum()), int(finite.size))
            hypothesis.verdicts.append(record)
            hypothesis.log_density_sum += record.log_density
            hypothesis.observations += record.observations
            hypothesis.updated_at = record.timestamp
            hypothesis.history.append({"time": record.timestamp, "event": "tested", **record.to_dict()})
            self.notebook.record(
                "hypothesis_updated",
                [hypothesis.id],
                hypothesis_id=hypothesis.id,
                experiment_id=experiment_id,
                verdict=verdict,
                inside=inside,
                median_z=median_z,
                log_density=record.log_density,
                observations=record.observations,
            )
            self._apply_rules(hypothesis, record)
            verdicts[hypothesis.id] = record
        return verdicts

    def _apply_rules(self, hypothesis: Hypothesis, record: ExperimentVerdict) -> None:
        contradictions = len(hypothesis.contradicting)
        if record.verdict == "contradicts":
            if hypothesis.status == Status.CHALLENGED or record.inside_95 < STRONG_CONTRADICTION:
                self._status(hypothesis, Status.CONTRADICTED, f"{contradictions} contradicting experiment(s); latest {record.experiment_id} had {record.inside_95:.0%} inside the 95% interval")
            elif hypothesis.status in (Status.PROPOSED, Status.SUPPORTED):
                self._status(hypothesis, Status.CHALLENGED, f"{record.experiment_id} had {record.inside_95:.0%} inside the 95% interval")
            return
        if record.verdict != "supports":
            return
        if hypothesis.status == Status.PROPOSED and contradictions == 0 and len(hypothesis.supporting) >= 2:
            self._status(hypothesis, Status.SUPPORTED, f"{len(hypothesis.supporting)} supporting experiments, none contradicting")
        elif hypothesis.status == Status.CHALLENGED:
            recent = [v.verdict for v in hypothesis.verdicts[-3:]]
            if len(recent) == 3 and all(v == "supports" for v in recent):
                self._status(hypothesis, Status.SUPPORTED, "three consecutive supporting experiments after a challenge")

    def refit(self, hypothesis_id: str, evidence: Evidence) -> Hypothesis:
        """Refit parameters on accumulated evidence. Structure is unchanged."""
        hypothesis = self.hypotheses[hypothesis_id]
        old = hypothesis.equation
        law = ParametricLaw(hypothesis.model.law.template, hypothesis.model.law.parameters, hypothesis.variables, hypothesis.model.law.values.copy())
        hypothesis.model = fit_uncertain(law, evidence.data, evidence.target, evidence.groups, bootstrap=self.bootstrap, seed=self.seed + self._counter)
        hypothesis.updated_at = time.time()
        hypothesis.history.append({"event": "refit", "old": old, "new": hypothesis.equation, "observations": len(evidence)})
        self.notebook.record("note", [hypothesis_id], text=f"Refit {hypothesis_id} on {len(evidence)} observations: {old} -> {hypothesis.equation}")
        return hypothesis

    # Lifecycle ------------------------------------------------------------------
    def _status(self, hypothesis: Hypothesis, status: Status, reason: str) -> None:
        if hypothesis.status == status:
            return
        old = hypothesis.set_status(status, reason)
        self.notebook.record("status_changed", [hypothesis.id], hypothesis_id=hypothesis.id, old=old.value, new=status.value, reason=reason)

    def reject(self, hypothesis_id: str, reason: str) -> None:
        self._status(self.hypotheses[hypothesis_id], Status.REJECTED, reason)
        self.notebook.record("rejected", [hypothesis_id], hypothesis_id=hypothesis_id, reason=reason)

    def archive(self, hypothesis_id: str, reason: str) -> None:
        self._status(self.hypotheses[hypothesis_id], Status.ARCHIVED, reason)
        self.notebook.record("archived", [hypothesis_id], hypothesis_id=hypothesis_id, reason=reason)

    def reactivate(self, hypothesis_id: str, reason: str) -> None:
        self._status(self.hypotheses[hypothesis_id], Status.PROPOSED, reason)
        self.notebook.record("reactivated", [hypothesis_id], hypothesis_id=hypothesis_id, reason=reason)

    def supersede(self, old_id: str, new_id: str, reason: str) -> None:
        self._status(self.hypotheses[old_id], Status.SUPERSEDED, f"superseded by {new_id}: {reason}")

    def merge_equivalents(self) -> list[tuple[str, str]]:
        """Merge active hypotheses with identical structure and overlapping parameter intervals."""
        merged = []
        active = [h for h in self.hypotheses.values() if h.status in ACTIVE]
        for index, keeper in enumerate(active):
            if keeper.status not in ACTIVE:
                continue
            for other in active[index + 1 :]:
                if other.status not in ACTIVE or other.structure != keeper.structure:
                    continue
                if not _parameters_overlap(keeper.model, other.model):
                    continue
                keeper.merged_from.append(other.id)
                self._status(other, Status.SUPERSEDED, f"merged into {keeper.id}: same structure, overlapping parameter intervals")
                self.notebook.record("merged", [keeper.id, other.id], hypothesis_id=keeper.id, merged_id=other.id, reason="same structure and overlapping 95% parameter intervals")
                merged.append((keeper.id, other.id))
        return merged

    def resolve_nested(self) -> list[tuple[str, str]]:
        """Parsimony rule for nested hypotheses.

        If the additive-term supports of active hypothesis A are a strict
        subset of those of active hypothesis B, both were tested on the same
        observations, and A's score is at least B's, then B is superseded:
        its extra terms add complexity without improving prediction.
        """
        resolved = []
        active = self.active()
        for simple in active:
            for rich in active:
                if simple is rich or rich.status not in ACTIVE or simple.status not in ACTIVE:
                    continue
                small, large = {t for t in simple.structure}, {t for t in rich.structure}
                if not small < large or simple.observations != rich.observations or simple.score is None or rich.score is None:
                    continue
                if simple.score >= rich.score:
                    self.supersede(rich.id, simple.id, f"nested simpler hypothesis {simple.id} scores {simple.score:.3g} >= {rich.score:.3g} on the same observations")
                    resolved.append((simple.id, rich.id))
        return resolved

    # Queries ----------------------------------------------------------------------
    def _select(self, ids: list[str] | None) -> list[Hypothesis]:
        if ids is not None:
            return [self.hypotheses[i] for i in ids]
        return [h for h in self.hypotheses.values() if h.status in ACTIVE]

    def active(self) -> list[Hypothesis]:
        return self._select(None)

    def rank(self, include_inactive: bool = False, record: bool = True) -> list[Hypothesis]:
        pool = list(self.hypotheses.values()) if include_inactive else self.active()
        ranked = sorted(pool, key=lambda h: (h.score is None, -(h.score or 0.0), h.complexity))
        if record and ranked:
            text = ", ".join(f"{h.id} ({h.score:.3g})" if h.score is not None else f"{h.id} (untested)" for h in ranked)
            self.notebook.record("ranking", [h.id for h in ranked], ranking=text)
        return ranked

    def leader(self) -> Hypothesis | None:
        ranked = self.rank(record=False)
        return ranked[0] if ranked else None

    def to_dict(self) -> dict:
        return {"hypotheses": [h.to_dict() for h in self.hypotheses.values()], "notebook": self.notebook.to_list()}


def _complexity(expression: sympy.Expr) -> int:
    return int(sum(1 for _ in sympy.preorder_traversal(expression)))


def _term_vector(expression: sympy.Expr) -> list[float]:
    values = []
    for term in sorted(terms(expression), key=lambda t: t.support):
        values.append(term.coefficient)
        values.extend(value for _, value in term.exponents)
    return values


def _term_intervals(model: UncertainLaw) -> list[tuple[float, float]]:
    point = _term_vector(model.law.expression())
    if model.samples.shape[0] < 2:
        return [(value, value) for value in point]
    draws = [_term_vector(model.law.expression(sample)) for sample in model.samples]
    draws = np.asarray([d for d in draws if len(d) == len(point)])
    if draws.size == 0:
        return [(value, value) for value in point]
    return [(float(np.percentile(draws[:, i], 2.5)), float(np.percentile(draws[:, i], 97.5))) for i in range(len(point))]


def _parameters_overlap(first: UncertainLaw, second: UncertainLaw) -> bool:
    """Every term coefficient and exponent has overlapping bootstrap 95% intervals."""
    a, b = _term_intervals(first), _term_intervals(second)
    if len(a) != len(b):
        return False
    return all(not (left[1] < right[0] or right[1] < left[0]) for left, right in zip(a, b, strict=True))
