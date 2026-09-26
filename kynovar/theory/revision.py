"""Residual monitoring, change detection, theory versions, and autonomous revision.

Validation of a replacement requires, on fresh experiments, that the new
law is supported, the current version is contradicted at least once, and the
new law's fitted noise is at most twice the instrument noise estimated when
the first version was adopted (the instrument is assumed stationary; a
replacement that only "fits" by inflating its noise is refused).

Monitor statistic per experiment: s_t = log(mean z^2), with z the
standardized residuals of the current theory's predictive distribution.
The monitor standardizes s_t using warm-up experiments collected right after
the theory is adopted, then runs a one-sided upper CUSUM

    S_t = max(0, S_{t-1} + (s_t - mu_w) / sd_w - k),   alarm when S_t > h,

with k = 0.5 and h = 5 fixed in advance (textbook values, not tuned here).
The false-positive rate and detection delay of this rule are measured
empirically by the caller.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from kynovar.discovery.evidence import Evidence, pairwise_evidence
from kynovar.discovery.expression import latex, pretty, structure_key, terms
from kynovar.discovery.lab import DesignRanges, ExperimentClient, collect, random_two_body
from kynovar.planning.campaign import candidate_set
from kynovar.theory.hypothesis import judge
from kynovar.theory.notebook import Notebook


def merge_evidence(parts: list[Evidence]) -> Evidence:
    data = {key: np.concatenate([p.data[key] for p in parts]) for key in parts[0].data}
    groups = np.concatenate([np.full(len(p), index) for index, p in enumerate(parts)])
    return Evidence(parts[0].kind, parts[0].variables, data, np.concatenate([p.target for p in parts]), np.concatenate([p.transverse for p in parts]), len(parts), int(sum(len(p) for p in parts)), groups)


def residual_statistic(model, evidence: Evidence) -> float:
    mean, std = model.predict({name: evidence.data[name] for name in model.law.variables})
    z = (evidence.target - mean) / np.maximum(std, 1e-12)
    return float(np.log(np.mean(z**2) + 1e-12))


@dataclass
class ResidualMonitor:
    k: float = 0.5
    h: float = 5.0
    warmup: int = 6
    min_sd: float = 0.1
    values: list[float] = field(default_factory=list)
    cusum: list[float] = field(default_factory=list)
    mu: float | None = None
    sd: float | None = None
    last_zero: int = 0

    def update(self, statistic: float) -> bool:
        """Add one experiment. Returns True when the alarm fires."""
        self.values.append(statistic)
        index = len(self.values) - 1
        if index < self.warmup:
            self.cusum.append(0.0)
            if index == self.warmup - 1:
                warm = np.asarray(self.values)
                self.mu = float(warm.mean())
                self.sd = max(float(warm.std(ddof=1)), self.min_sd)
            return False
        previous = self.cusum[-1] if self.cusum else 0.0
        value = max(0.0, previous + (statistic - self.mu) / self.sd - self.k)
        if value == 0.0:
            self.last_zero = index
        self.cusum.append(value)
        return value > self.h

    def change_estimate(self) -> int:
        """Index of the first experiment after the CUSUM last sat at zero."""
        return max(self.last_zero + 1, self.warmup)


@dataclass
class TheoryVersion:
    name: str
    version: int
    equation: str
    latex: str
    valid_from: int
    valid_to: int | None
    evidence_experiments: list[int]
    reason: str
    differences: dict
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def describe_difference(old, new) -> dict:
    old_terms = {t.support: t for t in terms(old)}
    new_terms = {t.support: t for t in terms(new)}
    changes = {"structure_changed": structure_key(old) != structure_key(new), "terms": []}
    for support in sorted(set(old_terms) | set(new_terms)):
        a, b = old_terms.get(support), new_terms.get(support)
        changes["terms"].append(
            {
                "support": list(support),
                "old_coefficient": a.coefficient if a else None,
                "new_coefficient": b.coefficient if b else None,
                "old_exponents": dict(a.exponents) if a else None,
                "new_exponents": dict(b.exponents) if b else None,
            }
        )
    return changes


class RevisionLoop:
    """Discover, monitor, investigate, rediscover, validate, version."""

    def __init__(self, client: ExperimentClient, ranges: DesignRanges, name: str = "Law K-17", seed: int = 0, initial: int = 6, investigation: int = 6, validation: int = 3, monitor: ResidualMonitor | None = None) -> None:
        self.client = client
        self.ranges = ranges
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.initial = initial
        self.investigation = investigation
        self.validation = validation
        self.monitor = monitor or ResidualMonitor()
        self.notebook = Notebook()
        self.versions: list[TheoryVersion] = []
        self.evidence: list[Evidence] = []
        self.current = None
        self.alarms: list[int] = []
        self.events: list[dict] = []
        self._monitored: list[int] = []

    # Helpers -------------------------------------------------------------------
    def _experiment(self, label: str) -> Evidence | None:
        kept, _ = collect(self.client, [random_two_body(self.rng, self.ranges, label=label)])
        index = self.client.ledger.experiments - 1
        if not kept:
            self.evidence.append(None)
            return None
        evidence = pairwise_evidence(kept)
        self.evidence.append(evidence)
        self.notebook.record("experiment", experiment_id=f"E-{index:04d}", summary=f"{label}, {len(evidence)} observations")
        return evidence

    def _discover(self, indices: list[int]):
        parts = [self.evidence[i] for i in indices if self.evidence[i] is not None]
        candidates = candidate_set(merge_evidence(parts), seed=self.seed + len(self.versions))
        return candidates[0]

    def _adopt(self, candidate, valid_from: int, indices: list[int], reason: str) -> TheoryVersion:
        previous = self.versions[-1] if self.versions else None
        expression = candidate.model.law.expression()
        differences = describe_difference(self.current.model.law.expression(), expression) if self.current is not None else {}
        if previous is not None:
            previous.valid_to = valid_from - 1
        version = TheoryVersion(self.name, len(self.versions) + 1, pretty(expression), latex(expression), valid_from, None, list(indices), reason, differences)
        self.versions.append(version)
        self.current = candidate
        self._reset_monitor()
        self.notebook.record("theory_version", theory_name=self.name, version=version.version, equation=version.equation, reason=reason, valid_from=valid_from)
        return version

    # Loop ------------------------------------------------------------------------
    def run(self, budget: int) -> dict:
        for _ in range(self.initial):
            self._experiment("initial survey")
        indices = list(range(len(self.evidence)))
        first = self._discover(indices)
        self.reference_noise = (first.model.absolute_sigma, first.model.relative_sigma)
        self._adopt(first, 0, indices, "initial discovery")
        while self.client.ledger.experiments < budget:
            evidence = self._experiment("monitoring")
            index = len(self.evidence) - 1
            if evidence is None:
                continue
            self._monitored.append(index)
            alarm = self.monitor.update(residual_statistic(self.current.model, evidence))
            if not alarm:
                continue
            self.alarms.append(index)
            statistic = self.monitor.cusum[-1]
            self.notebook.record("change_detected", experiment_index=index, statistic=statistic, threshold=self.monitor.h)
            outcome = self._investigate(index)
            self.events.append(outcome)
        if self.versions:
            self.versions[-1].valid_to = None
        return {
            "versions": [v.to_dict() for v in self.versions],
            "alarms": self.alarms,
            "investigations": self.events,
            "monitor_values": self.monitor.values,
            "monitored_experiments": self._monitored,
            "ledger": self.client.ledger.to_dict(),
            "notebook": self.notebook.to_list(),
        }

    def _investigate(self, alarm_index: int) -> dict:
        estimate = self._monitored[min(self.monitor.change_estimate(), len(self._monitored) - 1)]
        self.notebook.record("investigation", summary=f"entering investigation mode; change estimated near experiment {estimate}; running {self.investigation} survey experiments")
        start = len(self.evidence)
        for _ in range(self.investigation):
            self._experiment("investigation")
        # Only fresh post-alarm experiments: a real change must precede the alarm.
        post = [i for i in range(start, len(self.evidence)) if self.evidence[i] is not None]
        candidate = self._discover(post)
        validation = []
        for _ in range(self.validation):
            evidence = self._experiment("validation")
            if evidence is None:
                continue
            new_verdict = judge(candidate.model, {n: evidence.data[n] for n in candidate.model.law.variables}, evidence.target)
            old_verdict = judge(self.current.model, {n: evidence.data[n] for n in self.current.model.law.variables}, evidence.target)
            validation.append({"experiment": len(self.evidence) - 1, "new": new_verdict[0], "new_inside": new_verdict[1], "old": old_verdict[0], "old_inside": old_verdict[1]})
        reference = self.reference_noise
        sharp = candidate.model.relative_sigma <= 2 * max(reference[1], 0.01) and candidate.model.absolute_sigma <= 2 * max(reference[0], 1e-3)
        validated = bool(validation) and sharp and all(v["new"] == "supports" for v in validation) and any(v["old"] == "contradicts" for v in validation)
        outcome = {
            "alarm_index": alarm_index,
            "estimated_change_index": estimate,
            "candidate": pretty(candidate.model.law.expression()),
            "validation": validation,
            "candidate_noise": [candidate.model.absolute_sigma, candidate.model.relative_sigma],
            "reference_noise": list(reference),
            "sharp": bool(sharp),
            "validated": validated,
        }
        if validated:
            estimate = self.likelihood_change_point(candidate)
            outcome["estimated_change_index"] = estimate
            outcome["change_point_method"] = "argmax over tau of sum log density: old law before tau, new law from tau"
            used = post + [v["experiment"] for v in validation]
            self._adopt(candidate, estimate, used, f"residual alarm at experiment {alarm_index}; replacement supported by {len(validation)} validation experiments while the previous version was contradicted")
        else:
            self.notebook.record("investigation", summary=f"replacement candidate {outcome['candidate']} failed validation; keeping v{self.versions[-1].version} and restarting the CUSUM with the original baseline")
            self._restart_cusum()
        return outcome

    def likelihood_change_point(self, candidate) -> int:
        """Split point maximizing old-law fit before it and new-law fit after it."""
        start = self.versions[-1].valid_from
        indices = [i for i in range(start, len(self.evidence)) if self.evidence[i] is not None]
        old, new = [], []
        for i in indices:
            evidence = self.evidence[i]
            old.append(float(np.sum(np.maximum(self.current.model.log_density({n: evidence.data[n] for n in self.current.model.law.variables}, evidence.target), -50))))
            new.append(float(np.sum(np.maximum(candidate.model.log_density({n: evidence.data[n] for n in candidate.model.law.variables}, evidence.target), -50))))
        old, new = np.asarray(old), np.asarray(new)
        totals = [old[:t].sum() + new[t:].sum() for t in range(1, len(indices))]
        return indices[1 + int(np.argmax(totals))] if totals else indices[0]

    def _restart_cusum(self) -> None:
        """Keep the adoption-time baseline; only the accumulator restarts."""
        monitor = self.monitor
        if monitor.mu is None:
            return
        monitor.cusum.append(0.0)
        monitor.values.append(monitor.mu)
        monitor.last_zero = len(monitor.values) - 1
        self._monitored.append(self._monitored[-1] if self._monitored else 0)

    def _reset_monitor(self) -> None:
        self.monitor = ResidualMonitor(self.monitor.k, self.monitor.h, self.monitor.warmup, self.monitor.min_sd)
        self._monitored = []
