"""Hypotheses, theory manager lifecycle, notebook, and uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from kynovar.discovery.evidence import Evidence
from kynovar.discovery.expression import parse_expression
from kynovar.theory.hypothesis import SCORE_SEMANTICS, Status
from kynovar.theory.manager import TheoryManager
from kynovar.theory.notebook import Notebook
from kynovar.uncertainty.parametric import ParametricLaw, conformal_scale, coverage, fit_noise, fit_uncertain, parametrize

VARS = ("m1", "m2", "r")


def _evidence(k=2.5, p=2.0, r_range=(0.8, 3.0), experiments=8, frames=40, noise=0.03, seed=0) -> Evidence:
    rng = np.random.default_rng(seed)
    n = experiments * frames
    data = {"m1": rng.uniform(0.5, 2, n), "m2": rng.uniform(0.5, 2, n), "r": rng.uniform(*r_range, n)}
    clean = k * data["m1"] * data["m2"] / data["r"] ** p
    target = clean + rng.normal(0, 1, n) * (0.005 + noise * np.abs(clean))
    groups = np.repeat(np.arange(experiments), frames)
    return Evidence("pairwise", VARS, data, target, np.zeros(n), experiments, n, groups)


def test_parametrize_replaces_floats_only() -> None:
    template, parameters, values = parametrize(parse_expression("2.5*m1*m2*r**(-2.0)", VARS))
    assert len(parameters) == 2
    assert sorted(values) == pytest.approx([-2.0, 2.5])
    assert {str(s) for s in template.free_symbols} == {"m1", "m2", "r", "c0", "c1"}


def test_bootstrap_interval_contains_truth_and_noise_is_recovered() -> None:
    evidence = _evidence(seed=1)
    law = ParametricLaw.from_expression(parse_expression("1.0*m1*m2*r**(-1.0)", VARS), VARS)
    model = fit_uncertain(law, evidence.data, evidence.target, evidence.groups, bootstrap=30, seed=0)
    summary = {round(row["value"]): row for row in model.parameter_summary()}
    k_row = next(row for row in model.parameter_summary() if row["value"] > 0)
    p_row = next(row for row in model.parameter_summary() if row["value"] < 0)
    assert k_row["ci95"][0] <= 2.5 <= k_row["ci95"][1]
    assert p_row["ci95"][0] <= -2.0 <= p_row["ci95"][1]
    assert model.relative_sigma == pytest.approx(0.03, rel=0.3)
    assert summary


def test_fit_noise_recovers_absolute_and_relative_parts() -> None:
    rng = np.random.default_rng(2)
    prediction = rng.uniform(0.01, 10, 5000)
    y = prediction + rng.normal(0, 1, 5000) * np.sqrt(0.02**2 + (0.05 * prediction) ** 2)
    a, b = fit_noise(prediction, y)
    assert a == pytest.approx(0.02, rel=0.25)
    assert b == pytest.approx(0.05, rel=0.1)


def test_coverage_is_near_nominal_on_fresh_data() -> None:
    fitting, fresh = _evidence(seed=3), _evidence(seed=4)
    law = ParametricLaw.from_expression(parse_expression("1.0*m1*m2*r**(-1.0)", VARS), VARS)
    model = fit_uncertain(law, fitting.data, fitting.target, fitting.groups, bootstrap=20)
    measured = coverage(model, fresh.data, fresh.target)
    assert abs(measured["0.90"] - 0.9) < 0.05
    assert abs(measured["0.50"] - 0.5) < 0.07
    assert conformal_scale(model, fresh.data, fresh.target, 0.9) == pytest.approx(1.0, abs=0.2)


def test_manager_lifecycle_and_evidence_updates() -> None:
    manager = TheoryManager(bootstrap=20)
    early = _evidence(r_range=(1.3, 1.7), seed=5)
    power = manager.propose(parse_expression("1.0*m1*m2*r**(-1.0)", VARS), early, "test:power")
    linear = manager.propose(parse_expression("m1*m2*(1.0 - 0.5*r)", VARS), early, "test:linear")
    assert power.status == Status.PROPOSED and power.score is None
    for index in range(2):
        manager.test(f"near-{index}", _evidence(r_range=(1.3, 1.7), experiments=1, seed=10 + index))
    assert power.status == Status.SUPPORTED
    assert linear.status == Status.SUPPORTED
    manager.test("far-0", _evidence(r_range=(2.5, 3.5), experiments=1, seed=20))
    assert linear.status == Status.CONTRADICTED
    assert power.status == Status.SUPPORTED
    assert linear.contradicting == ["far-0"] and "far-0" in power.supporting
    assert manager.leader() is power
    record = power.to_dict()
    for key in ("id", "equation", "parameters", "fit_error", "complexity", "supporting_experiments", "contradicting_experiments", "created_at", "updated_at", "status", "score", "score_semantics"):
        assert key in record
    assert "not a probability" in SCORE_SEMANTICS
    assert "PROVEN" not in {status.value for status in Status}


def test_merge_reject_archive_reactivate_and_history() -> None:
    manager = TheoryManager(bootstrap=20)
    evidence = _evidence(seed=6)
    first = manager.propose(parse_expression("1.0*m1*m2*r**(-1.0)", VARS), evidence, "a")
    second = manager.propose(parse_expression("3.0*m2*m1*r**(-2.5)", VARS), evidence, "b")
    other = manager.propose(parse_expression("1.0*m1*r**(-1.0)", VARS), evidence, "c")
    merged = manager.merge_equivalents()
    assert merged == [(first.id, second.id)]
    assert second.status == Status.SUPERSEDED and first.merged_from == [second.id]
    manager.reject(other.id, "test rejection")
    assert other.status == Status.REJECTED
    manager.reactivate(other.id, "new evidence")
    assert other.status == Status.PROPOSED
    manager.archive(other.id, "set aside")
    assert other.status == Status.ARCHIVED
    kinds = [event.kind for event in manager.notebook.events]
    assert kinds.count("hypothesis_created") == 3 and "merged" in kinds and "rejected" in kinds
    assert [h["event"] for h in other.history][0] == "created"


def test_nested_parsimony_rule() -> None:
    manager = TheoryManager(bootstrap=10)
    evidence = _evidence(seed=7)
    simple = manager.propose(parse_expression("1.0*m1*m2*r**(-1.0)", VARS), evidence, "simple")
    rich = manager.propose(parse_expression("1.0*m1*m2*r**(-1.0) + 0.01*m1*m2", VARS), evidence, "rich")
    manager.test("e1", _evidence(experiments=2, seed=8))
    if simple.score >= rich.score:
        assert manager.resolve_nested() == [(simple.id, rich.id)]
        assert rich.status == Status.SUPERSEDED
    else:
        assert manager.resolve_nested() == []


def test_notebook_text_is_rendered_from_payload() -> None:
    notebook = Notebook()
    seen = []
    notebook.subscribe(seen.append)
    event = notebook.record("hypothesis_updated", ["H-0001"], hypothesis_id="H-0001", experiment_id="E-1", verdict="supports", inside=0.93, median_z=0.7)
    assert event.text == "Tested H-0001 on E-1: supports; 93% of observations inside its 95% interval (median |z| 0.70)."
    assert seen == [event]
    broken = notebook.record("hypothesis_updated", hypothesis_id="H-2")
    assert broken.text.startswith("hypothesis_updated:")
