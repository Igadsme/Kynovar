"""Symbolic discovery: expressions, fitting, baselines, evidence, recovery metrics."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from kynovar.discovery.baselines import log_linear_power_law, polynomial_baseline, sindy_baseline
from kynovar.discovery.evidence import finite_difference_states, pairwise_evidence, single_body_evidence
from kynovar.discovery.expression import (
    Node,
    canonicalize,
    evaluate,
    parse_expression,
    prune_terms,
    structure_key,
    symbol,
    terms,
    to_sympy,
)
from kynovar.discovery.lab import ExperimentClient, ExperimentDesign, observable_trouble
from kynovar.discovery.power_sum import power_sum_search
from kynovar.discovery.symbolic import RegressorConfig, SymbolicRegressor, fit_constants, weights_for
from kynovar.evaluation.law_recovery import recovery_record
from kynovar.evaluation.worlds import catalog

VARS = ("m1", "m2", "r")


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return {"m1": rng.uniform(0.5, 2, n), "m2": rng.uniform(0.5, 2, n), "r": rng.uniform(0.5, 3, n)}


def test_structural_equivalence_of_rewritten_laws() -> None:
    a = parse_expression("m1*m2/r**3", VARS)
    b = parse_expression("m2*m1*r**-3", VARS)
    c = parse_expression("4.01*m1*m2*r**-2.98", VARS)
    d = parse_expression("m1/r**3", VARS)
    assert structure_key(a) == structure_key(b) == structure_key(c)
    assert structure_key(a) != structure_key(d)
    assert canonicalize(a) == canonicalize(b)
    assert structure_key(parse_expression("(m1*m2)**1.0/r**3", VARS)) == structure_key(a)


def test_terms_report_coefficients_and_exponents() -> None:
    [term] = terms(parse_expression("4*m1*m2/r**3", VARS))
    assert term.coefficient == pytest.approx(4.0)
    assert dict(term.exponents) == {"m1": 1.0, "m2": 1.0, "r": -3.0}
    # Numerically negligible exponents do not create structure.
    [term] = terms(parse_expression("2*m1**1e-12*r", VARS))
    assert dict(term.exponents) == {"r": 1.0}


def test_invalid_expressions_are_rejected() -> None:
    data = {"x": np.array([-1.0, 0.0, 2.0])}
    assert evaluate(Node("div", [Node.const(1.0), Node.var("x")]), data) is None
    assert evaluate(Node("pow", [Node.var("x"), Node.const(0.5)]), data) is None
    assert evaluate(Node("log", [Node.var("x")]), data) is None
    value = evaluate(Node("add", [Node.var("x"), Node.const(1.0)]), data)
    assert np.allclose(value, [0.0, 1.0, 3.0])


def test_signed_variables_keep_absolute_value() -> None:
    vr = symbol("vr")
    expr = vr * sympy.Abs(vr)
    assert expr != vr**2
    assert symbol("r").is_positive and not symbol("vr").is_positive


def test_constant_fitting_is_separate_from_structure() -> None:
    data = _data()
    y = 4.0 * data["m1"] * data["m2"] / data["r"] ** 3
    tree = Node("mul", [Node.const(1.0), Node("mul", [Node("mul", [Node.var("m1"), Node.var("m2")]), Node("pow", [Node.var("r"), Node.const(-1.0)])])])
    nmse = fit_constants(tree, data, y, weights_for(y), iterations=200)
    assert nmse < 1e-10
    values = sorted(node.value for node in tree.constants())
    assert values == pytest.approx([-3.0, 4.0], rel=1e-5)
    assert structure_key(to_sympy(tree)) == (("m1", "m2", "r"),)


def test_symbolic_regressor_recovers_inverse_cube() -> None:
    data = _data(400)
    y = 4.0 * data["m1"] * data["m2"] / data["r"] ** 3
    result = SymbolicRegressor(VARS, RegressorConfig(population=120, generations=20, seed=1)).fit(data, y)
    record = recovery_record(canonicalize(result.best.expression()), sympy.Float(4.0) * symbol("m1") * symbol("m2") * symbol("r") ** -3)
    assert record["recovered"], result.best.expression()
    front = [candidate.complexity for candidate in result.pareto]
    assert front == sorted(front)


def test_baselines_behind_common_interface() -> None:
    data = _data()
    y = 2.5 * data["m1"] * data["m2"] / data["r"] ** 2
    for baseline in (polynomial_baseline, sindy_baseline, log_linear_power_law):
        result = baseline(data, y, VARS)
        assert set(result) >= {"method", "expression", "train_nmse", "complexity", "seconds"}
    loglinear = log_linear_power_law(data, y, VARS)
    assert recovery_record(loglinear["expression"], parse_expression("2.5*m1*m2/r**2", VARS))["recovered"]
    sindy = sindy_baseline(data, y, VARS)
    assert sindy["train_nmse"] < 1e-6


def test_power_sum_finds_two_term_law() -> None:
    rng = np.random.default_rng(3)
    data = {**_data(500, 3), "vr": rng.normal(0, 0.5, 500)}
    y = 2.0 * data["m1"] * data["m2"] / data["r"] ** 2 - 0.5 * data["vr"]
    half = {key: value[:250] for key, value in data.items()}, {key: value[250:] for key, value in data.items()}
    result = power_sum_search(half[0], y[:250], half[1], y[250:], ("m1", "m2", "r", "vr"))
    truth = parse_expression("2*m1*m2/r**2 - 0.5*vr", ("m1", "m2", "r", "vr"))
    assert recovery_record(prune_terms(result["expression"], data, ("m1", "m2", "r", "vr")), truth)["recovered"]


def test_evidence_from_laboratory_matches_law() -> None:
    world = catalog()["inverse_square_k2.5"]
    client = ExperimentClient(world.laboratory())
    states = client.run(ExperimentDesign.two_body(1.0, 2.0, 2.0, duration=0.5))
    assert observable_trouble(states) is None
    evidence = pairwise_evidence([states])
    expected = 2.5 * evidence.data["m1"] * evidence.data["m2"] / evidence.data["r"] ** 2
    assert np.allclose(evidence.target, expected, rtol=1e-9)
    assert np.abs(evidence.transverse).max() < 1e-9
    assert client.ledger.experiments == 1 and client.ledger.steps == 50


def test_finite_difference_reproduces_semi_implicit_velocity() -> None:
    world = catalog()["inverse_square_k2.5"]
    states = ExperimentClient(world.laboratory()).run(ExperimentDesign.two_body(1.0, 1.0, 2.0, duration=0.5))
    rebuilt = finite_difference_states(states, 0.01)
    assert np.allclose(rebuilt[:, :, 2:4], states[1:-1, :, 2:4], atol=1e-9)
    assert np.allclose(rebuilt[:, :, 4:6], states[2:, :, 4:6], rtol=1e-6, atol=1e-8) or np.allclose(rebuilt[:, :, 4:6], states[1:-1, :, 4:6], rtol=1e-6, atol=1e-8)


def test_single_body_evidence_for_drag() -> None:
    world = catalog()["linear_drag_c0.6"]
    states = ExperimentClient(world.laboratory()).run(ExperimentDesign.single(1.5, 2.0, duration=0.5))
    evidence = single_body_evidence([states])
    assert np.allclose(evidence.target, -0.6 * evidence.data["s"], rtol=1e-6)


def test_recovery_record_flags_wrong_exponent() -> None:
    truth = parse_expression("4*m1*m2/r**3", VARS)
    assert recovery_record(parse_expression("4.05*m1*m2/r**3.02", VARS), truth)["recovered"]
    wrong = recovery_record(parse_expression("4*m1*m2/r**2", VARS), truth)
    assert wrong["structural_match"] and not wrong["recovered"]
    assert not recovery_record(parse_expression("4*m1/r**3", VARS), truth)["structural_match"]
