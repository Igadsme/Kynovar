"""Greedy search over sums of power-law monomials.

Model family:  y = sum_k c_k * prod_{i in S_k} x_i ** a_ik * g_k(signed)

Positive variables get continuous exponents. Variables that change sign
(for example radial velocity) may enter only as v or v*|v|, so no
fractional power of a negative number is ever formed. Terms are added one
at a time; each addition refits every coefficient and exponent by
nonlinear least squares, and the search stops when held-out error stops
improving by a meaningful factor.
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass

import numpy as np
import sympy
from scipy.optimize import least_squares

from kynovar.discovery.expression import symbol
from kynovar.discovery.symbolic import weighted_nmse, weights_for

SIGNED_FORMS = ("", "lin", "quad")
EXPONENT_STARTS = (1.0, -2.0, 2.0, -1.0)


@dataclass(frozen=True)
class TermSpec:
    positive: tuple[str, ...]
    signed: str = ""
    signed_variable: str = ""

    def label(self) -> str:
        parts = list(self.positive)
        if self.signed:
            parts.append(f"{self.signed}({self.signed_variable})")
        return "*".join(parts) or "1"


def _signed_factor(spec: TermSpec, data) -> np.ndarray | float:
    if not spec.signed:
        return 1.0
    value = data[spec.signed_variable]
    return value if spec.signed == "lin" else value * np.abs(value)


def _evaluate(specs, params, data, length):
    total = np.zeros(length)
    offset = 0
    logs = {name: np.log(value) for name, value in data.items() if np.all(value > 0)}
    for spec in specs:
        coefficient = params[offset]
        exponents = params[offset + 1 : offset + 1 + len(spec.positive)]
        offset += 1 + len(spec.positive)
        log_term = np.zeros(length)
        for name, exponent in zip(spec.positive, exponents, strict=True):
            log_term = log_term + exponent * logs[name]
        total = total + coefficient * np.exp(np.clip(log_term, -60, 60)) * _signed_factor(spec, data)
    return total


def _fit(specs, starts, data, y, w):
    length = y.shape[0]

    def residual(params):
        return (_evaluate(specs, params, data, length) - y) * w

    best, best_error = None, float("inf")
    for start in starts:
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                solution = least_squares(residual, start, max_nfev=200, method="trf")
        except (ValueError, np.linalg.LinAlgError):
            continue
        prediction = _evaluate(specs, solution.x, data, length)
        if not np.isfinite(prediction).all():
            continue
        error = weighted_nmse(prediction, y, w)
        if error < best_error:
            best, best_error = solution.x, error
    return best, best_error


def candidate_specs(data: dict[str, np.ndarray], variables: tuple[str, ...], max_factors: int = 3) -> list[TermSpec]:
    positive = [name for name in variables if np.all(data[name] > 0)]
    signed = [name for name in variables if name not in positive]
    specs = []
    for size in range(0, min(max_factors, len(positive)) + 1):
        for subset in itertools.combinations(positive, size):
            specs.append(TermSpec(subset))
            for name in signed:
                specs.append(TermSpec(subset, "lin", name))
                specs.append(TermSpec(subset, "quad", name))
    return specs


def to_expression(specs, params) -> sympy.Expr:
    expression = sympy.Integer(0)
    offset = 0
    for spec in specs:
        term = sympy.Float(float(params[offset]))
        for name, exponent in zip(spec.positive, params[offset + 1 : offset + 1 + len(spec.positive)], strict=True):
            term = term * symbol(name) ** sympy.Float(float(exponent))
        if spec.signed:
            signed = symbol(spec.signed_variable)
            term = term * (signed if spec.signed == "lin" else signed * sympy.Abs(signed))
        offset += 1 + len(spec.positive)
        expression = expression + term
    return expression


def power_sum_search(train_data, train_y, validation_data, validation_y, variables: tuple[str, ...], max_terms: int = 3, improvement: float = 0.5) -> dict:
    """Forward selection of monomial terms, stopped on held-out error."""
    started = time.perf_counter()
    w = weights_for(train_y)
    vw = weights_for(validation_y)
    specs_pool = candidate_specs(train_data, variables)
    chosen: list[TermSpec] = []
    params = np.zeros(0)
    best_validation = float("inf")
    history = []
    for _ in range(max_terms):
        step_best = None
        for spec in specs_pool:
            if spec in chosen:
                continue
            trial = chosen + [spec]
            starts = []
            for exponent in EXPONENT_STARTS:
                new = [1.0] + [exponent if name == "r" else 1.0 for name in spec.positive]
                starts.append(np.concatenate([params, new]))
            fitted, error = _fit(trial, starts, train_data, train_y, w)
            if fitted is None:
                continue
            if step_best is None or error < step_best[2]:
                step_best = (trial, fitted, error)
        if step_best is None:
            break
        trial, fitted, error = step_best
        prediction = _evaluate(trial, fitted, validation_data, validation_y.shape[0])
        validation_error = weighted_nmse(prediction, validation_y, vw) if np.isfinite(prediction).all() else float("inf")
        history.append({"terms": [s.label() for s in trial], "train_nmse": error, "validation_nmse": validation_error})
        # Forward selection answers only whether another term explains held-out
        # signal. Complexity is applied once by SymbolicDiscoveryEngine when it
        # ranks completed laws; including it here can reject an exact second
        # term even when validation error falls effectively to zero.
        if chosen and not validation_error < improvement * best_validation:
            break
        chosen, params, best_validation = trial, fitted, validation_error
        if best_validation < 1e-14:
            break
    if not chosen:
        return {"method": "power_sum", "expression": None, "seconds": time.perf_counter() - started}
    prediction = _evaluate(chosen, params, train_data, train_y.shape[0])
    return {
        "method": "power_sum",
        "expression": to_expression(chosen, params),
        "train_nmse": weighted_nmse(prediction, train_y, w),
        "complexity": _complexity(chosen),
        "terms": len(chosen),
        "history": history,
        "seconds": time.perf_counter() - started,
    }


def _complexity(specs) -> int:
    return int(sum(2 + 3 * len(s.positive) + (2 if s.signed else 0) for s in specs))
