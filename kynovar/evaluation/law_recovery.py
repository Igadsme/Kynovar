"""Score a discovered law against a hidden ground-truth expression (evaluation only)."""

from __future__ import annotations

import numpy as np
import sympy

from kynovar.discovery.expression import structure_key, terms


def parameter_errors(discovered: sympy.Expr, truth: sympy.Expr) -> list[dict]:
    """Match additive terms by support and compare coefficients and exponents."""
    found = {term.support: term for term in terms(discovered)}
    rows = []
    for true_term in terms(truth):
        match = found.get(true_term.support)
        row = {
            "support": list(true_term.support),
            "true_coefficient": true_term.coefficient,
            "true_exponents": dict(true_term.exponents),
            "matched": match is not None,
        }
        if match is not None:
            row["coefficient"] = match.coefficient
            row["coefficient_relative_error"] = abs(match.coefficient - true_term.coefficient) / max(abs(true_term.coefficient), 1e-12)
            exponents = dict(match.exponents)
            row["exponents"] = exponents
            row["exponent_abs_error"] = {name: abs(exponents[name] - value) for name, value in true_term.exponents}
        rows.append(row)
    return rows


def recovery_record(discovered: sympy.Expr, truth: sympy.Expr, exponent_tolerance: float = 0.1, coefficient_tolerance: float = 0.1) -> dict:
    """Structural match plus parameter accuracy.

    `structural_match`: identical additive-term supports.
    `recovered`: structural match and every coefficient within
    `coefficient_tolerance` relative error and every exponent within
    `exponent_tolerance` absolute error.
    """
    structural = structure_key(discovered) == structure_key(truth)
    errors = parameter_errors(discovered, truth)
    coefficient_errors = [row["coefficient_relative_error"] for row in errors if row["matched"]]
    exponent_errors = [value for row in errors if row["matched"] for value in row["exponent_abs_error"].values()]
    recovered = bool(
        structural
        and all(row["matched"] for row in errors)
        and all(value <= coefficient_tolerance for value in coefficient_errors)
        and all(value <= exponent_tolerance for value in exponent_errors)
    )
    return {
        "structural_match": bool(structural),
        "recovered": recovered,
        "max_coefficient_relative_error": max(coefficient_errors) if coefficient_errors else None,
        "max_exponent_abs_error": max(exponent_errors) if exponent_errors else None,
        "terms": errors,
        "tolerances": {"coefficient_relative": coefficient_tolerance, "exponent_abs": exponent_tolerance},
    }


def rmse(law, evidence) -> float:
    prediction = law.predict(evidence.data)
    if not np.isfinite(prediction).all():
        return float("inf")
    return float(np.sqrt(np.mean((prediction - evidence.target) ** 2)))
