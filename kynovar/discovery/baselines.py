"""Regression baselines for law discovery: polynomial, sparse (SINDy-style), log-linear power law."""

from __future__ import annotations

import time

import numpy as np
import sympy

from kynovar.discovery.expression import symbol
from kynovar.discovery.symbolic import weighted_nmse, weights_for


def _symbols(variables: tuple[str, ...]) -> dict[str, sympy.Symbol]:
    return {name: symbol(name) for name in variables}


def _weighted_lstsq(columns: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    a = columns * w[:, None]
    b = y * w
    gram = a.T @ a + ridge * np.eye(a.shape[1])
    return np.linalg.solve(gram, a.T @ b)


def polynomial_baseline(data: dict[str, np.ndarray], y: np.ndarray, variables: tuple[str, ...], degree: int = 2) -> dict:
    """Least squares over monomials of the variables and their reciprocals, up to `degree`."""
    started = time.perf_counter()
    symbols = _symbols(variables)
    atoms = [(name, data[name], symbols[name]) for name in variables]
    atoms += [(f"1/{name}", 1.0 / data[name], 1 / symbols[name]) for name in variables if np.all(data[name] > 0)]
    features = [("1", np.ones_like(y), sympy.Integer(1))]
    frontier = [("1", np.ones_like(y), sympy.Integer(1), -1)]
    for _ in range(degree):
        grown = []
        for name, value, expr, last in frontier:
            for index in range(last + 1, len(atoms)) if last >= 0 else range(len(atoms)):
                atom_name, atom_value, atom_expr = atoms[index]
                grown.append((f"{name}*{atom_name}", value * atom_value, expr * atom_expr, index))
        # Allow repeated atoms (squares) by permitting index == last.
        for name, value, expr, last in frontier:
            if last >= 0:
                atom_name, atom_value, atom_expr = atoms[last]
                grown.append((f"{name}*{atom_name}", value * atom_value, expr * atom_expr, last))
        features += [(name, value, expr) for name, value, expr, _ in grown]
        frontier = grown
    columns = np.column_stack([value for _, value, _ in features])
    w = weights_for(y)
    coefficients = _weighted_lstsq(columns, y, w)
    expression = sum(float(c) * expr for c, (_, _, expr) in zip(coefficients, features, strict=True))
    prediction = columns @ coefficients
    return {
        "method": "polynomial",
        "expression": sympy.sympify(expression),
        "train_nmse": weighted_nmse(prediction, y, w),
        "complexity": int(3 * len(features)),
        "terms": len(features),
        "seconds": time.perf_counter() - started,
    }


def sindy_library(data: dict[str, np.ndarray], variables: tuple[str, ...]) -> list[tuple[str, np.ndarray, sympy.Expr]]:
    symbols = _symbols(variables)
    library = [("1", np.ones_like(next(iter(data.values()))), sympy.Integer(1))]
    if "r" in variables:
        m1, m2, r = symbols["m1"], symbols["m2"], symbols["r"]
        for q in np.arange(-1.0, 4.01, 0.25):
            q = float(round(q, 2))
            library.append((f"r^{-q}", data["r"] ** (-q), r ** sympy.Float(-q)))
            library.append((f"m1*m2*r^{-q}", data["m1"] * data["m2"] * data["r"] ** (-q), m1 * m2 * r ** sympy.Float(-q)))
        library.append(("m1", data["m1"], m1))
        library.append(("m1*m2", data["m1"] * data["m2"], m1 * m2))
        if "vr" in variables:
            vr = symbols["vr"]
            library.append(("vr", data["vr"], vr))
            library.append(("vr*|vr|", data["vr"] * np.abs(data["vr"]), vr * sympy.Abs(vr)))
    if "s" in variables:
        s, m = symbols["s"], symbols["m"]
        for power in (1, 2, 3):
            library.append((f"s^{power}", data["s"] ** power, s**power))
        library.append(("m", data["m"], m))
        library.append(("m*s", data["m"] * data["s"], m * s))
    return library


def sindy_baseline(data: dict[str, np.ndarray], y: np.ndarray, variables: tuple[str, ...], threshold: float = 0.05, iterations: int = 10) -> dict:
    """Sequentially thresholded least squares over a fixed term library.

    Exponents are restricted to the library grid (steps of 0.25), so an
    off-grid exponent cannot be recovered exactly by this baseline.
    """
    started = time.perf_counter()
    library = sindy_library(data, variables)
    columns = np.column_stack([value for _, value, _ in library])
    scale = np.sqrt(np.mean(columns**2, axis=0))
    scale[scale == 0] = 1.0
    normalized = columns / scale
    w = weights_for(y)
    active = np.ones(len(library), dtype=bool)
    coefficients = np.zeros(len(library))
    y_scale = float(np.sqrt(np.mean((y * w) ** 2))) or 1.0
    for _ in range(iterations):
        coefficients[:] = 0.0
        if not active.any():
            break
        coefficients[active] = _weighted_lstsq(normalized[:, active], y, w, ridge=1e-6)
        contribution = np.abs(coefficients) * np.sqrt(np.mean((normalized * w[:, None]) ** 2, axis=0)) / y_scale
        new_active = active & (contribution >= threshold)
        if np.array_equal(new_active, active):
            break
        active = new_active
    physical = coefficients / scale
    expression = sum(float(c) * expr for c, (_, _, expr), on in zip(physical, library, active, strict=True) if on)
    prediction = columns @ physical
    return {
        "method": "sindy",
        "expression": sympy.sympify(expression),
        "train_nmse": weighted_nmse(prediction, y, w),
        "complexity": int(4 * active.sum()),
        "terms": int(active.sum()),
        "seconds": time.perf_counter() - started,
    }


def log_linear_power_law(data: dict[str, np.ndarray], y: np.ndarray, variables: tuple[str, ...]) -> dict:
    """Fit y = C * prod(x_i ** a_i) by linear regression in log space.

    Valid only when y keeps one sign and every variable is positive. The
    result is one hypothesis family, not a general law search.
    """
    started = time.perf_counter()
    usable = [name for name in variables if np.all(data[name] > 0)]
    sign = float(np.sign(np.median(y)))
    valid = np.sign(y) == sign
    if sign == 0 or valid.mean() < 0.99 or not usable:
        return {"method": "log_linear", "expression": None, "reason": "target changes sign or variables are not positive", "seconds": time.perf_counter() - started}
    columns = np.column_stack([np.ones(valid.sum())] + [np.log(data[name][valid]) for name in usable])
    solution, *_ = np.linalg.lstsq(columns, np.log(np.abs(y[valid])), rcond=None)
    symbols = _symbols(variables)
    expression = sign * float(np.exp(solution[0]))
    for name, exponent in zip(usable, solution[1:], strict=True):
        expression = expression * symbols[name] ** sympy.Float(float(exponent))
    prediction = sign * np.exp(np.column_stack([np.ones(len(y))] + [np.log(data[name]) for name in usable]) @ solution)
    w = weights_for(y)
    return {
        "method": "log_linear",
        "expression": sympy.sympify(expression),
        "train_nmse": weighted_nmse(prediction, y, w),
        "complexity": int(2 + 3 * len(usable)),
        "terms": 1,
        "seconds": time.perf_counter() - started,
    }
