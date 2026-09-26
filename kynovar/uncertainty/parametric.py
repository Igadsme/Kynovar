"""Parametric law models: fitting, experiment-level bootstrap, predictive intervals.

A discovered expression becomes a template whose numeric constants are free
parameters c0, c1, ... Parameter uncertainty comes from refitting on
bootstrap resamples of whole experiments, so correlated frames within one
experiment are not treated as independent evidence.

Predictive intervals combine the bootstrap spread of the mean prediction
with a heteroscedastic residual model:

    sigma(x)^2 = var_boot(f(x)) + a^2 + (b * f(x))^2

where a (absolute) and b (relative) are fitted to the residuals by maximum
Gaussian likelihood. Intervals are Gaussian with that
sigma. Their coverage is an empirical question; `coverage` measures it on
held-out experiments and `conformal_scale` can widen them to a target level.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import sympy
from scipy.optimize import least_squares, minimize
from scipy.stats import norm

from kynovar.discovery.expression import symbol

LEVELS = (0.5, 0.8, 0.9, 0.95)


def parametrize(expr: sympy.Expr) -> tuple[sympy.Expr, list[sympy.Symbol], np.ndarray]:
    """Replace every Float in `expr` by a parameter symbol. Integers stay structural."""
    floats = sorted(expr.atoms(sympy.Float), key=lambda value: sympy.srepr(value))
    parameters = [sympy.Symbol(f"c{index}", real=True) for index in range(len(floats))]
    template = expr.xreplace(dict(zip(floats, parameters, strict=True)))
    return template, parameters, np.array([float(value) for value in floats], dtype=np.float64)


@dataclass
class ParametricLaw:
    template: sympy.Expr
    parameters: list[sympy.Symbol]
    variables: tuple[str, ...]
    values: np.ndarray
    _function: object = field(default=None, repr=False)

    @staticmethod
    def from_expression(expr: sympy.Expr, variables: tuple[str, ...]) -> "ParametricLaw":
        template, parameters, values = parametrize(sympy.sympify(expr))
        return ParametricLaw(template, parameters, tuple(variables), values)

    def function(self):
        if self._function is None:
            self._function = sympy.lambdify(self.parameters + [symbol(name) for name in self.variables], self.template, "numpy")
        return self._function

    def predict(self, data: dict[str, np.ndarray], values: np.ndarray | None = None) -> np.ndarray:
        values = self.values if values is None else values
        length = len(next(iter(data.values())))
        with np.errstate(all="ignore"):
            out = self.function()(*values, *[data[name] for name in self.variables])
        return np.asarray(out, dtype=np.float64) * np.ones(length)

    def expression(self, values: np.ndarray | None = None) -> sympy.Expr:
        values = self.values if values is None else values
        return self.template.xreplace({p: sympy.Float(float(v)) for p, v in zip(self.parameters, values, strict=True)})

    def fit(self, data, y, start: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Weighted least squares; weights 1/(|y| + median|y|). Returns values and weighted NMSE."""
        start = self.values if start is None else start
        y0 = float(np.median(np.abs(y))) or 1.0
        w = 1.0 / (np.abs(y) + y0)
        if not len(self.parameters):
            prediction = self.predict(data)
            return self.values, _nmse(prediction, y, w)

        def residual(values):
            prediction = self.predict(data, values)
            if not np.isfinite(prediction).all():
                return np.full(y.shape, 1e6)
            return (prediction - y) * w

        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            solution = least_squares(residual, start, method="trf", max_nfev=300)
        return solution.x, _nmse(self.predict(data, solution.x), y, w)


def _nmse(prediction, y, w) -> float:
    if not np.isfinite(prediction).all():
        return float("inf")
    residual = (prediction - y) * w
    centered = (y - np.average(y, weights=w**2)) * w
    return float(np.mean(residual**2) / (np.mean(centered**2) or 1e-12))


@dataclass
class UncertainLaw:
    """A fitted law with bootstrap parameter samples and a residual noise model."""

    law: ParametricLaw
    samples: np.ndarray  # (bootstrap, parameters)
    absolute_sigma: float
    relative_sigma: float
    nmse: float
    interval_scale: float = 1.0
    bootstrap: int = 0

    def parameter_summary(self) -> list[dict]:
        rows = []
        for index, parameter in enumerate(self.law.parameters):
            column = self.samples[:, index] if self.samples.size else np.array([self.law.values[index]])
            rows.append(
                {
                    "name": str(parameter),
                    "value": float(self.law.values[index]),
                    "std": float(np.std(column)),
                    "ci95": [float(np.percentile(column, 2.5)), float(np.percentile(column, 97.5))],
                }
            )
        return rows

    def predict(self, data) -> tuple[np.ndarray, np.ndarray]:
        """Mean prediction and predictive standard deviation."""
        mean = self.law.predict(data)
        if self.samples.shape[0] > 1:
            draws = np.stack([self.law.predict(data, values) for values in self.samples])
            draws = np.where(np.isfinite(draws), draws, np.nan)
            model_var = np.nanvar(draws, axis=0)
        else:
            model_var = np.zeros_like(mean)
        noise_var = self.absolute_sigma**2 + (self.relative_sigma * mean) ** 2
        return mean, self.interval_scale * np.sqrt(model_var + noise_var)

    def interval(self, data, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, std = self.predict(data)
        z = float(norm.ppf(0.5 + level / 2.0))
        return mean, mean - z * std, mean + z * std

    def log_density(self, data, y) -> np.ndarray:
        mean, std = self.predict(data)
        std = np.maximum(std, 1e-12)
        return norm.logpdf(y, loc=mean, scale=std)


def fit_uncertain(law: ParametricLaw, data, y, groups: np.ndarray | None, bootstrap: int = 50, seed: int = 0) -> UncertainLaw:
    values, nmse = law.fit(data, y)
    law.values = values
    absolute_sigma, relative_sigma = fit_noise(law.predict(data), y)
    rng = np.random.default_rng(seed)
    samples = []
    if bootstrap and len(law.parameters):
        unique = np.unique(groups) if groups is not None else None
        for _ in range(bootstrap):
            if unique is not None and len(unique) >= 3:
                chosen = rng.choice(unique, size=len(unique), replace=True)
                index = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
            else:
                index = rng.integers(0, len(y), size=len(y))
            sample, _ = law.fit({k: v[index] for k, v in data.items()}, y[index], start=values)
            samples.append(sample)
    return UncertainLaw(law, np.asarray(samples).reshape(len(samples), len(law.parameters)), absolute_sigma, relative_sigma, nmse, 1.0, len(samples))


def fit_noise(prediction: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood a, b for residual variance a^2 + (b * prediction)^2."""
    keep = np.isfinite(prediction) & np.isfinite(y)
    residual, prediction = (y - prediction)[keep], prediction[keep]
    if residual.size < 3:
        return 0.0, 0.0
    scale = float(np.sqrt(np.mean(residual**2))) or 1e-12

    def nll(log_params):
        a, b = np.exp(log_params)
        variance = a**2 + (b * prediction) ** 2 + 1e-30
        return float(0.5 * np.sum(np.log(variance) + residual**2 / variance))

    starts = [np.log([scale, 1e-3]), np.log([scale * 1e-3, max(scale / (np.mean(np.abs(prediction)) or 1.0), 1e-6)])]
    best = min((minimize(nll, start, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000}) for start in starts), key=lambda r: r.fun)
    a, b = np.exp(best.x)
    return float(a), float(b)


def coverage(model: UncertainLaw, data, y, levels=LEVELS) -> dict[str, float]:
    """Empirical fraction of held-out targets inside each nominal central interval."""
    mean, std = model.predict(data)
    std = np.maximum(std, 1e-300)
    z = np.abs(y - mean) / std
    return {f"{level:.2f}": float(np.mean(z <= norm.ppf(0.5 + level / 2.0))) for level in levels}


def conformal_scale(model: UncertainLaw, data, y, level: float = 0.9) -> float:
    """Factor that makes the `level` interval cover that fraction of calibration data."""
    mean, std = model.predict(data)
    z = np.abs(y - mean) / np.maximum(std / model.interval_scale, 1e-300)
    quantile = float(np.quantile(z, min(1.0, level * (1 + 1 / max(len(z), 1)))))
    return quantile / float(norm.ppf(0.5 + level / 2.0))
