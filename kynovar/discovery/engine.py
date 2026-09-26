"""SymbolicDiscoveryEngine: from observed evidence to candidate laws.

The engine runs several structure searches on a training subset of the
evidence and ranks the resulting equations on held-out experiments. It never
receives hidden law parameters; its only input is an `Evidence` table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import sympy

from kynovar.discovery.baselines import log_linear_power_law, polynomial_baseline, sindy_baseline
from kynovar.discovery.evidence import Evidence
from kynovar.discovery.expression import canonicalize, drop_null_exponents, expression_complexity, latex, pretty, prune_terms, structure_key, symbol, terms
from kynovar.discovery.power_sum import power_sum_search
from kynovar.discovery.symbolic import RegressorConfig, SymbolicRegressor, weighted_nmse, weights_for


@dataclass
class DiscoveredLaw:
    method: str
    expression: sympy.Expr
    complexity: int
    train_nmse: float
    validation_nmse: float
    validation_rmse: float
    seconds: float
    variables: tuple[str, ...]
    details: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return pretty(self.expression)

    @property
    def latex(self) -> str:
        return latex(self.expression)

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        symbols = [symbol(name) for name in self.variables]
        function = sympy.lambdify(symbols, self.expression, "numpy")
        length = len(next(iter(data.values())))
        with np.errstate(all="ignore"):
            value = function(*[data[name] for name in self.variables])
        return np.asarray(value, dtype=np.float64) * np.ones(length)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "expression": self.text,
            "latex": self.latex,
            "structure": [list(item) for item in structure_key(self.expression)],
            "terms": [
                {"coefficient": term.coefficient, "exponents": {name: value for name, value in term.exponents}}
                for term in terms(self.expression)
            ],
            "complexity": self.complexity,
            "train_nmse": self.train_nmse,
            "validation_nmse": self.validation_nmse,
            "validation_rmse": self.validation_rmse,
            "seconds": self.seconds,
            "details": self.details,
        }


@dataclass(frozen=True)
class EngineConfig:
    regressor: RegressorConfig = RegressorConfig()
    restarts: int = 2
    validation_fraction: float = 0.25
    selection: str = "log"
    selection_penalty: float = 2e-3
    log_selection_penalty: float = 0.1
    error_floor: float = 1e-12
    methods: tuple[str, ...] = ("symbolic", "power_sum", "log_linear", "sindy", "polynomial")
    prune_tolerance: float = 0.01
    seed: int = 0


class SymbolicDiscoveryEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def split(self, evidence: Evidence) -> tuple[Evidence, Evidence]:
        rng = np.random.default_rng(self.config.seed)
        if evidence.groups is not None and len(np.unique(evidence.groups)) >= 4:
            groups = np.unique(evidence.groups)
            held = rng.choice(groups, size=max(1, int(round(len(groups) * self.config.validation_fraction))), replace=False)
            mask = ~np.isin(evidence.groups, held)
        else:
            mask = rng.random(len(evidence)) >= self.config.validation_fraction
        return evidence.subset(mask), evidence.subset(~mask)

    def discover(self, evidence: Evidence) -> list[DiscoveredLaw]:
        """Return every method's law, best first by validation score."""
        train, validation = self.split(evidence)
        laws: list[DiscoveredLaw] = []
        for method in self.config.methods:
            started = time.perf_counter()
            if method == "symbolic":
                laws.extend(self._symbolic(train, validation))
                continue
            if method == "polynomial":
                result = polynomial_baseline(train.data, train.target, train.variables)
            elif method == "sindy":
                result = sindy_baseline(train.data, train.target, train.variables)
            elif method == "power_sum":
                result = power_sum_search(train.data, train.target, validation.data, validation.target, train.variables)
            elif method == "log_linear":
                result = log_linear_power_law(train.data, train.target, train.variables)
            else:
                raise ValueError(f"Unknown discovery method {method!r}.")
            if result.get("expression") is None:
                continue
            details = {"terms": result.get("terms")}
            if "history" in result:
                details["history"] = result["history"]
            laws.append(self._law(method, result["expression"], result["complexity"], result["train_nmse"], validation, time.perf_counter() - started, train.variables, details))
        laws.sort(key=self.selection_score)
        return laws

    def selection_score(self, law: DiscoveredLaw) -> float:
        """Lower is better.

        "log": log(max(validation NMSE, floor)) + log_selection_penalty * complexity,
        so an extra node must cut held-out error by about 10% to be accepted.
        "linear": validation NMSE + selection_penalty * complexity (the original rule).
        """
        if not np.isfinite(law.validation_nmse):
            return float("inf")
        if self.config.selection == "linear":
            return law.validation_nmse + self.config.selection_penalty * law.complexity
        return float(np.log(max(law.validation_nmse, self.config.error_floor)) + self.config.log_selection_penalty * law.complexity)

    def _symbolic(self, train: Evidence, validation: Evidence) -> list[DiscoveredLaw]:
        found = []
        for restart in range(self.config.restarts):
            config = RegressorConfig(**{**self.config.regressor.__dict__, "seed": self.config.regressor.seed + 1000 * restart + self.config.seed})
            regressor = SymbolicRegressor(train.variables, config)
            result = regressor.fit(train.data, train.target)
            candidates = sorted(result.pareto, key=lambda item: item.score)[:5]
            for rank, candidate in enumerate(candidates):
                expression = canonicalize(candidate.expression())
                found.append(
                    self._law(
                        "symbolic",
                        expression,
                        candidate.complexity,
                        candidate.nmse,
                        validation,
                        result.seconds,
                        train.variables,
                        {"restart": restart, "pareto_rank": rank, "generations": result.generations, "evaluations": result.evaluations},
                    )
                )
        return found

    def _law(self, method, expression, complexity, train_nmse, validation: Evidence, seconds, variables, details) -> DiscoveredLaw:
        pruned = drop_null_exponents(prune_terms(expression, validation.data, tuple(variables), self.config.prune_tolerance))
        if pruned != canonicalize(expression):
            details = {**details, "unpruned": pretty(expression)}
            expression = pruned
        details = {**details, "method_reported_complexity": int(complexity)}
        complexity = expression_complexity(expression)
        law = DiscoveredLaw(method, expression, int(complexity), float(train_nmse), float("inf"), float("inf"), float(seconds), tuple(variables), details)
        prediction = law.predict(validation.data)
        if np.isfinite(prediction).all():
            law.validation_nmse = weighted_nmse(prediction, validation.target, weights_for(validation.target))
            law.validation_rmse = float(np.sqrt(np.mean((prediction - validation.target) ** 2)))
        return law


def best_law(laws: list[DiscoveredLaw], method: str | None = None) -> DiscoveredLaw | None:
    chosen = [law for law in laws if method is None or law.method == method]
    return chosen[0] if chosen else None
