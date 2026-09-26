"""Genetic-programming symbolic regression with separate constant fitting.

Structure search mutates expression trees. Continuous constants inside a
structure are then fitted by nonlinear least squares. Candidates are ranked
by a weighted normalized error plus a complexity penalty:

    score = nmse + complexity_penalty * complexity

`nmse` is the weighted mean squared residual divided by the weighted variance
of the target. Weights are 1 / (|y| + median |y|), so a force law spanning
several orders of magnitude is not fitted only at its largest values.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from kynovar.discovery.expression import DEFAULT_BINARY, DEFAULT_UNARY, Node, evaluate


@dataclass(frozen=True)
class RegressorConfig:
    population: int = 160
    generations: int = 30
    max_depth: int = 5
    max_complexity: int = 21
    tournament: int = 5
    complexity_penalty: float = 2e-3
    binary: tuple[str, ...] = DEFAULT_BINARY
    unary: tuple[str, ...] = DEFAULT_UNARY
    fit_samples: int = 400
    fit_iterations: int = 60
    elite: int = 8
    seed: int = 0
    time_limit: float = 120.0


@dataclass
class Candidate:
    tree: Node
    nmse: float
    complexity: int
    score: float

    def expression(self):
        from kynovar.discovery.expression import to_sympy

        return to_sympy(self.tree)


@dataclass
class RegressionResult:
    best: Candidate
    pareto: list[Candidate]
    generations: int
    evaluations: int
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def weights_for(y: np.ndarray) -> np.ndarray:
    scale = float(np.median(np.abs(y))) or 1.0
    return 1.0 / (np.abs(y) + scale)


def weighted_nmse(prediction: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    residual = (prediction - y) * w
    centered = (y - np.average(y, weights=w**2)) * w
    denominator = float(np.mean(centered**2)) or 1e-12
    return float(np.mean(residual**2) / denominator)


def fit_constants(tree: Node, data: dict[str, np.ndarray], y: np.ndarray, w: np.ndarray, iterations: int = 60, restarts: int = 1, rng: np.random.Generator | None = None) -> float:
    """Fit every constant in place. Returns the weighted NMSE, or inf if invalid."""
    constants = tree.constants()
    if not constants:
        prediction = evaluate(tree, data)
        return float("inf") if prediction is None else weighted_nmse(prediction, y, w)
    best_values = np.array([node.value for node in constants])
    best_error = float("inf")
    starts = [best_values]
    generator = rng or np.random.default_rng(0)
    for _ in range(max(0, restarts - 1)):
        starts.append(best_values + generator.normal(0.0, 1.0, size=best_values.shape))

    def residual(values: np.ndarray) -> np.ndarray:
        for node, value in zip(constants, values, strict=True):
            node.value = float(value)
        prediction = evaluate(tree, data)
        if prediction is None:
            return np.full(y.shape, 1e6)
        return (prediction - y) * w

    for start in starts:
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                solution = least_squares(residual, start, max_nfev=iterations, method="trf", x_scale="jac")
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            continue
        for node, value in zip(constants, solution.x, strict=True):
            node.value = float(value)
        prediction = evaluate(tree, data)
        if prediction is None:
            continue
        error = weighted_nmse(prediction, y, w)
        if error < best_error:
            best_error = error
            best_values = solution.x.copy()
    for node, value in zip(constants, best_values, strict=True):
        node.value = float(value)
    return best_error


class SymbolicRegressor:
    def __init__(self, variables: tuple[str, ...], config: RegressorConfig | None = None) -> None:
        self.variables = variables
        self.config = config or RegressorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.evaluations = 0

    # Tree generation ------------------------------------------------------
    def _terminal(self) -> Node:
        if self.rng.random() < 0.7:
            return Node.var(str(self.rng.choice(self.variables)))
        return Node.const(float(self.rng.normal(0.0, 2.0)))

    def random_tree(self, depth: int) -> Node:
        if depth <= 1 or self.rng.random() < 0.3:
            return self._terminal()
        operators = list(self.config.binary) + list(self.config.unary)
        op = str(self.rng.choice(operators))
        if op == "pow":
            # Exponents are constants so a power stays a physical power law.
            return Node("pow", [self.random_tree(depth - 1), Node.const(float(self.rng.normal(0.0, 2.0)))])
        if op in self.config.unary:
            return Node(op, [self.random_tree(depth - 1)])
        return Node(op, [self.random_tree(depth - 1), self.random_tree(depth - 1)])

    # Variation --------------------------------------------------------------
    def _replace_random(self, tree: Node, replacement_fn) -> Node:
        tree = tree.copy()
        nodes = tree.nodes()
        target = nodes[int(self.rng.integers(len(nodes)))]
        new = replacement_fn(target)
        target.op, target.children, target.name, target.value = new.op, new.children, new.name, new.value
        return tree

    def mutate(self, tree: Node) -> Node:
        choice = self.rng.random()
        if choice < 0.35:
            return self._replace_random(tree, lambda _node: self.random_tree(3))
        if choice < 0.55:
            def point(node: Node) -> Node:
                if node.op == "var":
                    return Node.var(str(self.rng.choice(self.variables)))
                if node.op == "const":
                    return Node.const(node.value + float(self.rng.normal(0.0, 1.0)))
                if node.op in ("add", "sub", "mul", "div"):
                    return Node(str(self.rng.choice(["add", "sub", "mul", "div"])), [child.copy() for child in node.children])
                return node.copy()
            return self._replace_random(tree, point)
        if choice < 0.8:
            def wrap(node: Node) -> Node:
                kind = self.rng.random()
                if kind < 0.4:
                    return Node("mul", [node.copy(), Node.var(str(self.rng.choice(self.variables)))])
                if kind < 0.7:
                    return Node("pow", [node.copy(), Node.const(float(self.rng.normal(0.0, 2.0)))])
                if kind < 0.85:
                    return Node("mul", [Node.const(1.0), node.copy()])
                return Node("add", [node.copy(), Node.const(0.0)])
            return self._replace_random(tree, wrap)
        # Hoist: replace the tree with one of its subtrees.
        nodes = tree.nodes()
        return nodes[int(self.rng.integers(len(nodes)))].copy()

    def crossover(self, first: Node, second: Node) -> Node:
        donor_nodes = second.nodes()
        donor = donor_nodes[int(self.rng.integers(len(donor_nodes)))].copy()
        return self._replace_random(first, lambda _node: donor)

    # Scoring ---------------------------------------------------------------
    def score(self, tree: Node, data, y, w) -> Candidate | None:
        complexity = tree.complexity()
        if complexity > self.config.max_complexity or tree.depth() > self.config.max_depth + 2:
            return None
        self.evaluations += 1
        nmse = fit_constants(tree, data, y, w, iterations=self.config.fit_iterations, rng=self.rng)
        if not np.isfinite(nmse):
            return None
        return Candidate(tree, nmse, complexity, nmse + self.config.complexity_penalty * complexity)

    def fit(self, data: dict[str, np.ndarray], y: np.ndarray) -> RegressionResult:
        started = time.perf_counter()
        config = self.config
        y = np.asarray(y, dtype=np.float64)
        if y.shape[0] > config.fit_samples:
            chosen = self.rng.choice(y.shape[0], size=config.fit_samples, replace=False)
            data = {key: value[chosen] for key, value in data.items()}
            y = y[chosen]
        w = weights_for(y)
        population: list[Candidate] = []
        while len(population) < config.population:
            candidate = self.score(self.random_tree(int(self.rng.integers(2, config.max_depth + 1))), data, y, w)
            if candidate is not None:
                population.append(candidate)
        pareto: dict[int, Candidate] = {}
        history = []
        generation = 0
        for generation in range(config.generations):
            population.sort(key=lambda item: item.score)
            for candidate in population:
                best = pareto.get(candidate.complexity)
                if best is None or candidate.nmse < best.nmse:
                    pareto[candidate.complexity] = Candidate(candidate.tree.copy(), candidate.nmse, candidate.complexity, candidate.score)
            history.append({"generation": generation, "best_score": population[0].score, "best_nmse": population[0].nmse})
            if time.perf_counter() - started > config.time_limit:
                break
            next_population = [Candidate(item.tree.copy(), item.nmse, item.complexity, item.score) for item in population[: config.elite]]
            while len(next_population) < config.population:
                parent = self._tournament(population)
                if self.rng.random() < 0.3:
                    child_tree = self.crossover(parent.tree, self._tournament(population).tree)
                else:
                    child_tree = self.mutate(parent.tree)
                child = self.score(child_tree, data, y, w)
                if child is not None:
                    next_population.append(child)
            population = next_population
        front = _pareto_front(list(pareto.values()))
        best = min(front, key=lambda item: item.score)
        return RegressionResult(best, front, generation + 1, self.evaluations, time.perf_counter() - started, history)

    def _tournament(self, population: list[Candidate]) -> Candidate:
        picks = self.rng.choice(len(population), size=min(self.config.tournament, len(population)), replace=False)
        return min((population[int(index)] for index in picks), key=lambda item: item.score)


def _pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    front = []
    best_error = float("inf")
    for candidate in sorted(candidates, key=lambda item: (item.complexity, item.nmse)):
        if candidate.nmse < best_error:
            front.append(candidate)
            best_error = candidate.nmse
    return front
