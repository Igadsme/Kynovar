"""Expression trees for symbolic discovery.

Trees are small and immutable in spirit: mutation code copies before editing.
Numerical evaluation is vectorized and returns None for invalid results, so a
candidate that divides by zero or takes a fractional power of a negative
number is rejected instead of producing a misleading score.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import sympy

BINARY = ("add", "sub", "mul", "div", "pow")
UNARY = ("neg", "sqrt", "abs", "log", "exp", "sin", "cos")
DEFAULT_BINARY = ("add", "sub", "mul", "div", "pow")
DEFAULT_UNARY: tuple[str, ...] = ()
SIGNED_VARIABLES = frozenset({"vr", "vx", "vy", "x", "y"})


def symbol(name: str) -> sympy.Symbol:
    """Variables that can change sign are real; all others are positive."""
    if name in SIGNED_VARIABLES:
        return sympy.Symbol(name, real=True)
    return sympy.Symbol(name, positive=True)


@dataclass
class Node:
    op: str
    children: list["Node"] = field(default_factory=list)
    name: str = ""
    value: float = 0.0

    def copy(self) -> "Node":
        return copy.deepcopy(self)

    # Construction helpers -------------------------------------------------
    @staticmethod
    def var(name: str) -> "Node":
        return Node("var", name=name)

    @staticmethod
    def const(value: float) -> "Node":
        return Node("const", value=float(value))

    # Traversal ------------------------------------------------------------
    def nodes(self) -> list["Node"]:
        found = [self]
        for child in self.children:
            found.extend(child.nodes())
        return found

    def constants(self) -> list["Node"]:
        return [node for node in self.nodes() if node.op == "const"]

    def complexity(self) -> int:
        return len(self.nodes())

    def depth(self) -> int:
        return 1 + max((child.depth() for child in self.children), default=0)

    def variables(self) -> set[str]:
        return {node.name for node in self.nodes() if node.op == "var"}

    def __str__(self) -> str:
        return str(to_sympy(self))


def evaluate(node: Node, data: dict[str, np.ndarray]) -> np.ndarray | None:
    """Vectorized value, or None when any sample is non-finite."""
    with np.errstate(all="ignore"):
        value = _eval(node, data)
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        length = len(next(iter(data.values())))
        value = np.full(length, float(value))
    if not np.isfinite(value).all():
        return None
    return value


def _eval(node: Node, data: dict[str, np.ndarray]):
    op = node.op
    if op == "const":
        return np.float64(node.value)
    if op == "var":
        return data[node.name]
    values = [_eval(child, data) for child in node.children]
    if any(value is None for value in values):
        return None
    if op == "add":
        return values[0] + values[1]
    if op == "sub":
        return values[0] - values[1]
    if op == "mul":
        return values[0] * values[1]
    if op == "div":
        return values[0] / values[1]
    if op == "pow":
        base, exponent = values
        result = np.power(base, exponent)
        return result
    if op == "neg":
        return -values[0]
    if op == "sqrt":
        return np.sqrt(values[0])
    if op == "abs":
        return np.abs(values[0])
    if op == "log":
        return np.log(values[0])
    if op == "exp":
        return np.exp(np.clip(values[0], -50.0, 50.0))
    if op == "sin":
        return np.sin(values[0])
    if op == "cos":
        return np.cos(values[0])
    raise ValueError(f"Unknown operator {op!r}.")


# ---------------------------------------------------------------------------
# Symbolic form, canonicalization, and structure
# ---------------------------------------------------------------------------


def to_sympy(node: Node) -> sympy.Expr:
    op = node.op
    if op == "const":
        return sympy.Float(node.value, 12)
    if op == "var":
        return symbol(node.name)
    args = [to_sympy(child) for child in node.children]
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "div":
        return args[0] / args[1]
    if op == "pow":
        return args[0] ** args[1]
    if op == "neg":
        return -args[0]
    if op == "sqrt":
        return sympy.sqrt(args[0])
    if op == "abs":
        return sympy.Abs(args[0])
    if op == "log":
        return sympy.log(args[0])
    if op == "exp":
        return sympy.exp(args[0])
    if op == "sin":
        return sympy.sin(args[0])
    if op == "cos":
        return sympy.cos(args[0])
    raise ValueError(f"Unknown operator {op!r}.")


def canonicalize(expr: sympy.Expr) -> sympy.Expr:
    """Deterministic simplified form. Positive symbols let powers of products split."""
    expanded = sympy.expand_power_base(sympy.sympify(expr), force=True)
    combined = sympy.powsimp(expanded, force=True, combine="exp")
    return sympy.expand(combined)


def parse_expression(text: str, variables: tuple[str, ...]) -> sympy.Expr:
    symbols = {name: symbol(name) for name in variables}
    return sympy.sympify(text, locals=symbols)


@dataclass(frozen=True)
class Term:
    """One additive term: coefficient * prod(base ** exponent)."""

    coefficient: float
    exponents: tuple[tuple[str, float], ...]

    @property
    def support(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.exponents)


def terms(expr: sympy.Expr) -> list[Term]:
    """Split a canonical expression into additive monomial-like terms.

    A base that is not a plain symbol (for example exp(r)) is kept as a
    string with its numbers replaced by `c`, so it still has a structure.
    """
    canonical = canonicalize(expr)
    found: list[Term] = []
    for term in sympy.Add.make_args(canonical):
        coefficient, rest = term.as_coeff_Mul()
        powers = rest.as_powers_dict() if rest != 1 else {}
        exponents = []
        for base, exponent in powers.items():
            if exponent == 0 or (exponent.is_number and abs(float(exponent)) < 1e-6):
                continue
            name = base.name if isinstance(base, sympy.Symbol) else _numbers_to_c(base)
            if exponent.is_number:
                exponents.append((name, float(exponent)))
            else:
                exponents.append((f"{name}**({_numbers_to_c(exponent)})", 1.0))
        found.append(Term(float(coefficient), tuple(sorted(exponents))))
    return found


def structure_key(expr: sympy.Expr) -> tuple[tuple[str, ...], ...]:
    """Structure with every numeric value treated as a free parameter.

    `m1*m2/r**3`, `m2*m1*r**-3`, and `4.01*m1*m2*r**-2.98` share one key.
    Exponent values are compared separately as parameters.
    """
    return tuple(sorted(term.support for term in terms(expr)))


def _numbers_to_c(expr: sympy.Expr) -> str:
    replaced = expr.replace(lambda item: item.is_Number and item not in (sympy.Integer(1), sympy.Integer(-1)), lambda item: sympy.Symbol("c"))
    return str(replaced)


def latex(expr: sympy.Expr, digits: int = 4) -> str:
    rounded = expr.xreplace({number: sympy.Float(number, digits) for number in expr.atoms(sympy.Float)})
    return sympy.latex(rounded)


def pretty(expr: sympy.Expr, digits: int = 4) -> str:
    rounded = expr.xreplace({number: sympy.Float(number, digits) for number in expr.atoms(sympy.Float)})
    return str(rounded)


def drop_null_exponents(expr: sympy.Expr, tolerance: float = 1e-9) -> sympy.Expr:
    """Replace x**e with 1 when |e| is numerically zero (e.g. 1e-16 from a fit)."""
    return expr.replace(
        lambda item: item.is_Pow and item.exp.is_Number and abs(float(item.exp)) < tolerance,
        lambda item: sympy.Integer(1),
    )


def expression_complexity(expr: sympy.Expr) -> int:
    """Node count of the expression tree: one uniform measure for every method."""
    return int(sum(1 for _ in sympy.preorder_traversal(drop_null_exponents(sympy.sympify(expr)))))


def prune_terms(expr: sympy.Expr, data: dict[str, np.ndarray], variables: tuple[str, ...], tolerance: float = 0.01) -> sympy.Expr:
    """Drop additive terms whose RMS contribution is below `tolerance` of the total's RMS."""
    canonical = canonicalize(expr)
    parts = sympy.Add.make_args(canonical)
    if len(parts) < 2:
        return canonical
    symbols = [symbol(name) for name in variables]
    length = len(next(iter(data.values())))
    values = []
    with np.errstate(all="ignore"):
        for part in parts:
            value = np.asarray(sympy.lambdify(symbols, part, "numpy")(*[data[name] for name in variables]), dtype=np.float64) * np.ones(length)
            values.append(value)
    total = float(np.sqrt(np.mean(np.sum(values, axis=0) ** 2))) or 1.0
    kept = [part for part, value in zip(parts, values, strict=True) if np.sqrt(np.mean(value**2)) >= tolerance * total]
    return sympy.Add(*kept) if kept else canonical


def numerically_equivalent(a: sympy.Expr, b: sympy.Expr, variables: tuple[str, ...], samples: int = 64, seed: int = 0) -> bool:
    """Random-point identity test for two expressions with the same constants."""
    rng = np.random.default_rng(seed)
    symbols = [symbol(name) for name in variables]
    fa = sympy.lambdify(symbols, a, "numpy")
    fb = sympy.lambdify(symbols, b, "numpy")
    points = [rng.uniform(0.3, 3.0, size=samples) for _ in variables]
    with np.errstate(all="ignore"):
        va = np.asarray(fa(*points), dtype=np.float64) * np.ones(samples)
        vb = np.asarray(fb(*points), dtype=np.float64) * np.ones(samples)
    return bool(np.allclose(va, vb, rtol=1e-8, atol=1e-10))
