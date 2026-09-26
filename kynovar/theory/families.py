"""Hypothesis families the scientist can fit to pairwise evidence.

These are generic candidate forms a physicist would try for a central force
between two masses. They are chosen without reference to any hidden world.
"""

from __future__ import annotations

import sympy

from kynovar.discovery.expression import symbol

M1, M2, R = symbol("m1"), symbol("m2"), symbol("r")

PAIRWISE_FAMILIES: dict[str, sympy.Expr] = {
    "power_law": sympy.Float(1.0) * M1 * M2 * R ** sympy.Float(-1.0),
    "exponential_screening": sympy.Float(1.0) * M1 * M2 * sympy.exp(sympy.Float(-1.0) * R),
    "linear_in_r": M1 * M2 * (sympy.Float(1.0) + sympy.Float(-0.5) * R),
    "yukawa": sympy.Float(1.0) * M1 * M2 * sympy.exp(sympy.Float(-0.5) * R) * R ** sympy.Float(-1.0),
}
