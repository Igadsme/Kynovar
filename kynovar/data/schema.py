"""Observable tensor layout for dynamics datasets.

Training code uses these channels only. Hidden-law fields are not part of the layout.
"""

from __future__ import annotations

STATE_FEATURES: tuple[str, ...] = (
    "x",
    "y",
    "vx",
    "vy",
    "ax",
    "ay",
    "mass",
    "radius",
)
NODE_FEATURES: tuple[str, ...] = ("x", "y", "vx", "vy", "mass", "radius")
EDGE_FEATURES: tuple[str, ...] = ("dx", "dy", "distance", "dvx", "dvy")

POS = slice(0, 2)
VEL = slice(2, 4)
ACC = slice(4, 6)
MASS = 6
RADIUS = 7
NODE_INDEX = (0, 1, 2, 3, 6, 7)

SCHEMA_VERSION = 1
