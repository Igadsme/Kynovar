"""Evaluation-only catalog of hidden-law worlds for law-discovery experiments.

Each entry builds a sealed `Laboratory` and carries the ground-truth
expression written in the discovery evidence variables. Discovery code must
receive only the laboratory; the `truth` field is for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy

from kynovar.discovery.expression import symbol
from kynovar.laboratory import Laboratory
from kynovar.simulator.forces import ForceLaw, LinearDrag, PairwisePowerLaw, QuadraticDrag, SpringForce
from kynovar.simulator.state import BodyInit
from kynovar.simulator.world import World

M1, M2, R, VR, M, S = (symbol(name) for name in ("m1", "m2", "r", "vr", "m", "s"))


@dataclass(frozen=True)
class HiddenWorld:
    name: str
    family: str
    evidence: str  # "pairwise" or "single"
    laws: tuple[ForceLaw, ...]
    truth: sympy.Expr
    include_velocity: bool = False
    notes: str = ""
    parameters: dict = field(default_factory=dict)

    def laboratory(self, universe_id: str = "K-0042") -> Laboratory:
        placeholder = (BodyInit(id=0, x=0.0, y=0.0, vx=0.0, vy=0.0, mass=1.0, radius=0.05),)
        return Laboratory(World(bodies=placeholder, force_laws=self.laws, dt=0.01, universe_id=universe_id))


def power_law(k: float, p: float, name: str | None = None) -> HiddenWorld:
    return HiddenWorld(
        name or f"power_k{k:g}_p{p:g}",
        "inverse_power",
        "pairwise",
        (PairwisePowerLaw(k=k, p=p),),
        sympy.Float(k) * M1 * M2 * R ** sympy.Float(-p),
        parameters={"k": k, "p": p},
    )


class ChangingLaboratory:
    """Runs experiments in `before` until `change_after` experiments, then in `after`.

    The switch is silent: the scientist-facing interface is identical to a
    `Laboratory`. The change index is evaluation-only knowledge.
    """

    def __init__(self, before: HiddenWorld, after: HiddenWorld, change_after: int | None, universe_id: str = "K-0042") -> None:
        self._before = before.laboratory(universe_id)
        self._after = after.laboratory(universe_id)
        self.change_after = change_after
        self.count = 0

    @property
    def universe_id(self) -> str:
        return self._before.universe_id

    def run(self, experiment):
        chosen = self._before if self.change_after is None or self.count < self.change_after else self._after
        self.count += 1
        return chosen.run(experiment)


def catalog() -> dict[str, HiddenWorld]:
    worlds = [
        power_law(4.0, 3.0, "inverse_cube_k4"),
        power_law(2.5, 2.0, "inverse_square_k2.5"),
        power_law(3.0, 1.5, "inverse_power_k3_p1.5"),
        power_law(1.5, 2.6, "inverse_power_k1.5_p2.6"),
        HiddenWorld(
            "linear_drag_c0.6",
            "linear_drag",
            "single",
            (LinearDrag(0.6),),
            -sympy.Float(0.6) * S,
            parameters={"coefficient": 0.6},
        ),
        HiddenWorld(
            "quadratic_drag_c0.3",
            "quadratic_drag",
            "single",
            (QuadraticDrag(0.3),),
            -sympy.Float(0.3) * S**2,
            parameters={"coefficient": 0.3},
        ),
        HiddenWorld(
            "spring_k2_L1.5",
            "spring",
            "pairwise",
            (SpringForce(2.0, 1.5),),
            sympy.Float(2.0) * R - sympy.Float(3.0),
            parameters={"stiffness": 2.0, "rest_length": 1.5},
            notes="Evidence target is the attractive force k (r - L) = k r - k L; masses do not enter.",
        ),
        HiddenWorld(
            "power_plus_drag",
            "combination",
            "pairwise",
            (PairwisePowerLaw(k=2.0, p=2.0), LinearDrag(0.5)),
            sympy.Float(2.0) * M1 * M2 * R ** sympy.Float(-2.0) - sympy.Float(0.5) * VR,
            include_velocity=True,
            parameters={"k": 2.0, "p": 2.0, "drag": 0.5},
            notes="Per-body drag projected on the line of centers gives -c * vr.",
        ),
    ]
    return {world.name: world for world in worlds}
