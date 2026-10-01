"""Server-side universe registry.

This module is the only place in the backend that knows hidden laws. It
builds sealed 3D laboratories, hands discovery sessions an ExperimentClient,
and computes evaluation metrics (discovered law vs ground truth) that are
reported to humans, never to the session.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field

import numpy as np
import sympy

from kynovar.discovery.expression import parse_expression, symbol
from kynovar.discovery.lab import ExperimentClient, ExperimentDesign, Instrument
from kynovar.discovery.session import DEFAULT_RANGES, DiscoverySession
from kynovar.evaluation.law_recovery import recovery_record
from kynovar.planning.forward import simulate_pair
from kynovar.simulator.forces import PairwisePowerLaw
from kynovar.simulator.world3d import BodyInit3D, Experiment3D, Laboratory3D, World3D, observation_array_3d

PRESETS = {
    "K-0042": {"k": 3.2, "p": 2.4},
    "K-DEMO": {"k": 4.0, "p": 3.0},
}
TEST_DESIGNS = (
    ExperimentDesign((1.0, 1.5), ((-0.9, 0.0, 0.2), (0.9, 0.1, -0.2)), ((0.0, 0.25, 0.0), (0.0, -0.2, 0.05)), 1.0, 0.01, "eval-orbit"),
    ExperimentDesign((0.8, 1.8), ((-1.2, 0.3, 0.0), (1.1, -0.2, 0.3)), ((0.1, 0.0, 0.1), (-0.05, 0.1, 0.0)), 1.0, 0.01, "eval-infall"),
    ExperimentDesign((1.6, 0.7), ((0.0, -1.3, 0.4), (0.2, 1.2, -0.3)), ((0.2, 0.0, 0.0), (-0.3, 0.0, 0.1)), 1.0, 0.01, "eval-cross"),
)


def universe_id_for(seed: int) -> str:
    return "K-" + hashlib.sha256(f"kynovar-universe-{seed}".encode()).hexdigest()[:4].upper()


@dataclass
class Universe:
    universe_id: str
    k: float
    p: float
    instrument: Instrument
    session: DiscoverySession
    laboratory: Laboratory3D = field(repr=False)

    @property
    def truth(self) -> sympy.Expr:
        return sympy.Float(self.k) * symbol("m1") * symbol("m2") * symbol("r") ** sympy.Float(-self.p)

    def public(self) -> dict:
        return {"universe_id": self.universe_id, "dimension": 3, "governing_laws": "UNKNOWN", **self.session.summary()}


def build_laboratory(universe_id: str, k: float, p: float) -> Laboratory3D:
    placeholder = (BodyInit3D(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.05),)
    return Laboratory3D(World3D(placeholder, (PairwisePowerLaw(k=k, p=p),), dt=0.01, universe_id=universe_id))


class Registry:
    def __init__(self) -> None:
        self.universes: dict[str, Universe] = {}
        self._lock = threading.RLock()

    def create(self, preset: str | None = None, seed: int = 0, noise: float = 0.02, budget: int = 16) -> Universe:
        if preset in PRESETS:
            universe_id, law = preset, PRESETS[preset]
        else:
            rng = np.random.default_rng(seed)
            universe_id = universe_id_for(seed)
            law = {"k": float(np.round(rng.uniform(1.0, 4.0), 2)), "p": float(np.round(rng.uniform(1.2, 3.0), 2))}
        laboratory = build_laboratory(universe_id, law["k"], law["p"])
        instrument = Instrument(acceleration_noise=0.002, relative_acceleration_noise=noise, seed=seed)
        session = DiscoverySession(ExperimentClient(laboratory, instrument), DEFAULT_RANGES, budget=budget, seed=seed)
        universe = Universe(universe_id, law["k"], law["p"], instrument, session, laboratory)
        with self._lock:
            previous = self.universes.get(universe_id)
            if previous is not None:
                previous.session.stop()
            self.universes[universe_id] = universe
        return universe

    def get(self, universe_id: str) -> Universe:
        with self._lock:
            return self.universes[universe_id]

    def list_public(self) -> list[dict]:
        with self._lock:
            return [universe.public() for universe in self.universes.values()]

    def close(self) -> None:
        with self._lock:
            universes = list(self.universes.values())
        for universe in universes:
            universe.session.stop()


def evaluate(universe: Universe) -> dict:
    """Ground-truth comparison for humans. The session never receives this."""
    session = universe.session
    leader = session.manager.leader()
    if leader is None:
        return {"available": False, "reason": "no theory yet"}
    discovered = parse_expression(leader.equation, leader.variables)
    record = recovery_record(discovered, universe.truth)
    rmses = []
    truth_lab = build_laboratory(universe.universe_id + "-EVAL", universe.k, universe.p)
    for design in TEST_DESIGNS:
        actual = observation_array_3d(truth_lab.run(Experiment3D(design.bodies(), design.duration, design.dt)))[:, :, 0:3]
        path = simulate_pair(leader.model, design, stride=1)
        if path is None:
            rmses.append(None)
            continue
        steps = min(len(actual), len(path["positions"]))
        rmses.append(float(np.sqrt(np.mean((actual[:steps] - path["positions"][:steps]) ** 2))))
    first_correct = None
    for entry in session.leader_history:
        if recovery_record(parse_expression(entry["equation"], leader.variables), universe.truth)["recovered"]:
            first_correct = entry["experiments"]
            break
    return {
        "available": True,
        "label": "EVALUATION ONLY: compares Kynovar's theory with the hidden ground truth; Kynovar never sees this.",
        "ground_truth": f"F = {universe.k:g} * m1 * m2 / r^{universe.p:g}",
        "ground_truth_latex": sympy.latex(universe.truth),
        "discovered": leader.equation,
        "discovered_latex": leader.latex,
        "discovered_status": leader.status.value,
        "structural_match": record["structural_match"],
        "recovered": record["recovered"],
        "max_coefficient_relative_error": record["max_coefficient_relative_error"],
        "max_exponent_abs_error": record["max_exponent_abs_error"],
        "trajectory_rmse_test_designs": rmses,
        "trajectory_rmse_mean": float(np.mean([r for r in rmses if r is not None])) if any(r is not None for r in rmses) else None,
        "experiments_to_first_correct_leader": first_correct,
        "experiments_total": session.client.ledger.experiments,
    }
