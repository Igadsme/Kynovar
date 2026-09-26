"""An interactive discovery session: the scientist's state behind the web lab.

The session owns an `ExperimentClient` and nothing else from the world. It
runs experiments (chosen autonomously or by the user), maintains competing
hypotheses, answers "predict the future" requests from its leading theory,
and runs challenge experiments that can contradict it.
"""

from __future__ import annotations

import threading
import time
import uuid

import numpy as np

from kynovar.discovery.evidence import pairwise_evidence
from kynovar.discovery.expression import structure_key
from kynovar.discovery.lab import DesignRanges, ExperimentClient, ExperimentDesign, observable_trouble, random_two_body_3d
from kynovar.falsification.challenger import CRITERIA, Challenger, CounterexampleStore
from kynovar.planning.acquisition import standardized_disagreement
from kynovar.planning.campaign import candidate_set
from kynovar.planning.design_space import Feasibility, feasibility_reason
from kynovar.planning.forward import simulate_pair
from kynovar.theory.hypothesis import ACTIVE, Status
from kynovar.theory.manager import TheoryManager

DEFAULT_RANGES = DesignRanges(separation=(1.0, 3.0), speed=(0.0, 0.35), duration=1.0)


def _round(array, digits=4):
    return np.round(np.asarray(array, dtype=float), digits).tolist()


def _finite_prefix(positions: np.ndarray) -> np.ndarray:
    finite = np.isfinite(positions).all(axis=(1, 2))
    stop = int(np.argmin(finite)) if not finite.all() else len(positions)
    return positions[:stop]


class DiscoverySession:
    def __init__(self, client: ExperimentClient, ranges: DesignRanges = DEFAULT_RANGES, budget: int = 16, seed: int = 0) -> None:
        self.client = client
        self.ranges = ranges
        self.budget = budget
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.manager = TheoryManager(bootstrap=24, seed=seed)
        self.store = CounterexampleStore()
        self.challenger = Challenger(ranges, candidates=48, seed=seed + 1, sampler=random_two_body_3d)
        self.status = "UNINITIALIZED"
        self.experiments: list[dict] = []
        self.runs: list[np.ndarray] = []
        self.designs: list[ExperimentDesign] = []
        self.candidates = []
        self.leader_history: list[dict] = []
        self.pending: dict[str, dict] = {}
        self.events: list[dict] = []
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.manager.notebook.subscribe(lambda event: self._emit("notebook", event.to_dict()))

    # Event log -------------------------------------------------------------------
    def _emit(self, kind: str, payload: dict) -> None:
        with self.lock:
            self.events.append({"index": len(self.events), "type": kind, "time": time.time(), "payload": payload})

    def events_since(self, index: int) -> list[dict]:
        with self.lock:
            return self.events[index:]

    # Experiments -----------------------------------------------------------------
    def run_experiment(self, design: ExperimentDesign, source: str) -> dict:
        with self.lock:
            states = self.client.run(design)
            reason = observable_trouble(states)
            experiment_id = f"E-{len(self.experiments) + 1:04d}"
            record = {
                "id": experiment_id,
                "source": source,
                "design": design.to_dict(),
                "accepted": reason is None,
                "reason": reason,
                "dt": design.dt,
                "frame_stride": 2,
                "positions": _round(states[::2, :, 0:3]),
                "masses": list(design.masses),
            }
            self.experiments.append(record)
            self.designs.append(design)
            self.manager.notebook.record("experiment", experiment_id=experiment_id, summary=f"{source}; {'usable' if reason is None else 'rejected: ' + reason}")
            self._emit("experiment", record)
            if reason is None:
                self.runs.append(states)
                self._update_theory(experiment_id, states)
            return record

    def _update_theory(self, experiment_id: str, states: np.ndarray) -> None:
        new = pairwise_evidence([states])
        if self.manager.active():
            self.manager.test(experiment_id, new)
        if len(self.runs) < 2:
            return
        evidence = pairwise_evidence(self.runs)
        self.candidates = candidate_set(evidence, seed=self.seed + len(self.runs))
        known = {h.structure for h in self.manager.hypotheses.values()}
        for candidate in self.candidates[:3]:
            key = structure_key(candidate.model.law.expression())
            if key not in known:
                self.manager.adopt(candidate.model, evidence.variables, f"family:{candidate.name}", candidate.complexity)
                known.add(key)
        for hypothesis in self.manager.active():
            self.manager.refit(hypothesis.id, evidence)
        self.manager.merge_equivalents()
        if all(h.observations for h in self.manager.active()):
            self.manager.resolve_nested()
        leader = self.manager.leader()
        if leader is not None:
            self.leader_history.append({"experiments": self.client.ledger.experiments, "hypothesis": leader.id, "equation": leader.equation, "status": leader.status.value})
        self._emit("theory", self.theory_state())

    def choose_design(self) -> tuple[ExperimentDesign, str]:
        if len(self.runs) < 3 or len(self.candidates) < 2:
            return random_two_body_3d(self.rng, self.ranges, label="survey"), "survey (random design)"
        models = [c.model for c in self.candidates[:4]]
        best, best_score = None, -np.inf
        for _ in range(32):
            design = random_two_body_3d(self.rng, self.ranges)
            if feasibility_reason(design, self.ranges, models, self.designs, Feasibility()) is not None:
                continue
            score = standardized_disagreement(models, design)
            if score > best_score:
                best, best_score = design, score
        if best is None:
            return random_two_body_3d(self.rng, self.ranges), "survey (no feasible active design)"
        return best, f"active (standardized disagreement {best_score:.3g})"

    # Autonomous loop ----------------------------------------------------------------
    def start(self) -> None:
        with self.lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self.status = "RUNNING"
            self._emit("status", {"status": self.status})
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self.lock:
            if self.status == "RUNNING":
                self.status = "STOPPED"
                self._emit("status", {"status": self.status})

    def _loop(self) -> None:
        try:
            while not self._stop.is_set() and self.client.ledger.experiments < self.budget:
                design, source = self.choose_design()
                self.run_experiment(design, source)
                if self._converged():
                    with self.lock:
                        self.status = "CONVERGED"
                        leader = self.manager.leader()
                        self.manager.notebook.record("note", text=f"Leading hypothesis {leader.id} has been SUPPORTED with an unchanged structure for 4 consecutive experiments; autonomous search paused. Status is SUPPORTED, not proven.")
                        self._emit("status", {"status": self.status})
                    return
                time.sleep(0.05)
            with self.lock:
                if self.status == "RUNNING":
                    self.status = "BUDGET_EXHAUSTED" if self.client.ledger.experiments >= self.budget else "STOPPED"
                    self._emit("status", {"status": self.status})
        except Exception as error:  # surfaced to the UI instead of dying silently
            with self.lock:
                self.status = "ERROR"
                self._emit("status", {"status": self.status, "error": repr(error)})

    def _converged(self, window: int = 4) -> bool:
        if len(self.leader_history) < window:
            return False
        recent = self.leader_history[-window:]
        return len({r["hypothesis"] for r in recent}) == 1 and all(r["status"] == Status.SUPPORTED.value for r in recent)

    # Challenge ------------------------------------------------------------------------
    def predict(self, design: ExperimentDesign, criterion: str = "user") -> dict:
        with self.lock:
            leader = self.manager.leader()
            if leader is None:
                raise ValueError("No theory yet: run experiments first.")
            path = simulate_pair(leader.model, design, stride=2)
            if path is None:
                raise ValueError("The leading theory produces invalid values for this design.")
            data = {n: path["data"][n] for n in leader.variables}
            mean, low, high = leader.model.interval(data)
            challenge_id = f"C-{uuid.uuid4().hex[:8]}"
            self.pending[challenge_id] = {"design": design, "leader": leader.id, "criterion": criterion, "predicted_positions": path["positions"]}
            return {
                "challenge_id": challenge_id,
                "hypothesis": leader.id,
                "equation": leader.equation,
                "latex": leader.latex,
                "design": design.to_dict(),
                "criterion": criterion,
                "frame_stride": 2,
                "predicted_positions": _round(path["positions"]),
                "predicted_force": {"mean": _round(mean), "low95": _round(low), "high95": _round(high)},
                "predicted_max_acceleration": path["max_acceleration"],
            }

    def adversarial_design(self, criterion: str | None = None) -> tuple[ExperimentDesign, str, float]:
        with self.lock:
            leader = self.manager.leader()
            if leader is None:
                raise ValueError("No theory to break yet.")
            others = [h for h in self.manager.active() if h is not leader]
            evidence = pairwise_evidence(self.runs)
            options = self.challenger.propose(leader, others, evidence, self.designs)
            if not options:
                raise ValueError("No feasible challenge design found.")
            if criterion is None or criterion not in options:
                criterion = "extrapolation" if "extrapolation" in options else next(iter(options))
            design, score = options[criterion]
            return design, criterion, float(score)

    def reveal(self, challenge_id: str) -> dict:
        with self.lock:
            pending = self.pending.pop(challenge_id, None)
            if pending is None:
                raise KeyError(challenge_id)
            leader = self.manager.hypotheses[pending["leader"]]
            before = len(self.store.items)
            record, evidence = self.challenger.challenge(self.client, self.manager, leader, pending["design"], pending["criterion"], 0.0, self.store)
            # The challenge always runs through the client, even when the outcome is
            # unusable for testing; mirror it in the experiment log.
            states = self.client_last_states()
            predicted = pending["predicted_positions"]
            actual = _finite_prefix(states[::2, :, 0:3]) if states is not None else None
            rmse = None
            if actual is not None and evidence is not None:
                steps = min(len(actual), len(predicted))
                rmse = float(np.sqrt(np.mean((actual[:steps] - predicted[:steps]) ** 2)))
            experiment = {
                "id": f"E-{len(self.experiments) + 1:04d}",
                "source": f"challenge {challenge_id} ({pending['criterion']})",
                "design": pending["design"].to_dict(),
                "accepted": evidence is not None,
                "reason": None if evidence is not None else record.outcome.get("reason"),
                "dt": pending["design"].dt,
                "frame_stride": 2,
                "positions": _round(actual) if actual is not None else [],
                "masses": list(pending["design"].masses),
            }
            self.experiments.append(experiment)
            self.designs.append(pending["design"])
            self._emit("experiment", experiment)
            if evidence is not None:
                self.runs.append(states)
                refit_evidence = pairwise_evidence(self.runs)
                for hypothesis in self.manager.active():
                    self.manager.refit(hypothesis.id, refit_evidence)
            self._emit("theory", self.theory_state())
            return {
                "challenge": record.to_dict(),
                "actual_positions": _round(actual) if actual is not None else [],
                "trajectory_rmse": rmse,
                "new_counterexamples": [c.to_dict() for c in self.store.items[before:]],
                "hypothesis_status": leader.status.value,
            }

    def client_last_states(self) -> np.ndarray:
        return self.client.last_states

    # State ------------------------------------------------------------------------------
    def theory_state(self) -> dict:
        with self.lock:
            ranked = self.manager.rank(include_inactive=True, record=False)
            leader = self.manager.leader()
            return {
                "status": self.status,
                "leader": leader.id if leader else None,
                "hypotheses": [h.to_dict() for h in ranked],
                "counterexamples": [c.to_dict() for c in self.store.items],
                "ledger": self.client.ledger.to_dict(),
                "theory_state": "UNINITIALIZED" if leader is None else leader.status.value,
            }

    def summary(self) -> dict:
        with self.lock:
            leader = self.manager.leader()
            return {
                "status": self.status,
                "experiments": self.client.ledger.experiments,
                "usable_experiments": len(self.runs),
                "budget": self.budget,
                "theory_state": "UNINITIALIZED" if leader is None else leader.status.value,
                "leader": leader.to_dict() if leader else None,
                "active_hypotheses": len([h for h in self.manager.hypotheses.values() if h.status in ACTIVE]),
                "counterexamples": len(self.store.items),
                "ledger": self.client.ledger.to_dict(),
                "criteria": list(CRITERIA),
            }
