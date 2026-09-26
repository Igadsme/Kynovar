#!/usr/bin/env python3
"""Milestone 6: challenge experiments against the current theory.

Per seed: early narrow-range evidence (noisy), generic hypothesis families
fitted to it, one held-out narrow experiment so every hypothesis has a
score. Then a budget of challenge experiments targets the current leader:

  * challenger: best feasible design for each criterion, criteria in turn;
  * random: a random feasible design each time (baseline).

Testing is prequential: each challenge's prediction and interval are
recorded before the experiment runs, using parameters fitted to the evidence
available at that moment; afterwards every active hypothesis is refitted on
all evidence. Evaluation (here only) checks which hypotheses have the hidden
structure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from kynovar.discovery.evidence import pairwise_evidence  # noqa: E402
from kynovar.discovery.expression import structure_key  # noqa: E402
from kynovar.discovery.lab import DesignRanges, ExperimentClient, Instrument, collect, random_two_body  # noqa: E402
from kynovar.evaluation.worlds import catalog  # noqa: E402
from kynovar.falsification.challenger import CRITERIA, Challenger, CounterexampleStore  # noqa: E402
from kynovar.planning.design_space import Feasibility, feasibility_reason  # noqa: E402
from kynovar.theory.families import PAIRWISE_FAMILIES  # noqa: E402
from kynovar.theory.hypothesis import ACTIVE  # noqa: E402
from kynovar.theory.manager import TheoryManager  # noqa: E402

RANGES = DesignRanges(separation=(0.8, 3.5), speed=(0.0, 0.3), duration=0.3)
NARROW = DesignRanges(separation=(1.3, 1.7), speed=(0.0, 0.1), duration=0.3)


def truth_deviation(hypothesis, truth) -> float:
    """Relative RMS deviation from the hidden law over the full design range (evaluation only)."""
    import sympy

    from kynovar.discovery.expression import symbol

    grid = np.meshgrid(np.linspace(0.5, 2.0, 6), np.linspace(0.5, 2.0, 6), np.linspace(0.8, 3.5, 30), indexing="ij")
    data = {"m1": grid[0].ravel(), "m2": grid[1].ravel(), "r": grid[2].ravel(), "vr": np.zeros(grid[0].size)}
    f = sympy.lambdify([symbol(n) for n in ("m1", "m2", "r", "vr")], truth, "numpy")
    true = np.asarray(f(data["m1"], data["m2"], data["r"], data["vr"]), dtype=float) * np.ones(grid[0].size)
    predicted = hypothesis.model.law.predict({n: data[n] for n in hypothesis.variables})
    return float(np.sqrt(np.mean((predicted - true) ** 2)) / np.sqrt(np.mean(true**2)))


def merge_evidence(parts):
    from kynovar.discovery.evidence import Evidence

    data = {key: np.concatenate([p.data[key] for p in parts]) for key in parts[0].data}
    groups = np.concatenate([np.full(len(p), index) for index, p in enumerate(parts)])
    return Evidence("pairwise", parts[0].variables, data, np.concatenate([p.target for p in parts]), np.concatenate([p.transverse for p in parts]), len(parts), int(sum(len(p) for p in parts)), groups)


def random_feasible(rng, models, previous):
    for _ in range(200):
        design = random_two_body(rng, RANGES)
        if feasibility_reason(design, RANGES, models, previous, Feasibility()) is None:
            return design
    return design


def run(world, seed, mode, budget, instrument, store_path) -> dict:
    client = ExperimentClient(world.laboratory(), Instrument(**{**instrument.to_dict(), "seed": seed}))
    rng = np.random.default_rng(seed)
    manager = TheoryManager(bootstrap=40, seed=seed)
    early, _ = collect(client, [random_two_body(rng, NARROW) for _ in range(3)])
    evidence = pairwise_evidence(early, include_velocity=True)
    for name, expression in PAIRWISE_FAMILIES.items():
        manager.propose(expression, evidence, f"family:{name}")
    holdout, _ = collect(client, [random_two_body(rng, NARROW)])
    for states in holdout:
        manager.test("holdout-0", pairwise_evidence([states], include_velocity=True))
    runs = early + holdout
    truth_key = structure_key(world.truth)
    wrong = [h.id for h in manager.hypotheses.values() if h.structure != truth_key]
    right = [h.id for h in manager.hypotheses.values() if h.structure == truth_key]
    store = CounterexampleStore(store_path)
    accumulated = [pairwise_evidence([states], include_velocity=True) for states in early + holdout]
    evidence = merge_evidence(accumulated)
    for hypothesis in manager.active():
        manager.refit(hypothesis.id, evidence)
    challenger = Challenger(RANGES, seed=seed + 11)
    designs = []
    challenges = []
    all_wrong_contradicted_at = None
    wrong_alive_after_holdout = [i for i in wrong if manager.hypotheses[i].status in ACTIVE]
    for index in range(budget):
        active = manager.active()
        if not active:
            break
        leader = manager.leader()
        others = [h for h in active if h is not leader]
        if mode == "challenger":
            options = challenger.propose(leader, others, evidence, designs)
            criterion = CRITERIA[index % len(CRITERIA)]
            if criterion not in options:
                criterion = next(iter(options)) if options else None
            if criterion is None:
                break
            design, score = options[criterion]
        else:
            criterion, score = "random", 0.0
            design = random_feasible(rng, [h.model for h in active], designs)
        designs.append(design)
        record, new_evidence = challenger.challenge(client, manager, leader, design, criterion, score, store)
        if new_evidence is not None:
            runs.append(None)
            accumulated.append(new_evidence)
            evidence = merge_evidence(accumulated)
            for hypothesis in manager.active():
                manager.refit(hypothesis.id, evidence)
        challenges.append({**record.to_dict(), "leader_equation": leader.equation, "leader_has_hidden_structure": leader.structure == truth_key})
        if all_wrong_contradicted_at is None and all(manager.hypotheses[i].status.value == "CONTRADICTED" for i in wrong):
            all_wrong_contradicted_at = index + 1
    final_leader = manager.leader()
    deviations = {h.id: truth_deviation(h, world.truth) for h in manager.hypotheses.values()}
    materially_wrong = [i for i, value in deviations.items() if value > 0.05]
    return {
        "truth_relative_deviation_final": deviations,
        "materially_wrong_final": materially_wrong,
        "materially_wrong_all_contradicted": all(manager.hypotheses[i].status.value == "CONTRADICTED" for i in materially_wrong),
        "materially_wrong_active": [i for i in materially_wrong if manager.hypotheses[i].status in ACTIVE],
        "world": world.name,
        "seed": seed,
        "mode": mode,
        "wrong_structure_hypotheses": wrong,
        "wrong_alive_after_holdout": wrong_alive_after_holdout,
        "true_structure_hypotheses": right,
        "challenges_to_contradict_all_wrong": all_wrong_contradicted_at,
        "true_structure_falsely_contradicted": any(manager.hypotheses[i].status.value == "CONTRADICTED" for i in right),
        "counterexamples": [c.to_dict() for c in store.items if c.challenge_id in {r["challenge_id"] for r in challenges}],
        "final_leader": final_leader.equation if final_leader else None,
        "final_leader_status": final_leader.status.value if final_leader else None,
        "final_leader_has_hidden_structure": bool(final_leader and final_leader.structure == truth_key),
        "statuses": {h.id: {"equation": h.equation, "status": h.status.value, "source": h.source} for h in manager.hypotheses.values()},
        "challenges": challenges,
        "notebook": manager.notebook.to_list(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--world", default="inverse_square_k2.5")
    parser.add_argument("--out", default="results/falsification/m6")
    args = parser.parse_args()
    instrument = Instrument(acceleration_noise=0.02, relative_acceleration_noise=0.10)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    store_path = out / "counterexamples.jsonl"
    world = catalog()[args.world]
    rows = []
    started = time.perf_counter()
    for seed in range(args.seeds):
        for mode in ("challenger", "random"):
            row = run(world, seed, mode, args.budget, instrument, store_path)
            rows.append(row)
            print(f"seed={seed} mode={mode} materially_wrong_active={row['materially_wrong_active']} wrong_alive_after_holdout={len(row['wrong_alive_after_holdout'])} all_wrong_contradicted_at={row['challenges_to_contradict_all_wrong']} true_falsely_contradicted={row['true_structure_falsely_contradicted']} final={row['final_leader']} ({row['final_leader_status']})", flush=True)
    summary = {}
    for mode in ("challenger", "random"):
        chosen = [r for r in rows if r["mode"] == mode]
        steps = np.array([r["challenges_to_contradict_all_wrong"] or args.budget + 1 for r in chosen], dtype=float)
        summary[mode] = {
            "n": len(chosen),
            "all_wrong_contradicted": int(sum(r["challenges_to_contradict_all_wrong"] is not None for r in chosen)),
            "mean_challenges_censored": float(steps.mean()),
            "std_challenges_censored": float(steps.std(ddof=1)) if len(steps) > 1 else 0.0,
            "true_structure_false_contradictions": int(sum(r["true_structure_falsely_contradicted"] for r in chosen)),
            "materially_wrong_all_contradicted": int(sum(r["materially_wrong_all_contradicted"] for r in chosen)),
            "materially_wrong_left_active_total": int(sum(len(r["materially_wrong_active"]) for r in chosen)),
            "final_leader_correct_structure": int(sum(r["final_leader_has_hidden_structure"] for r in chosen)),
            "counterexamples": int(sum(len(r["counterexamples"]) for r in chosen)),
        }
    payload = {"definitions": {"materially_wrong": "final fitted hypothesis deviates from the hidden law by more than 5% relative RMS over the full design range (evaluation only)", "challenges_to_contradict_all_wrong": "challenges until every hypothesis without the hidden structure is CONTRADICTED; censored at budget+1"}, "world": args.world, "instrument": instrument.to_dict(), "budget": args.budget, "summary": summary, "runs": rows, "seconds": time.perf_counter() - started}
    (out / "falsification.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
