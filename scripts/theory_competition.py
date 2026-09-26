#!/usr/bin/env python3
"""Milestone 4: competing hypotheses, evidence updates, and uncertainty calibration.

Protocol per seed (hidden world: evaluation catalog entry, noisy instrument):
  1. Early evidence from a narrow range of separations.
  2. Hypotheses from the discovery engine plus scientist-chosen alternative
     families, each fitted to the same early evidence.
  3. Held-out early experiments (same narrow range) test every hypothesis.
  4. Later experiments over a wide range test every hypothesis again.
  5. Interval coverage of the leading hypothesis on fresh experiments.

Acceptance: at step 3 at least two hypotheses of different structure remain
non-contradicted; after step 4 the hypotheses that do not match the hidden
structure are contradicted and the leader matches it. Structure matching is
done here, in evaluation, and never inside the theory code.
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
import sympy  # noqa: E402

from kynovar.discovery.engine import EngineConfig, SymbolicDiscoveryEngine  # noqa: E402
from kynovar.discovery.evidence import pairwise_evidence  # noqa: E402
from kynovar.discovery.expression import structure_key, symbol  # noqa: E402
from kynovar.discovery.lab import DesignRanges, ExperimentClient, Instrument, collect, random_two_body  # noqa: E402
from kynovar.discovery.symbolic import RegressorConfig  # noqa: E402
from kynovar.evaluation.worlds import catalog  # noqa: E402
from kynovar.theory.hypothesis import ACTIVE  # noqa: E402
from kynovar.theory.manager import TheoryManager  # noqa: E402
from kynovar.uncertainty.parametric import LEVELS, coverage  # noqa: E402

NARROW = DesignRanges(separation=(1.3, 1.7), speed=(0.0, 0.1), duration=0.3)
WIDE = DesignRanges(separation=(0.8, 3.5), speed=(0.0, 0.3), duration=1.0)
M1, M2, R = symbol("m1"), symbol("m2"), symbol("r")
ALTERNATIVES = {
    "family:exponential_screening": sympy.Float(1.0) * M1 * M2 * sympy.exp(sympy.Float(-1.0) * R),
    "family:linear_in_r": M1 * M2 * (sympy.Float(1.0) + sympy.Float(-0.5) * R),
    "family:power_law": sympy.Float(1.0) * M1 * M2 * R ** sympy.Float(-1.0),
}


def evidence_from(client, ranges, rng, count, label):
    designs = [random_two_body(rng, ranges, label=f"{label}-{i}") for i in range(count)]
    kept, rejected = collect(client, designs)
    return kept, rejected


def run(seed: int, world_name: str, instrument: Instrument) -> dict:
    world = catalog()[world_name]
    client = ExperimentClient(world.laboratory(), Instrument(**{**instrument.to_dict(), "seed": seed}))
    rng = np.random.default_rng(seed)
    manager = TheoryManager(bootstrap=40, seed=seed)
    early, _ = evidence_from(client, NARROW, rng, 8, "early")
    early_evidence = pairwise_evidence(early)
    manager.notebook.record("experiment", experiment_id="early-batch", summary=f"{len(early)} narrow-range two-body experiments, {len(early_evidence)} observations")

    engine = SymbolicDiscoveryEngine(EngineConfig(regressor=RegressorConfig(generations=15, population=120, seed=seed), restarts=1, methods=("symbolic", "power_sum"), seed=seed))
    proposals = []
    for law in engine.discover(early_evidence)[:3]:
        proposals.append((law.expression, f"engine:{law.method}"))
    proposals += [(expression, source) for source, expression in ALTERNATIVES.items()]
    for expression, source in proposals:
        try:
            manager.propose(expression, early_evidence, source)
        except (ValueError, TypeError, ZeroDivisionError) as error:
            manager.notebook.record("note", text=f"Could not fit proposal from {source}: {error}")
    manager.merge_equivalents()

    stages = {}
    for stage, ranges, count in (("early_holdout", NARROW, 4), ("wide", WIDE, 8)):
        runs, rejected = evidence_from(client, ranges, rng, count, stage)
        for index, states in enumerate(runs):
            experiment_id = f"{stage}-{index}"
            evidence = pairwise_evidence([states])
            manager.notebook.record("experiment", experiment_id=experiment_id, summary=f"{len(evidence)} observations, separations {evidence.data['r'].min():.2f}-{evidence.data['r'].max():.2f}")
            manager.test(experiment_id, evidence)
        stages[stage] = {
            "experiments": len(runs),
            "rejected": len(rejected),
            "statuses": {h.id: h.status.value for h in manager.hypotheses.values()},
            "active": [h.id for h in manager.active()],
        }
        if stage == "early_holdout":
            stages[stage]["active_structures"] = len({h.structure for h in manager.active()})

    truth_key = structure_key(world.truth)
    wrong_active_before_parsimony = [h.id for h in manager.hypotheses.values() if h.status in ACTIVE and h.structure != truth_key]
    nested = manager.resolve_nested()
    ranked = manager.rank()
    leader = ranked[0] if ranked else None

    fresh, _ = evidence_from(client, WIDE, rng, 10, "calibration")
    calibration = None
    if leader is not None and fresh:
        fresh_evidence = pairwise_evidence(fresh)
        calibration = coverage(leader.model, fresh_evidence.data, fresh_evidence.target)

    hypotheses = {h.id: {**h.to_dict(), "matches_hidden_structure_evaluation_only": h.structure == truth_key} for h in manager.hypotheses.values()}
    wrong_active = [h.id for h in manager.hypotheses.values() if h.status in ACTIVE and h.structure != truth_key]
    early_plausible = stages["early_holdout"]["active_structures"]
    return {
        "seed": seed,
        "world": world_name,
        "stages": stages,
        "leader": leader.id if leader else None,
        "leader_equation": leader.equation if leader else None,
        "leader_matches_hidden_structure": bool(leader is not None and leader.structure == truth_key),
        "early_distinct_plausible_structures": early_plausible,
        "wrong_structures_active_before_parsimony_rule": wrong_active_before_parsimony,
        "nested_resolutions": nested,
        "wrong_structures_still_active": wrong_active,
        "alternatives_contradicted": [h.id for h in manager.hypotheses.values() if h.source.startswith("family:") and h.status.value == "CONTRADICTED"],
        "acceptance": bool(early_plausible >= 2 and leader is not None and leader.structure == truth_key and any(h.status.value == "CONTRADICTED" and h.structure != truth_key and h.verdicts and h.verdicts[0].verdict != "contradicts" for h in manager.hypotheses.values())),
        "strict_acceptance_no_wrong_structure_active": bool(early_plausible >= 2 and leader is not None and leader.structure == truth_key and not wrong_active),
        "calibration_coverage": calibration,
        "hypotheses": hypotheses,
        "notebook": manager.notebook.to_list(),
        "notebook_text": manager.notebook.summary(),
        "ledger": client.ledger.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--world", default="inverse_square_k2.5")
    parser.add_argument("--relative-noise", type=float, default=0.03)
    parser.add_argument("--absolute-noise", type=float, default=0.005)
    parser.add_argument("--out", default="results/theory/m4")
    args = parser.parse_args()
    instrument = Instrument(acceleration_noise=args.absolute_noise, relative_acceleration_noise=args.relative_noise)
    started = time.perf_counter()
    runs = []
    for seed in range(args.seeds):
        result = run(seed, args.world, instrument)
        runs.append(result)
        print(
            f"seed={seed} early_structures={result['early_distinct_plausible_structures']} leader={result['leader_equation']} "
            f"matches={result['leader_matches_hidden_structure']} wrong_active={result['wrong_structures_still_active']} "
            f"acceptance={result['acceptance']} coverage={result['calibration_coverage']}",
            flush=True,
        )
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    coverage_table = {level: [r["calibration_coverage"][level] for r in runs if r["calibration_coverage"]] for level in (f"{l:.2f}" for l in LEVELS)}
    summary = {
        "world": args.world,
        "instrument": instrument.to_dict(),
        "seeds": args.seeds,
        "acceptance_rate": float(np.mean([r["acceptance"] for r in runs])),
        "acceptance_definition": "at least two distinct structures non-contradicted after early held-out tests; afterwards at least one early-plausible wrong structure is CONTRADICTED and the leader has the hidden structure",
        "strict_acceptance_rate": float(np.mean([r["strict_acceptance_no_wrong_structure_active"] for r in runs])),
        "wrong_structure_active_before_parsimony_rate": float(np.mean([bool(r["wrong_structures_active_before_parsimony_rule"]) for r in runs])),
        "leader_structure_match_rate": float(np.mean([r["leader_matches_hidden_structure"] for r in runs])),
        "early_two_plausible_rate": float(np.mean([r["early_distinct_plausible_structures"] >= 2 for r in runs])),
        "calibration": {level: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for level, v in coverage_table.items() if v},
        "seconds": time.perf_counter() - started,
    }
    (out / "theory_competition.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=2, default=str))
    (out / "notebook_seed0.txt").write_text(runs[0]["notebook_text"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
