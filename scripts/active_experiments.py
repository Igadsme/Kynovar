#!/usr/bin/env python3
"""Milestone 5: does active experiment selection identify the law with fewer experiments?

For each hidden world, seed, and strategy (passive, random, grid, active),
a campaign runs a fixed budget of experiments. Every campaign starts from the
same two narrow-range experiments. After each step the evaluation harness
checks whether the scientist's leader matches the hidden law (structure,
coefficient within 10%, exponents within 0.1). "Experiments to
identification" is the first step after which the leader stays correct for
the rest of the budget; campaigns that never reach that state are censored
at budget + 1 and counted as failures.
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

from kynovar.discovery.lab import DesignRanges, ExperimentClient, Instrument, random_two_body  # noqa: E402
from kynovar.evaluation.law_recovery import recovery_record  # noqa: E402
from kynovar.evaluation.worlds import catalog  # noqa: E402
from kynovar.planning.acquisition import ACQUISITION_SEMANTICS  # noqa: E402
from kynovar.planning.campaign import run_campaign  # noqa: E402
from kynovar.planning.strategies import STRATEGIES  # noqa: E402

RANGES = DesignRanges(separation=(0.8, 3.5), speed=(0.0, 0.3), duration=0.3)
NARROW = DesignRanges(separation=(1.3, 1.7), speed=(0.0, 0.1), duration=0.3)


def identification_step(records, truth) -> tuple[int | None, list[bool]]:
    correct = [bool(r.leader_expression is not None and recovery_record(r.leader_expression, truth)["recovered"]) for r in records]
    for index in range(len(correct)):
        if all(correct[index:]):
            return index + 1, correct
    return None, correct


def run_one(world, strategy_name, seed, budget, instrument) -> dict:
    client = ExperimentClient(world.laboratory(), Instrument(**{**instrument.to_dict(), "seed": seed}))
    rng = np.random.default_rng(seed)
    initial = [random_two_body(rng, NARROW, label=f"initial-{i}") for i in range(2)]
    strategy = STRATEGIES[strategy_name](RANGES, seed=seed + 7)
    started = time.perf_counter()
    records = run_campaign(client, strategy, budget, initial, seed=seed)
    step, correct = identification_step(records, world.truth)
    final = records[-1]
    return {
        "world": world.name,
        "strategy": strategy_name,
        "seed": seed,
        "experiments_to_identification": step,
        "identified": step is not None,
        "correct_by_step": correct,
        "final_leader": final.to_dict()["leader_expression"],
        "final_leader_family": final.leader,
        "observations_at_identification": records[step - 1].observations if step else None,
        "simulation_steps_at_identification": records[step - 1].simulation_steps if step else None,
        "seconds_at_identification": records[step - 1].seconds if step else None,
        "total_seconds": time.perf_counter() - started,
        "rejected_by_strategy_feasibility": strategy.rejections,
        "laboratory_rejections": sum(1 for r in records if not r.accepted),
        "steps": [r.to_dict() for r in records],
    }


def summarize(rows, budget):
    table = {}
    for world in sorted({r["world"] for r in rows}):
        for strategy in STRATEGIES:
            chosen = [r for r in rows if r["world"] == world and r["strategy"] == strategy]
            if not chosen:
                continue
            steps = np.array([r["experiments_to_identification"] or budget + 1 for r in chosen], dtype=float)
            identified = [r for r in chosen if r["identified"]]
            table.setdefault(world, {})[strategy] = {
                "n": len(chosen),
                "identified": len(identified),
                "mean_experiments_censored": float(steps.mean()),
                "std_experiments_censored": float(steps.std(ddof=1)) if len(steps) > 1 else 0.0,
                "median_experiments_censored": float(np.median(steps)),
                "mean_observations_if_identified": float(np.mean([r["observations_at_identification"] for r in identified])) if identified else None,
                "mean_simulation_steps_if_identified": float(np.mean([r["simulation_steps_at_identification"] for r in identified])) if identified else None,
                "mean_wall_seconds_if_identified": float(np.mean([r["seconds_at_identification"] for r in identified])) if identified else None,
                "mean_total_seconds": float(np.mean([r["total_seconds"] for r in chosen])),
            }
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--worlds", nargs="*", default=["inverse_square_k2.5", "inverse_power_k1.5_p2.6"])
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    parser.add_argument("--relative-noise", type=float, default=0.10)
    parser.add_argument("--absolute-noise", type=float, default=0.02)
    parser.add_argument("--out", default="results/planning/m5")
    args = parser.parse_args()
    instrument = Instrument(acceleration_noise=args.absolute_noise, relative_acceleration_noise=args.relative_noise)
    worlds = catalog()
    rows = []
    started = time.perf_counter()
    for name in args.worlds:
        for seed in range(args.seeds):
            for strategy in args.strategies:
                row = run_one(worlds[name], strategy, seed, args.budget, instrument)
                rows.append(row)
                print(f"world={name} seed={seed} strategy={strategy} identified_at={row['experiments_to_identification']} final={row['final_leader']} seconds={row['total_seconds']:.1f}", flush=True)
    summary = summarize(rows, args.budget)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "budget": args.budget,
        "instrument": instrument.to_dict(),
        "ranges": RANGES.__dict__,
        "initial_ranges": NARROW.__dict__,
        "acquisition": ACQUISITION_SEMANTICS,
        "identification_rule": "leader structure matches, coefficient within 10%, exponents within 0.1, and stays correct to the end of the budget",
        "summary": summary,
        "runs": rows,
        "seconds": time.perf_counter() - started,
    }
    (out / "active_experiments.json").write_text(json.dumps(payload, indent=2, default=str))
    lines = ["# Milestone 5 experiment selection", "", f"Budget {args.budget} experiments, {args.seeds} seeds, censored at budget+1 when never identified.", f"Acquisition: {ACQUISITION_SEMANTICS}.", "", "| World | Strategy | Identified | Mean exps (censored) | Std | Obs. at id. | Wall s at id. |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for world, strategies in summary.items():
        for strategy, s in strategies.items():
            obs = f"{s['mean_observations_if_identified']:.0f}" if s["mean_observations_if_identified"] else "-"
            wall = f"{s['mean_wall_seconds_if_identified']:.1f}" if s["mean_wall_seconds_if_identified"] else "-"
            lines.append(f"| {world} | {strategy} | {s['identified']}/{s['n']} | {s['mean_experiments_censored']:.2f} | {s['std_experiments_censored']:.2f} | {obs} | {wall} |")
    (out / "active_experiments.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
