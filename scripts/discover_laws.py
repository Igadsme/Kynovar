"""Milestone 3 law-discovery experiment across hidden-law families.

For every world in the evaluation catalog and every seed:
  * discovery runs through a sealed laboratory (observations only),
  * each method's best law is scored against the hidden truth,
  * prediction RMSE is measured on fresh in-range experiments and on
    extrapolation experiments outside the training ranges.

Writes results/discovery/m3/law_recovery.json and law_recovery.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from kynovar.discovery.engine import EngineConfig
from kynovar.discovery.evidence import pairwise_evidence, single_body_evidence
from kynovar.discovery.lab import DesignRanges, ExperimentClient, collect, random_single, random_two_body
from kynovar.discovery.pipeline import discover
from kynovar.discovery.symbolic import RegressorConfig
from kynovar.evaluation.law_recovery import recovery_record, rmse
from kynovar.evaluation.worlds import catalog

ROOT = Path(__file__).resolve().parents[1]
TRAIN = DesignRanges()
EXTRAPOLATE = DesignRanges(mass=(2.0, 3.0), separation=(2.5, 4.0), speed=(0.4, 0.8), single_speed=(3.0, 5.0))


def test_evidence(world, ranges: DesignRanges, seed: int, experiments: int = 12):
    client = ExperimentClient(world.laboratory("K-EVAL"))
    rng = np.random.default_rng(seed)
    if world.evidence == "single":
        kept, _ = collect(client, [random_single(rng, ranges) for _ in range(experiments)])
        return single_body_evidence(kept) if kept else None
    kept, _ = collect(client, [random_two_body(rng, ranges) for _ in range(experiments)])
    return pairwise_evidence(kept, include_velocity=True) if kept else None


def run_world(world, seed: int, engine_config: EngineConfig) -> dict:
    client = ExperimentClient(world.laboratory())
    report = discover(client, ranges=TRAIN, engine_config=engine_config, seed=seed)
    laws = report.pairwise_laws if world.evidence == "pairwise" else report.single_laws
    interpolation = test_evidence(world, TRAIN, seed + 10_000)
    extrapolation = test_evidence(world, EXTRAPOLATE, seed + 20_000)
    methods = {}
    for law in laws:
        if law.method in methods:
            continue
        record = recovery_record(law.expression, world.truth)
        methods[law.method] = {
            **law.to_dict(),
            **{key: record[key] for key in ("structural_match", "recovered", "max_coefficient_relative_error", "max_exponent_abs_error")},
            "parameter_terms": record["terms"],
            "test_rmse": rmse(law, interpolation) if interpolation is not None else None,
            "extrapolation_rmse": rmse(law, extrapolation) if extrapolation is not None else None,
        }
    selected = laws[0] if laws else None
    return {
        "world": world.name,
        "family": world.family,
        "seed": seed,
        "evidence": world.evidence,
        "local_force_detected": report.local_force_detected,
        "selected_method": selected.method if selected else None,
        "selected": methods.get(selected.method) if selected else None,
        "methods": methods,
        "ledger": report.ledger,
        "rejected_experiments": len(report.rejected),
        "discovery_seconds": report.seconds,
        "test_scale": float(np.sqrt(np.mean(interpolation.target**2))) if interpolation is not None else None,
        "extrapolation_scale": float(np.sqrt(np.mean(extrapolation.target**2))) if extrapolation is not None else None,
        "notes": report.notes,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for world in sorted({row["world"] for row in rows}):
        chosen = [row for row in rows if row["world"] == world]
        entry = {"family": chosen[0]["family"], "seeds": len(chosen)}
        methods = sorted({m for row in chosen for m in row["methods"]})
        for method in ["selected", *methods]:
            records = [row["selected"] if method == "selected" else row["methods"].get(method) for row in chosen]
            records = [record for record in records if record]
            if not records:
                continue
            entry[method] = {
                "structural_recovery_rate": float(np.mean([r["structural_match"] for r in records])),
                "full_recovery_rate": float(np.mean([r["recovered"] for r in records])),
                "n": len(records),
                "median_test_rmse": float(np.median([r["test_rmse"] for r in records if r["test_rmse"] is not None])),
                "median_extrapolation_rmse": float(np.median([r["extrapolation_rmse"] for r in records if r["extrapolation_rmse"] is not None])),
                "median_complexity": float(np.median([r["complexity"] for r in records])),
                "median_seconds": float(np.median([r["seconds"] for r in records])),
            }
        summary[world] = entry
    return summary


def markdown(payload: dict) -> str:
    lines = [
        "# Milestone 3 law recovery",
        "",
        f"Seeds per world: {payload['seeds']}. Generations: {payload['engine']['generations']}, population {payload['engine']['population']}, restarts {payload['engine']['restarts']}.",
        "Recovered means identical term structure, every coefficient within 10% relative error, and every exponent within 0.1 absolute error.",
        "RMSE is in force units on fresh experiments; extrapolation uses larger masses, separations, and speeds than training.",
        "",
        "| World | Method | Structural | Recovered | Test RMSE | Extrap. RMSE | Complexity | Seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for world, entry in payload["summary"].items():
        for method, stats in entry.items():
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"| {world} | {method} | {stats['structural_recovery_rate']:.2f} | {stats['full_recovery_rate']:.2f} | {stats['median_test_rmse']:.3g} | {stats['median_extrapolation_rmse']:.3g} | {stats['median_complexity']:.0f} | {stats['median_seconds']:.1f} |"
            )
    lines += ["", "## Selected law per run", ""]
    for row in payload["runs"]:
        selected = row["selected"]
        text = selected["expression"] if selected else "(none)"
        lines.append(f"- {row['world']} seed {row['seed']}: `{text}` via {row['selected_method']} (recovered={selected['recovered'] if selected else False})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--worlds", nargs="*", default=None)
    parser.add_argument("--out", default="results/discovery/m3")
    args = parser.parse_args()
    worlds = catalog()
    names = args.worlds or list(worlds)
    rows = []
    started = time.perf_counter()
    for name in names:
        for seed in range(args.seeds):
            config = EngineConfig(regressor=RegressorConfig(generations=args.generations, population=args.population, seed=seed), restarts=args.restarts, seed=seed)
            row = run_world(worlds[name], seed, config)
            rows.append(row)
            selected = row["selected"]
            print(f"world={name} seed={seed} method={row['selected_method']} law={selected['expression'] if selected else None} recovered={selected['recovered'] if selected else False} seconds={row['discovery_seconds']:.1f}", flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    payload = {
        "seeds": args.seeds,
        "engine": {"generations": args.generations, "population": args.population, "restarts": args.restarts},
        "train_ranges": TRAIN.__dict__,
        "extrapolation_ranges": EXTRAPOLATE.__dict__,
        "summary": summarize(rows),
        "runs": rows,
        "seconds": time.perf_counter() - started,
        "reproducibility": {"commit": commit, "python": platform.python_version(), "machine": platform.machine(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }
    (out / "law_recovery.json").write_text(json.dumps(payload, indent=2, default=str))
    (out / "law_recovery.md").write_text(markdown(payload))
    print(f"wrote {out} in {payload['seconds']:.0f}s")


if __name__ == "__main__":
    main()
