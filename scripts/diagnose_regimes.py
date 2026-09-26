#!/usr/bin/env python3
"""Evaluation-only diagnosis of which hidden laws and initial conditions go unstable."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kynovar.data.regimes import load_regimes  # noqa: E402
from kynovar.evaluation.regimes import diagnose_prior, timestep_study  # noqa: E402
from kynovar.simulator.universe import load_power_law_config  # noqa: E402
from kynovar.utils.paths import validate_startup  # noqa: E402
from kynovar.utils.reproducibility import runtime_record  # noqa: E402


def _orbit(k: float, p: float, separation: float, speed_fraction: float):
    """Two unit masses on a (possibly eccentric) mutual orbit around the origin."""
    from kynovar.simulator.state import BodyInit

    acceleration = k * 1.0 / separation**p
    speed = speed_fraction * float(np.sqrt(acceleration * separation / 2.0))
    half = separation / 2.0
    return (
        BodyInit(id=0, x=-half, y=0.0, vx=0.0, vy=-speed, mass=1.0, radius=0.05),
        BodyInit(id=1, x=half, y=0.0, vx=0.0, vy=speed, mass=1.0, radius=0.05),
    )


def unstable(row: dict) -> bool:
    return row["fraction_outside"] > 0.0 or row["min_distance"] < 0.1 or row["nonfinite"] > 0


def summarize(rows: list[dict]) -> dict:
    flags = np.array([unstable(row) for row in rows])
    return {
        "trajectories": len(rows),
        "unstable_fraction": float(flags.mean()),
        "mean_fraction_outside": float(np.mean([row["fraction_outside"] for row in rows])),
        "median_max_acceleration": float(np.median([row["max_acceleration"] for row in rows])),
        "p95_max_acceleration": float(np.percentile([row["max_acceleration"] for row in rows], 95)),
        "median_max_position": float(np.median([row["max_position"] for row in rows])),
        "near_singular_trajectories": int(sum(row["near_singular_frames"] > 0 for row in rows)),
        "overlap_trajectories": int(sum(row["overlap_frames"] > 0 for row in rows)),
        "nonfinite_values": int(sum(row["nonfinite"] for row in rows)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", type=int, default=150)
    parser.add_argument("--experiments", type=int, default=2)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    validate_startup()
    out = ROOT / "results" / "diagnostics" / "stable-v1"
    plots = ROOT / "results" / "plots" / "stable-v1"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = load_power_law_config()
    rows = diagnose_prior(base, args.universes, args.experiments, args.duration, args.dt, args.seed)
    report = {
        "record_type": "evaluation_diagnostic",
        "warning": "Contains hidden law parameters. Evaluation only.",
        "prior": "configs/simulator/power_law.yaml",
        "duration": args.duration,
        "dt": args.dt,
        "summary": summarize(rows),
        "run": runtime_record(ROOT),
    }
    k = np.array([row["k"] for row in rows])
    p = np.array([row["p"] for row in rows])
    flags = np.array([unstable(row) for row in rows], dtype=float)
    sep = np.array([row["initial_min_separation"] for row in rows])
    acc0 = np.array([row["initial_max_acceleration"] for row in rows])
    p_edges = np.linspace(0.5, 4.0, 8)
    k_edges = np.linspace(0.5, 10.0, 6)
    grid = np.full((len(k_edges) - 1, len(p_edges) - 1), np.nan)
    for a in range(len(k_edges) - 1):
        for b in range(len(p_edges) - 1):
            chosen = (k >= k_edges[a]) & (k < k_edges[a + 1]) & (p >= p_edges[b]) & (p < p_edges[b + 1])
            if chosen.any():
                grid[a, b] = flags[chosen].mean()
    report["instability_by_k_p"] = {
        "k_edges": k_edges.tolist(),
        "p_edges": p_edges.tolist(),
        "unstable_fraction": [[None if np.isnan(value) else float(value) for value in row] for row in grid],
    }
    sep_edges = np.array([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 6.0])
    report["instability_by_initial_separation"] = [
        {
            "low": float(sep_edges[i]),
            "high": float(sep_edges[i + 1]),
            "count": int(((sep >= sep_edges[i]) & (sep < sep_edges[i + 1])).sum()),
            "unstable_fraction": float(flags[(sep >= sep_edges[i]) & (sep < sep_edges[i + 1])].mean())
            if ((sep >= sep_edges[i]) & (sep < sep_edges[i + 1])).any()
            else None,
        }
        for i in range(len(sep_edges) - 1)
    ]
    acc_edges = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 1e12])
    report["instability_by_initial_acceleration"] = [
        {
            "low": float(acc_edges[i]),
            "high": float(acc_edges[i + 1]),
            "count": int(((acc0 >= acc_edges[i]) & (acc0 < acc_edges[i + 1])).sum()),
            "unstable_fraction": float(flags[(acc0 >= acc_edges[i]) & (acc0 < acc_edges[i + 1])].mean())
            if ((acc0 >= acc_edges[i]) & (acc0 < acc_edges[i + 1])).any()
            else None,
        }
        for i in range(len(acc_edges) - 1)
    ]

    figure, axis = plt.subplots(figsize=(6, 4))
    image = axis.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                        extent=[p_edges[0], p_edges[-1], k_edges[0], k_edges[-1]])
    figure.colorbar(image, label="unstable fraction")
    axis.set_xlabel("hidden p")
    axis.set_ylabel("hidden k")
    axis.set_title("Default prior: unstable trajectories")
    figure.tight_layout()
    figure.savefig(plots / "instability_k_p.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(sep, np.maximum([row["max_acceleration"] for row in rows], 1e-6), c=p, s=10, cmap="plasma")
    axis.set_yscale("log")
    axis.set_xlabel("initial minimum separation")
    axis.set_ylabel("max |a| over trajectory")
    axis.set_title("Color: hidden p")
    figure.tight_layout()
    figure.savefig(plots / "max_acceleration_vs_separation.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(acc0 + 1e-6, [row["fraction_outside"] for row in rows], s=10)
    axis.set_xscale("log")
    axis.set_xlabel("initial max |a| (observable)")
    axis.set_ylabel("fraction of body-frames outside operating box")
    figure.tight_layout()
    figure.savefig(plots / "outside_vs_initial_acceleration.png", dpi=140)
    plt.close(figure)

    regimes = load_regimes()
    regime_reports = {}
    for name, regime in regimes.items():
        regime_rows = diagnose_prior(regime.config(base), 40, 2, args.duration, args.dt, args.seed + 1)
        regime_reports[name] = {"prior_only_no_rejection": summarize(regime_rows)}
    report["regime_priors_before_rejection"] = regime_reports

    report["timestep_study"] = [
        {"case": "circular orbit k=3, p=2, r=2", **timestep_study(3.0, 2.0, _orbit(3.0, 2.0, 2.0, 1.0), 3.0)},
        {"case": "eccentric orbit k=2, p=2.5, r=2, 60% circular speed", **timestep_study(2.0, 2.5, _orbit(2.0, 2.5, 2.0, 0.6), 3.0)},
        {"case": "close pass k=9, p=3.8, r=1, 30% circular speed", **timestep_study(9.0, 3.8, _orbit(9.0, 3.8, 1.0, 0.3), 1.0)},
    ]
    report["seconds"] = time.perf_counter() - started
    (out / "regimes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["regime_priors_before_rejection"], indent=2))
    for case in report["timestep_study"]:
        print(case["case"])
        for row in case["rows"]:
            print("  ", row)
    print(f"seconds={report['seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
