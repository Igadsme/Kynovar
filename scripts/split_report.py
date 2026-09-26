#!/usr/bin/env python3
"""Evaluation-only split distribution and stability report.

Observable distributions (body count, kinetic energy, force m|a|, |a|) and
trajectory stability statistics are computed per split. Hidden k and p are
recovered by regenerating each universe from the operator record, and appear
only in this evaluation report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kynovar.data.generate import mix_seed  # noqa: E402
from kynovar.evaluation.ground_truth import ground_truth_record  # noqa: E402
from kynovar.evaluation.regimes import trajectory_statistics  # noqa: E402
from kynovar.simulator.universe import generate_universe, load_power_law_config  # noqa: E402


def operator_config(run: dict):
    base = load_power_law_config()
    return replace(
        base,
        k_range=tuple(run["coefficient_range"]),
        p_range=tuple(run["exponent_range"]),
        count_range=tuple(run["count_range"]),
        mass_range=tuple(run["mass_range"]),
        radius_range=tuple(run["radius_range"]),
        position_range=tuple(run["position_range"]),
        velocity_range=tuple(run["velocity_range"]),
        minimum_center_distance=float(run["minimum_center_distance"]),
    )


def hidden_law(run: dict, config, index: int, replacements: int) -> tuple[str, float, float]:
    parts = (run["dataset_seed"], index, 1) if replacements == 0 else (run["dataset_seed"], index, 1, replacements)
    world = generate_universe(mix_seed(*parts), difficulty=run["difficulty"], config=config)
    law = ground_truth_record(world)["forces"][0]["parameters"]
    return world.universe_id, float(law["k"]), float(law["p"])


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def dataset_report(name: str) -> dict:
    root = ROOT / "datasets" / "generated" / name
    manifest = json.loads((root / "manifest.json").read_text())
    run = json.loads((root / "operator" / "run.json").read_text())
    splits = json.loads((ROOT / "datasets" / "splits" / name / "splits.json").read_text())
    config = operator_config(run)
    membership = {uid: split for split, ids in splits.items() if isinstance(ids, list) for uid in ids}
    rows: dict[str, dict[str, list]] = {}
    mismatched = 0
    for entry in manifest["universes"]:
        index = int(Path(entry["file"]).stem)
        uid, k, p = hidden_law(run, config, index, int(entry.get("replacements", 0)))
        if uid != entry["universe_id"]:
            mismatched += 1
            k = p = float("nan")
        split = membership.get(entry["universe_id"], "unassigned")
        bucket = rows.setdefault(split, {key: [] for key in ("k", "p", "bodies", "kinetic", "force", "acceleration", "max_acceleration", "min_distance", "near_singular", "overlap", "outside", "attempts")})
        bucket["k"].append(k)
        bucket["p"].append(p)
        archive = np.load(root / entry["file"])
        for number, experiment in enumerate(entry["experiments"]):
            states = archive[f"states_{number:04d}"].astype(np.float64)
            velocity, acceleration, mass = states[:, :, 2:4], states[:, :, 4:6], states[:, :, 6]
            stats = trajectory_statistics(states)
            bucket["bodies"].append(states.shape[1])
            bucket["kinetic"].append(float((0.5 * mass * (velocity**2).sum(-1)).sum(-1).mean()))
            bucket["force"].append(float(np.median(mass * np.linalg.norm(acceleration, axis=-1))))
            bucket["acceleration"].append(float(np.median(np.linalg.norm(acceleration, axis=-1))))
            bucket["max_acceleration"].append(stats["max_acceleration"])
            bucket["min_distance"].append(stats["min_distance"])
            bucket["near_singular"].append(stats["near_singular_frames"] > 0)
            bucket["overlap"].append(stats["overlap_frames"] > 0)
            bucket["outside"].append(stats["fraction_outside"])
            bucket["attempts"].append(experiment.get("attempts", 1))
    overlap = {
        f"{a}/{b}": len(set(splits[a]) & set(splits[b]))
        for a in ("train", "validation", "test")
        for b in ("train", "validation", "test")
        if a < b
    }
    report = {"dataset": name, "universe_id_mismatches": mismatched, "split_overlap": overlap, "splits": {}}
    for split, bucket in rows.items():
        report["splits"][split] = {
            "universes": len(bucket["k"]),
            "trajectories": len(bucket["bodies"]),
            "hidden_k_evaluation_only": describe(bucket["k"]),
            "hidden_p_evaluation_only": describe(bucket["p"]),
            "body_count": {str(n): int(np.sum(np.asarray(bucket["bodies"]) == n)) for n in sorted(set(bucket["bodies"]))},
            "mean_kinetic_energy": describe(bucket["kinetic"]),
            "median_force": describe(bucket["force"]),
            "median_acceleration": describe(bucket["acceleration"]),
            "max_acceleration": describe(bucket["max_acceleration"]),
            "min_distance": describe(bucket["min_distance"]),
            "near_singular_fraction": float(np.mean(bucket["near_singular"])),
            "overlap_fraction": float(np.mean(bucket["overlap"])),
            "mean_fraction_outside_operating_box": float(np.mean(bucket["outside"])),
            "mean_attempts": float(np.mean(bucket["attempts"])),
        }
    report["_raw"] = rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["stable-v1", "development"])
    args = parser.parse_args()
    out = ROOT / "results" / "diagnostics" / "stable-v1"
    plots = ROOT / "results" / "plots" / "stable-v1"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    reports = [dataset_report(name) for name in args.datasets]
    stable = next(report for report in reports if report["dataset"] == args.datasets[0])
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, key, label, log in zip(
        axes.ravel(),
        ("k", "p", "kinetic", "force", "acceleration", "max_acceleration"),
        ("hidden k (evaluation only)", "hidden p (evaluation only)", "mean kinetic energy", "median force m|a|", "median |a|", "max |a|"),
        (False, False, True, True, True, True),
    ):
        for split, color in (("train", "#1f4e79"), ("validation", "#c55a11"), ("test", "#548235")):
            values = np.asarray(stable["_raw"].get(split, {}).get(key, []), dtype=float)
            values = values[np.isfinite(values) & ((values > 0) if log else True)]
            if values.size == 0:
                continue
            bins = np.geomspace(values.min(), values.max(), 25) if log and values.min() > 0 else 25
            axis.hist(values, bins=bins, histtype="step", density=True, label=split, color=color, linewidth=1.5)
        if log:
            axis.set_xscale("log")
        axis.set_title(label, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"{args.datasets[0]}: split distributions")
    figure.tight_layout()
    figure.savefig(plots / "split_distributions.png", dpi=130)
    plt.close(figure)
    for report in reports:
        report.pop("_raw")
    (out / "split_report.json").write_text(json.dumps(reports, indent=2))
    for report in reports:
        print(report["dataset"], "overlap", report["split_overlap"], "id mismatches", report["universe_id_mismatches"])
        for split, stats in report["splits"].items():
            print(
                f"  {split}: universes={stats['universes']} k_med={stats['hidden_k_evaluation_only'].get('median', float('nan')):.3f} "
                f"p_med={stats['hidden_p_evaluation_only'].get('median', float('nan')):.3f} near_singular={stats['near_singular_fraction']:.3f} "
                f"overlap={stats['overlap_fraction']:.3f} outside={stats['mean_fraction_outside_operating_box']:.3f} "
                f"max_acc_p95={stats['max_acceleration']['p95']:.3g} attempts={stats['mean_attempts']:.2f}"
            )


if __name__ == "__main__":
    main()
