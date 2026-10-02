#!/usr/bin/env python3
"""Sequential, quantitative M1-M8 acceptance runner."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass
class Stage:
    milestone: str
    name: str
    command: list[str]
    artifacts: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, Any] = field(default_factory=dict)
    evaluator: str | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    returncode: int | None = None
    status: str = "NOT_RUN"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _load(stage: Stage, index: int = 0) -> dict[str, Any]:
    return json.loads((ROOT / stage.artifacts[index]).read_text())


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _benchmark(stage: Stage) -> tuple[dict[str, Any], list[str]]:
    data = _load(stage)
    models = {row["model"]: row for row in data.get("models", [])}
    required = set(stage.thresholds["required_models"])
    target_name = "interaction_gnn" if "interaction_gnn" in models else "gnn"
    target, baseline = models.get(target_name), models.get("constant_velocity")
    errors = []
    if data.get("dataset") != "stable-v1":
        errors.append("benchmark dataset is not stable-v1")
    if not required.issubset(models):
        errors.append(f"missing models: {sorted(required - set(models))}")
    if not target or not baseline:
        return {"models": sorted(models)}, errors + ["target or constant-velocity baseline missing"]
    one_r2 = target["horizons"]["1"]["acceleration"]["r2"]
    long_horizon = "100" if "100" in target["horizons"] else max(target["horizons"], key=int)
    target_rmse = target["horizons"][long_horizon]["position_bounded"]["rmse"]
    baseline_rmse = baseline["horizons"][long_horizon]["position_bounded"]["rmse"]
    ratio = target_rmse / baseline_rmse
    counts = data.get("universes", {})
    if min(counts.values(), default=0) <= 0:
        errors.append("empty train/validation/test split")
    if one_r2 < stage.thresholds["min_one_step_acceleration_r2"]:
        errors.append("one-step acceleration R2 below threshold")
    if ratio > stage.thresholds["max_long_horizon_rmse_ratio"]:
        errors.append("long-horizon position RMSE ratio above threshold")
    return {
        "dataset": data.get("dataset"), "target_model": target_name,
        "split_universes": counts, "one_step_acceleration_r2": one_r2,
        "long_horizon": int(long_horizon), "long_horizon_position_rmse": target_rmse,
        "constant_velocity_position_rmse": baseline_rmse,
        "position_rmse_ratio": ratio, "models": sorted(models),
    }, errors


def _ood(stage: Stage) -> tuple[dict[str, Any], list[str]]:
    data = _load(stage)
    ratios, names, errors = [], [], []
    for scenario in data.get("scenarios", []):
        names.append(scenario["name"])
        models = {row["model"]: row for row in scenario.get("models", [])}
        target_name = "interaction_gnn" if "interaction_gnn" in models else "gnn"
        if target_name not in models or "constant_velocity" not in models:
            errors.append(f"{scenario['name']}: target/baseline missing")
            continue
        horizon = "50" if "50" in models[target_name]["horizons"] else max(models[target_name]["horizons"], key=int)
        target = models[target_name]["horizons"][horizon]["position_bounded"]["rmse"]
        baseline = models["constant_velocity"]["horizons"][horizon]["position_bounded"]["rmse"]
        if not (_finite(target) and _finite(baseline) and baseline > 0):
            errors.append(f"{scenario['name']}: non-finite OOD metric")
            continue
        ratios.append(target / baseline)
    expected = set(stage.thresholds["required_scenarios"])
    if not expected.issubset(names):
        errors.append(f"missing OOD scenarios: {sorted(expected - set(names))}")
    median = float(sorted(ratios)[len(ratios) // 2]) if ratios else math.inf
    wins = sum(r <= 1.0 for r in ratios)
    if median > stage.thresholds["max_median_rmse_ratio"]:
        errors.append("median OOD RMSE ratio above threshold")
    if wins < stage.thresholds["min_scenarios_beating_baseline"]:
        errors.append("too few OOD scenarios beat baseline")
    return {"scenarios": names, "bounded_position_rmse_ratios": ratios,
            "median_rmse_ratio": median, "scenarios_beating_constant_velocity": wins}, errors


def _m3(stage: Stage) -> tuple[dict[str, Any], list[str]]:
    data = _load(stage)
    summary, errors = data.get("summary", {}), []
    chosen = [entry.get("selected", {}) for entry in summary.values()]
    structural = min((x.get("structural_recovery_rate", 0) for x in chosen), default=0)
    recovered = min((x.get("full_recovery_rate", 0) for x in chosen), default=0)
    normalized, coefficient_errors, exponent_errors = [], [], []
    for run in data.get("runs", []):
        selected = run.get("selected") or {}
        if _finite(selected.get("test_rmse")) and _finite(run.get("test_scale")) and run["test_scale"] > 0:
            normalized.append(selected["test_rmse"] / run["test_scale"])
        if _finite(selected.get("max_coefficient_relative_error")):
            coefficient_errors.append(selected["max_coefficient_relative_error"])
        if _finite(selected.get("max_exponent_abs_error")):
            exponent_errors.append(selected["max_exponent_abs_error"])
    max_norm = max(normalized, default=math.inf)
    if data.get("seeds", 0) < stage.thresholds["min_seeds"]:
        errors.append("insufficient seed count")
    if structural < stage.thresholds["min_structural_recovery_rate"]:
        errors.append("structural recovery below threshold")
    if recovered < stage.thresholds["min_full_recovery_rate"]:
        errors.append("full recovery below threshold")
    if max_norm > stage.thresholds["max_normalized_test_rmse"]:
        errors.append("held-out normalized RMSE above threshold")
    return {
        "worlds": len(summary), "seeds_per_world": data.get("seeds"),
        "minimum_structural_recovery_rate": structural,
        "minimum_full_recovery_rate": recovered,
        "max_normalized_test_rmse": max_norm,
        "max_coefficient_relative_error": max(coefficient_errors, default=None),
        "max_exponent_absolute_error": max(exponent_errors, default=None),
    }, errors


def _m4(stage: Stage) -> tuple[dict[str, Any], list[str]]:
    data = _load(stage)
    summary, errors = data.get("summary", {}), []
    for key, minimum in (
        ("acceptance_rate", stage.thresholds["min_acceptance_rate"]),
        ("leader_structure_match_rate", stage.thresholds["min_leader_match_rate"]),
        ("early_two_plausible_rate", stage.thresholds["min_competing_rate"]),
    ):
        if summary.get(key, 0) < minimum:
            errors.append(f"{key} below threshold")
    calibration_error = max(
        (abs(float(level) - row["mean"]) for level, row in summary.get("calibration", {}).items()),
        default=math.inf,
    )
    if calibration_error > stage.thresholds["max_calibration_absolute_error"]:
        errors.append("predictive calibration outside threshold")
    if "PROVEN" in json.dumps(data):
        errors.append("forbidden PROVEN theory state present")
    metrics = {key: summary.get(key) for key in (
        "seeds", "acceptance_rate", "strict_acceptance_rate",
        "leader_structure_match_rate", "early_two_plausible_rate",
    )}
    metrics["max_calibration_absolute_error"] = calibration_error
    return metrics, errors


def _m5(stage: Stage) -> tuple[dict[str, Any], list[str]]:
    data = _load(stage)
    summary, errors = data.get("summary", {}), []
    active_means, random_means, passive_means = [], [], []
    active_success = random_success = 0
    minimum_n = math.inf
    for world, strategies in summary.items():
        if not {"active", "random", "passive"}.issubset(strategies):
            errors.append(f"{world}: comparison strategy missing")
            continue
        active, random, passive = strategies["active"], strategies["random"], strategies["passive"]
        minimum_n = min(minimum_n, active["n"], random["n"], passive["n"])
        active_means.append(active["mean_experiments_censored"])
        random_means.append(random["mean_experiments_censored"])
        passive_means.append(passive["mean_experiments_censored"])
        active_success += active["identified"]
        random_success += random["identified"]
        if active["identified"] < random["identified"]:
            errors.append(f"{world}: active recovered fewer laws than random")
    active_mean = sum(active_means) / len(active_means) if active_means else math.inf
    random_mean = sum(random_means) / len(random_means) if random_means else 0
    passive_mean = sum(passive_means) / len(passive_means) if passive_means else 0
    ratio = active_mean / random_mean if random_mean else math.inf
    total = int(minimum_n * len(active_means)) if active_means and minimum_n != math.inf else 0
    rate = active_success / total if total else 0
    if minimum_n < stage.thresholds["min_seeds_per_world"]:
        errors.append("insufficient seeds per world")
    if rate < stage.thresholds["min_active_recovery_rate"]:
        errors.append("active recovery rate below threshold")
    if ratio > stage.thresholds["max_active_to_random_experiment_ratio"]:
        errors.append("active selection lacks required efficiency gain over random")
    if active_mean > passive_mean:
        errors.append("active selection is worse than passive")
    return {
        "worlds": len(active_means), "minimum_seeds_per_world": minimum_n,
        "active_identified": active_success, "random_identified": random_success,
        "active_recovery_rate": rate, "active_mean_experiments": active_mean,
        "random_mean_experiments": random_mean,
        "passive_mean_experiments": passive_mean,
        "active_to_random_experiment_ratio": ratio,
    }, errors


def _m6(stage: Stage) -> tuple[dict[str, Any], list[str]]:
    data = _load(stage)
    challenger, errors = data.get("summary", {}).get("challenger", {}), []
    n = challenger.get("n", 0)
    material_rate = challenger.get("materially_wrong_all_contradicted", 0) / n if n else 0
    leader_rate = challenger.get("final_leader_correct_structure", 0) / n if n else 0
    if n < stage.thresholds["min_seeds"]:
        errors.append("insufficient falsification seeds")
    if material_rate < stage.thresholds["min_materially_wrong_contradiction_rate"]:
        errors.append("materially wrong theories not contradicted often enough")
    if challenger.get("true_structure_false_contradictions", 0) > stage.thresholds["max_true_structure_false_contradictions"]:
        errors.append("true structure falsely contradicted")
    if leader_rate < stage.thresholds["min_correct_leader_rate"]:
        errors.append("correct final leader rate below threshold")
    criteria = {row.get("criterion") for run in data.get("runs", [])
                for row in run.get("challenges", []) if row.get("criterion")}
    return {"seeds": n, "materially_wrong_contradiction_rate": material_rate,
            "true_structure_false_contradictions": challenger.get("true_structure_false_contradictions"),
            "correct_leader_rate": leader_rate, "criteria_observed": sorted(criteria)}, errors


def _m7(stage: Stage) -> tuple[dict[str, Any], list[str]]:
    data = _load(stage)
    summary, change, control = data.get("summary", {}), data.get("change", []), data.get("control", [])
    incorrect = sum(bool(row.get("replacement_validated")) and not bool(row.get("final_recovered_new_law")) for row in change)
    change_seeds = [row.get("seed") for row in change]
    control_seeds = [row.get("seed") for row in control]
    checks = [
        (len(change) >= stage.thresholds["min_streams"], "insufficient change streams"),
        (len(control) >= stage.thresholds["min_streams"], "insufficient control streams"),
        (min(change_seeds, default=-1) >= stage.thresholds["min_change_seed"], "final change seeds overlap historical range"),
        (min(control_seeds, default=-1) >= stage.thresholds["min_control_seed"], "final control seeds overlap historical range"),
        (summary.get("detected", 0) / max(len(change), 1) >= stage.thresholds["min_detection_rate"], "change detection rate below threshold"),
        (summary.get("replacement_validated", 0) / max(len(change), 1) >= stage.thresholds["min_adoption_rate"], "theory adoption rate below threshold"),
        (summary.get("final_recovered_new_law", 0) / max(len(change), 1) >= stage.thresholds["min_revised_recovery_rate"], "revised-law recovery below threshold"),
        (summary.get("control_false_alarm_rate_per_monitored_experiment_upper_bound", math.inf) <= stage.thresholds["max_control_false_alarm_rate"], "control false-alarm rate above threshold"),
        (summary.get("detection_delay", {}).get("mean", math.inf) <= stage.thresholds["max_mean_detection_delay"], "mean detection delay above threshold"),
        (summary.get("change_estimate_error", {}).get("mean_abs", math.inf) <= stage.thresholds["max_mean_abs_change_point_error"], "change-point error above threshold"),
        (incorrect <= stage.thresholds["max_incorrect_replacements"], "incorrect replacement count above threshold"),
    ]
    errors = [message for passed, message in checks if not passed]
    return {
        "change_seed_range": [min(change_seeds, default=None), max(change_seeds, default=None)],
        "control_seed_range": [min(control_seeds, default=None), max(control_seeds, default=None)],
        "change_streams": len(change), "control_streams": len(control),
        "detected": summary.get("detected"),
        "mean_detection_delay": summary.get("detection_delay", {}).get("mean"),
        "control_false_alarm_rate": summary.get("control_false_alarm_rate_per_monitored_experiment_upper_bound"),
        "mean_abs_change_point_error": summary.get("change_estimate_error", {}).get("mean_abs"),
        "replacement_validated": summary.get("replacement_validated"),
        "final_recovered_new_law": summary.get("final_recovered_new_law"),
        "incorrect_replacements": incorrect,
    }, errors


def _m8(stage: Stage) -> tuple[dict[str, Any], list[str]]:
    data = _load(stage)
    summary, errors = data.get("summary", {}), []
    if summary.get("status") != "CONVERGED":
        errors.append("interactive run did not converge")
    if summary.get("theory_state") != "SUPPORTED":
        errors.append("interactive theory not supported")
    if not summary.get("leader"):
        errors.append("interactive leader missing")
    if not data.get("experiments") or not data.get("notebook"):
        errors.append("interactive experiments/notebook missing")
    criteria = summary.get("criteria", [])
    if not {"uncertainty", "disagreement", "extrapolation", "weakly_observed"}.issubset(criteria):
        errors.append("challenger criteria missing")
    return {"status": summary.get("status"), "theory_state": summary.get("theory_state"),
            "experiments": summary.get("experiments"), "usable_experiments": summary.get("usable_experiments"),
            "criteria": criteria, "evaluation_present": bool(data.get("metrics"))}, errors


EVALUATORS: dict[str, Callable[[Stage], tuple[dict[str, Any], list[str]]]] = {
    "benchmark": _benchmark, "ood": _ood, "m3": _m3, "m4": _m4,
    "m5": _m5, "m6": _m6, "m7": _m7, "m8": _m8,
}


def run(stage: Stage, commit: str, reuse_artifacts: bool = False) -> Stage:
    started = time.perf_counter()
    try:
        reused = reuse_artifacts and bool(stage.artifacts)
        if reused:
            stage.config["execution"] = "validated_existing_artifacts"
        else:
            result = subprocess.run(stage.command, cwd=ROOT, check=False)
            stage.returncode = result.returncode
        missing = [path for path in stage.artifacts
                   if not (ROOT / path).is_file() or (ROOT / path).stat().st_size == 0]
        if (stage.returncode not in (None, 0)) or missing:
            stage.status = "FAIL"
            stage.error = f"exit={stage.returncode}; missing/empty artifacts={missing}"
        elif stage.evaluator:
            stage.metrics, failures = EVALUATORS[stage.evaluator](stage)
            stage.status = "FAIL" if failures else "PASS"
            stage.error = "; ".join(failures) if failures else None
        else:
            stage.status = "PASS"
    except Exception as exc:
        stage.status = "FAIL"
        stage.error = f"{type(exc).__name__}: {exc}"
    stage.seconds = time.perf_counter() - started
    stage.config = {"git_commit": commit, **stage.config}
    return stage


def stages(profile: str) -> list[Stage]:
    base = [
        Stage("M1", "Python/API test suite", [PYTHON, "-m", "pytest"], [],
              config={"scope": "simulator, information boundary, discovery, theory, API, WebSocket, 2D/3D"}),
        Stage("M8", "Frontend clean install, typecheck, and production build",
              ["sh", "scripts/frontend_verify.sh"], [],
              config={"runtime": "docker-node22" if os.environ.get("KYNOVAR_FRONTEND_DOCKER") == "1" else "local-node"}),
    ]
    if profile == "smoke":
        return base
    return base + [
        Stage("M2", "stable-v1 dynamics benchmark",
              [PYTHON, "scripts/run_benchmark.py", "--config", "configs/experiments/stable-v1.yaml"],
              ["results/benchmarks/stable-v1/benchmark.json", "results/benchmarks/stable-v1/benchmark.md"],
              config={"file": "configs/experiments/stable-v1.yaml"}, seeds={"dataset": 101, "split": 101, "train": 101},
              evaluator="benchmark", thresholds={"required_models": ["constant_velocity", "linear", "mlp", "gru", "gnn", "interaction_gnn_single", "interaction_gnn"], "min_one_step_acceleration_r2": 0.50, "max_long_horizon_rmse_ratio": 0.80}),
        Stage("M2", "stable-v1 OOD evaluation",
              [PYTHON, "scripts/run_ood.py", "--config", "configs/experiments/stable-v1.yaml"],
              ["results/ood/stable-v1/ood.json", "results/ood/stable-v1/ood.md"],
              config={"file": "configs/experiments/stable-v1.yaml", "OOD": True}, seeds={"base": 101},
              evaluator="ood", thresholds={"required_scenarios": ["exponent_near", "exponent_moderate", "coefficient", "mass", "velocity", "body_count", "long_horizon", "challenging_regime", "extreme_regime"], "max_median_rmse_ratio": 1.0, "min_scenarios_beating_baseline": 5}),
        Stage("M3", "law recovery final",
              [PYTHON, "scripts/discover_laws.py", "--seeds", "5", "--seed-offset", "200", "--pairwise-experiments", "36", "--out", "results/discovery/m3-final"],
              ["results/discovery/m3-final/law_recovery.json", "results/discovery/m3-final/law_recovery.md"],
              config={"pairwise_experiment_attempts": 36}, seeds={"range": [200, 204]}, evaluator="m3",
              thresholds={"min_seeds": 5, "min_structural_recovery_rate": 0.80, "min_full_recovery_rate": 0.80, "max_normalized_test_rmse": 0.10}),
        Stage("M4", "theory competition", [PYTHON, "scripts/theory_competition.py", "--seeds", "5"],
              ["results/theory/m4/theory_competition.json"], seeds={"range": [0, 4]}, evaluator="m4",
              thresholds={"min_acceptance_rate": 0.80, "min_leader_match_rate": 1.0, "min_competing_rate": 1.0, "max_calibration_absolute_error": 0.15}),
        Stage("M5", "active vs random/passive selection",
              [PYTHON, "scripts/active_experiments.py", "--seeds", "10", "--out", "results/planning/m5-final"],
              ["results/planning/m5-final/active_experiments.json", "results/planning/m5-final/active_experiments.md"],
              seeds={"range": [0, 9]}, evaluator="m5",
              thresholds={"min_seeds_per_world": 10, "min_active_recovery_rate": 0.80, "max_active_to_random_experiment_ratio": 0.95}),
        Stage("M6", "falsification", [PYTHON, "scripts/falsification.py", "--seeds", "8"],
              ["results/falsification/m6/falsification.json"], seeds={"range": [0, 7]}, evaluator="m6",
              thresholds={"min_seeds": 8, "min_materially_wrong_contradiction_rate": 0.75, "max_true_structure_false_contradictions": 0, "min_correct_leader_rate": 0.875}),
        Stage("M7", "theory revision fresh seeds",
              [PYTHON, "scripts/theory_shift.py", "--seeds", "20", "--seed-offset", "2000", "--out", "results/theory_shift/m7-final"],
              ["results/theory_shift/m7-final/theory_shift.json"], seeds={"change": [2000, 2019], "control": [3000, 3019]}, evaluator="m7",
              thresholds={"min_streams": 20, "min_change_seed": 2000, "min_control_seed": 3000, "min_detection_rate": 0.95, "min_adoption_rate": 0.95, "min_revised_recovery_rate": 0.95, "max_control_false_alarm_rate": 0.01, "max_mean_detection_delay": 5.0, "max_mean_abs_change_point_error": 2.0, "max_incorrect_replacements": 0}),
        Stage("M8", "interactive laboratory evidence",
              [PYTHON, "-c", "print('validate existing M8 evidence')"],
              ["results/interactive/m8/e2e_run.json"], evaluator="m8",
              thresholds={"status": "CONVERGED", "theory_state": "SUPPORTED"}),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="validate existing scientific artifacts while still running tests and frontend checks",
    )
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    rows: list[Stage] = []
    for stage in stages(args.profile):
        print(f"[{stage.milestone}] {stage.name}", flush=True)
        rows.append(run(stage, commit, reuse_artifacts=args.reuse_artifacts))
        suffix = f": {rows[-1].error}" if rows[-1].error else ""
        print(f"  {rows[-1].status} ({rows[-1].seconds:.1f}s){suffix}", flush=True)
        if rows[-1].status == "FAIL":
            break
    milestone_status = {}
    for milestone in (f"M{i}" for i in range(1, 9)):
        selected = [row for row in rows if row.milestone == milestone]
        milestone_status[milestone] = (
            "PASS" if selected and all(row.status == "PASS" for row in selected)
            else "FAIL" if any(row.status == "FAIL" for row in selected)
            else "NOT_RUN"
        )
    complete = args.profile == "full" and all(value == "PASS" for value in milestone_status.values())
    smoke_pass = args.profile == "smoke" and all(row.status == "PASS" for row in rows)
    payload = {
        "schema_version": 2, "profile": args.profile,
        "reuse_artifacts": args.reuse_artifacts,
        "status": "PASS" if complete or smoke_pass else "FAIL",
        "git_commit": commit, "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "milestones": milestone_status, "stages": [asdict(row) for row in rows],
    }
    out = ROOT / "results" / "acceptance"
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / f"{args.profile}.json"
    ledger.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "milestones": milestone_status, "ledger": str(ledger)}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
