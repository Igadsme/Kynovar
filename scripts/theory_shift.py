#!/usr/bin/env python3
"""Milestone 7: silent hidden-law change, detection, investigation, and revision.

Change streams: the hidden exponent switches (default 3.0 -> 1.5) after
`change_after` experiments without any notification. Control streams keep
the original law for the whole budget. The same revision loop runs on both.

Measured: false alarms (control streams and pre-change segments), detection
and delay, change-point estimate error, and whether the adopted v2 matches the
new hidden law (evaluation only).
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

from kynovar.discovery.lab import DesignRanges, ExperimentClient, Instrument  # noqa: E402
from kynovar.evaluation.law_recovery import recovery_record  # noqa: E402
from kynovar.evaluation.worlds import ChangingLaboratory, power_law  # noqa: E402
from kynovar.theory.revision import RevisionLoop  # noqa: E402

RANGES = DesignRanges(separation=(1.0, 3.5), speed=(0.0, 0.3), duration=0.3)


def run(seed, k, p_before, p_after, change_after, budget, instrument) -> dict:
    before, after = power_law(k, p_before), power_law(k, p_after)
    laboratory = ChangingLaboratory(before, after, change_after)
    client = ExperimentClient(laboratory, Instrument(**{**instrument.to_dict(), "seed": seed}))
    loop = RevisionLoop(client, RANGES, seed=seed)
    started = time.perf_counter()
    result = loop.run(budget)
    alarms = result["alarms"]
    first_version = sympy.sympify(0)
    record = {"seed": seed, "change_after": change_after, "budget": budget, "alarms": alarms, "seconds": time.perf_counter() - started}
    v1 = loop.versions[0]
    record["v1_equation"] = v1.equation
    record["v1_recovered"] = recovery_record(_expr(loop, 0), before.truth)["recovered"]
    if change_after is None:
        record["false_alarms"] = len(alarms)
        record["false_alarm_experiments"] = alarms
    else:
        pre = [a for a in alarms if a < change_after]
        post = [a for a in alarms if a >= change_after]
        record["false_alarms_before_change"] = len(pre)
        record["detected"] = bool(post)
        record["detection_delay_experiments"] = (post[0] - change_after + 1) if post else None
        adopted = [e for e in result["investigations"] if e["validated"] and e["alarm_index"] >= change_after]
        record["estimated_change_index"] = adopted[0]["estimated_change_index"] if adopted else None
        record["change_estimate_error"] = (adopted[0]["estimated_change_index"] - change_after) if adopted else None
        record["versions"] = len(loop.versions)
        record["replacement_validated"] = bool(adopted)
        if len(loop.versions) >= 2:
            last = _expr(loop, len(loop.versions) - 1)
            record["final_equation"] = loop.versions[-1].equation
            record["final_recovered_new_law"] = recovery_record(last, after.truth)["recovered"]
            record["final_structural_match_new_law"] = recovery_record(last, after.truth)["structural_match"]
        else:
            record["final_recovered_new_law"] = False
    record["versions_detail"] = result["versions"]
    record["investigations"] = result["investigations"]
    record["notebook"] = result["notebook"]
    del first_version
    return record


def _expr(loop, index):
    from kynovar.discovery.expression import parse_expression

    return parse_expression(loop.versions[index].equation, ("m1", "m2", "r"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--change-after", type=int, default=24)
    parser.add_argument("--k", type=float, default=2.5)
    parser.add_argument("--p-before", type=float, default=3.0)
    parser.add_argument("--p-after", type=float, default=1.5)
    parser.add_argument("--out", default="results/theory_shift/m7")
    args = parser.parse_args()
    instrument = Instrument(acceleration_noise=0.005, relative_acceleration_noise=0.03)
    rows = {"change": [], "control": []}
    started = time.perf_counter()
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        change = run(seed, args.k, args.p_before, args.p_after, args.change_after, args.budget, instrument)
        rows["change"].append(change)
        print(f"change seed={seed} alarms={change['alarms']} delay={change['detection_delay_experiments']} est_err={change['change_estimate_error']} final={change.get('final_equation')} recovered={change['final_recovered_new_law']}", flush=True)
        control = run(1000 + seed, args.k, args.p_before, args.p_after, None, args.budget, instrument)
        rows["control"].append(control)
        print(f"control seed={1000 + seed} alarms={control['alarms']}", flush=True)
    change, control = rows["change"], rows["control"]
    monitored_control = sum(args.budget - 12 for _ in control)
    delays = [r["detection_delay_experiments"] for r in change if r["detection_delay_experiments"] is not None]
    errors = [r["change_estimate_error"] for r in change if r["change_estimate_error"] is not None]
    summary = {
        "setting": {"k": args.k, "p_before": args.p_before, "p_after": args.p_after, "change_after": args.change_after, "budget": args.budget, "seed_offset": args.seed_offset, "instrument": instrument.to_dict(), "monitor": "one-sided CUSUM on log mean z^2, k=0.5, h=5, 6 warm-up experiments"},
        "control_streams": len(control),
        "control_streams_with_false_alarm": int(sum(r["false_alarms"] > 0 for r in control)),
        "control_false_alarms_total": int(sum(r["false_alarms"] for r in control)),
        "control_false_alarm_rate_per_monitored_experiment_upper_bound": float(sum(r["false_alarms"] for r in control) / max(monitored_control, 1)),
        "change_streams": len(change),
        "pre_change_false_alarms_total": int(sum(r["false_alarms_before_change"] for r in change)),
        "detected": int(sum(r["detected"] for r in change)),
        "detection_delay": {"mean": float(np.mean(delays)) if delays else None, "std": float(np.std(delays, ddof=1)) if len(delays) > 1 else None, "median": float(np.median(delays)) if delays else None, "n": len(delays)},
        "change_estimate_error": {"mean": float(np.mean(errors)) if errors else None, "mean_abs": float(np.mean(np.abs(errors))) if errors else None, "n": len(errors)},
        "replacement_validated": int(sum(r["replacement_validated"] for r in change)),
        "final_recovered_new_law": int(sum(r["final_recovered_new_law"] for r in change)),
        "v1_recovered_old_law": int(sum(r["v1_recovered"] for r in change + control)),
        "seconds": time.perf_counter() - started,
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "theory_shift.json").write_text(json.dumps({"summary": summary, **rows}, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
