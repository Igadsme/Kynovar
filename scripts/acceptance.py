#!/usr/bin/env python3
"""Reproducible M1-M8 acceptance runner.

Smoke runs cheap structural checks. Full runs the scientific evaluations that
can take hours and writes one machine-readable ledger. A stage passes only
when its command exits zero and every declared artifact exists and is nonempty.
Scientific metric interpretation remains in each milestone report; this runner
never upgrades a preliminary result merely because a process completed.
"""
from __future__ import annotations

import argparse, json, platform, subprocess, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

@dataclass
class Stage:
    milestone: str
    name: str
    command: list[str]
    artifacts: list[str]
    seconds: float = 0.0
    returncode: int | None = None
    status: str = "PENDING"
    error: str | None = None


def run(stage: Stage) -> Stage:
    started=time.perf_counter()
    try:
        result=subprocess.run(stage.command,cwd=ROOT,check=False)
        stage.returncode=result.returncode
        missing=[p for p in stage.artifacts if not (ROOT/p).is_file() or (ROOT/p).stat().st_size == 0]
        if result.returncode == 0 and not missing:
            stage.status="PASS"
        else:
            stage.status="FAIL"
            stage.error=(f"exit={result.returncode}; missing/empty artifacts={missing}" if missing else f"exit={result.returncode}")
    except Exception as exc:
        stage.status="FAIL"; stage.error=f"{type(exc).__name__}: {exc}"
    stage.seconds=time.perf_counter()-started
    return stage


def stages(profile: str) -> list[Stage]:
    base=[
      Stage("M1-M8","Python/API test suite",[PYTHON,"-m","pytest"],[]),
      Stage("M8","Frontend typecheck",["npm","run","typecheck"],[]),
      Stage("M8","Frontend production build",["npm","run","build"],[]),
    ]
    # npm commands execute from frontend through a tiny shell-free wrapper below.
    for s in base:
        if s.command[0] == "npm": s.command=["npm","--prefix",str(ROOT/"frontend"),*s.command[1:]]
    if profile == "smoke": return base
    return base + [
      Stage("M2","stable-v1 dynamics benchmark",[PYTHON,"scripts/run_benchmark.py","--config","configs/experiments/stable-v1.yaml"],["results/benchmarks/stable-v1/benchmark.json","results/benchmarks/stable-v1/benchmark.md"]),
      Stage("M2","stable-v1 OOD evaluation",[PYTHON,"scripts/run_ood.py","--config","configs/experiments/stable-v1.yaml"],["results/ood/stable-v1/ood.json","results/ood/stable-v1/ood.md"]),
      Stage("M3","law recovery rerun",[PYTHON,"scripts/discover_laws.py","--seeds","5","--out","results/discovery/m3-final"],["results/discovery/m3-final/law_recovery.json","results/discovery/m3-final/law_recovery.md"]),
      Stage("M4","theory competition",[PYTHON,"scripts/theory_competition.py"],["results/theory/m4/theory_competition.json"]),
      Stage("M5","active experiment selection",[PYTHON,"scripts/active_experiments.py","--seeds","10","--out","results/planning/m5-final"],["results/planning/m5-final/active_experiments.json","results/planning/m5-final/active_experiments.md"]),
      Stage("M6","falsification",[PYTHON,"scripts/falsification.py"],["results/falsification/m6/falsification.json"]),
      Stage("M7","theory revision fresh seeds",[PYTHON,"scripts/theory_shift.py","--seeds","20","--seed-offset","2000","--out","results/theory_shift/m7-final"],["results/theory_shift/m7-final/theory_shift.json"]),
    ]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--profile",choices=("smoke","full"),default="smoke"); a=p.parse_args()
    rows=[]
    for stage in stages(a.profile):
        print(f"[{stage.milestone}] {stage.name}",flush=True); rows.append(run(stage)); print(f"  {rows[-1].status} ({rows[-1].seconds:.1f}s)",flush=True)
        if rows[-1].status == "FAIL": break
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    payload={"profile":a.profile,"status":"PASS" if rows and all(r.status=="PASS" for r in rows) else "FAIL","git_commit":commit,"python":platform.python_version(),"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"stages":[asdict(r) for r in rows]}
    out=ROOT/"results"/"acceptance"; out.mkdir(parents=True,exist_ok=True); (out/f"{a.profile}.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"status":payload["status"],"ledger":str(out/f'{a.profile}.json')},indent=2)); return 0 if payload["status"]=="PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
