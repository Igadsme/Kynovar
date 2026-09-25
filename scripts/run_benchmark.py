#!/usr/bin/env python3
"""Train every baseline and the GNN, then write measured benchmark files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kynovar.data.config import load_run_config  # noqa: E402
from kynovar.evaluation.benchmark import run_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Kynovar dynamics models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "development.yaml",
    )
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(load_run_config(args.config), retrain=args.retrain)
    print(f"device={report['device']}")
    for model in report["models"]:
        one = model["horizons"].get("1", {}).get("position", {}).get("rmse")
        print(f"model={model['model']} parameters={model['trainable_parameters']} one_step_position_rmse={one}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
