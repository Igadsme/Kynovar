#!/usr/bin/env python3
"""Evaluate trained models on shifted universes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kynovar.data.config import load_run_config  # noqa: E402
from kynovar.evaluation.benchmark import run_ood  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run out-of-distribution dynamics evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "development.yaml",
    )
    args = parser.parse_args(argv)
    payload = run_ood(load_run_config(args.config))
    for scenario in payload["scenarios"]:
        print(f"scenario={scenario['name']} universes={scenario['universes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
