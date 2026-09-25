#!/usr/bin/env python3
"""Generate an observable trajectory dataset and universe-level splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kynovar.data.config import load_run_config  # noqa: E402
from kynovar.evaluation.benchmark import prepare_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate observable Kynovar trajectories.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "development.yaml",
    )
    parser.add_argument("--worlds", type=int, default=None)
    parser.add_argument("--experiments-per-world", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    if args.worlds is not None or args.experiments_per_world is not None or args.duration is not None or args.dt is not None or args.seed is not None:
        config = replace_config(config, args)
    store, splits, _normalizer = prepare_dataset(config)
    print(f"universes={len(store.universe_ids)}")
    print(f"train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}")
    store.close()
    return 0


def replace_config(config, args):
    from kynovar.data.config import apply_overrides

    return apply_overrides(
        config,
        worlds=args.worlds,
        experiments_per_world=args.experiments_per_world,
        duration=args.duration,
        dt=args.dt,
        dataset_seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
