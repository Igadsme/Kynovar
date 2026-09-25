#!/usr/bin/env python3
"""Simulate one hidden universe and write observations.

Ground truth is written only when --write-ground-truth is set, and only to a
separate file. The observation file is checked before it is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kynovar.laboratory import (  # noqa: E402
    Experiment,
    Laboratory,
    bodies_from_observation,
)
from kynovar.simulator.boundary import assert_public_record, trajectory_to_dict  # noqa: E402
from kynovar.simulator.universe import generate_universe  # noqa: E402
from kynovar.utils.paths import ensure_within_project, validate_startup  # noqa: E402
from kynovar.utils.reproducibility import runtime_record  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one experiment in a hidden power-law universe and write observations."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--experiment-id", default="E-0001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--write-ground-truth",
        type=Path,
        default=None,
        help="Optional evaluation file. Kept separate from the observation file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = validate_startup()
    output = ensure_within_project(args.output, paths)
    truth_path = None
    if args.write_ground_truth is not None:
        truth_path = ensure_within_project(args.write_ground_truth, paths)
        if truth_path == output:
            raise SystemExit("Observations and ground truth must be different files.")

    world = generate_universe(args.seed, difficulty=args.difficulty)
    experiment = Experiment(
        objects=bodies_from_observation(world.observe()),
        duration=args.duration,
        dt=args.dt,
        experiment_id=args.experiment_id,
    )
    trajectory = Laboratory(world).run(experiment)
    payload = trajectory_to_dict(trajectory)
    payload["run"] = runtime_record(paths.root)
    assert_public_record(payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if truth_path is not None:
        from kynovar.evaluation.ground_truth import ground_truth_record

        record = ground_truth_record(world)
        record["run"] = runtime_record(paths.root)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(f"universe_id={world.universe_id}")
    print(f"frames={len(trajectory.observations)}")
    print(f"bodies={len(trajectory.observations[0].bodies)}")
    print(f"observations={output}")
    if truth_path is not None:
        print(f"ground_truth_file={truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
