"""Universe-level train, validation, and test assignment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def split_counts(n_universes: int, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> tuple[int, int, int]:
    """Return train, validation, and test counts.

    Every split is non-empty when there are at least three universes. Remainder
    universes go to the split with the largest fractional part.
    """
    if n_universes < 3:
        raise ValueError("At least 3 universes are required for train, validation, and test.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1, got {fractions}.")
    raw = [n_universes * fraction for fraction in fractions]
    counts = [max(1, int(math.floor(value))) for value in raw]
    while sum(counts) > n_universes:
        index = max(range(3), key=lambda item: counts[item])
        if counts[index] == 1:
            raise ValueError("Cannot keep every split non-empty for this universe count.")
        counts[index] -= 1
    remainder = n_universes - sum(counts)
    fractional = [raw[item] - math.floor(raw[item]) for item in range(3)]
    while remainder > 0:
        index = max(range(3), key=lambda item: (fractional[item], item))
        counts[index] += 1
        fractional[index] = -1.0
        remainder -= 1
    return counts[0], counts[1], counts[2]


def split_universe_ids(
    universe_ids: list[str],
    assignment_id: int,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Assign each universe id to one split. The assignment is a pure function of the ids and assignment_id."""
    if len(set(universe_ids)) != len(universe_ids):
        raise ValueError("Universe ids must be unique.")
    train_count, validation_count, test_count = split_counts(len(universe_ids), fractions)
    order = list(universe_ids)
    rng = np.random.default_rng(assignment_id)
    rng.shuffle(order)
    train = order[:train_count]
    validation = order[train_count : train_count + validation_count]
    test = order[train_count + validation_count : train_count + validation_count + test_count]
    if len(test) != test_count:
        raise RuntimeError("Split sizes did not match the requested counts.")
    return {"train": train, "validation": validation, "test": test}


def write_splits(path: Path, splits: dict[str, list[str]], assignment_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": list(splits["train"]),
        "validation": list(splits["validation"]),
        "test": list(splits["test"]),
        "assignment_id": int(assignment_id),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_splits(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train": list(payload["train"]),
        "validation": list(payload["validation"]),
        "test": list(payload["test"]),
    }


def assert_disjoint_splits(splits: dict[str, list[str]]) -> None:
    train, validation, test = set(splits["train"]), set(splits["validation"]), set(splits["test"])
    if train & validation or train & test or validation & test:
        raise AssertionError("Universe ids overlap across splits.")
    if not train or not validation or not test:
        raise AssertionError("Every split must contain at least one universe.")
