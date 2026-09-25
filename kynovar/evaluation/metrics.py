"""Scalar error summaries for trajectory predictions."""

from __future__ import annotations

import math

import numpy as np


class RunningScore:
    """Component-wise RMSE, MAE, R², and relative RMSE."""

    def __init__(self) -> None:
        self.sse = 0.0
        self.sae = 0.0
        self.count = 0
        self.true_sum = 0.0
        self.true_sq = 0.0

    def update(self, predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
        error = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        chosen = error[valid]
        actual = np.asarray(truth, dtype=np.float64)[valid]
        if chosen.size == 0:
            return
        self.sse += float(np.square(chosen).sum())
        self.sae += float(np.abs(chosen).sum())
        self.count += int(chosen.size)
        self.true_sum += float(actual.sum())
        self.true_sq += float(np.square(actual).sum())

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            raise ValueError("Cannot summarize an empty prediction set.")
        mse = self.sse / self.count
        rmse = math.sqrt(mse)
        mae = self.sae / self.count
        mean = self.true_sum / self.count
        total_variance = self.true_sq - self.count * mean * mean
        r2 = None if total_variance <= 1e-12 else 1.0 - self.sse / total_variance
        root_mean_square = math.sqrt(self.true_sq / self.count)
        relative = None if root_mean_square <= 1e-12 else rmse / root_mean_square
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "relative_rmse": relative,
            "components": self.count,
        }


def rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(error))))
