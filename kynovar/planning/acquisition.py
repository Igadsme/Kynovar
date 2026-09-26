"""Acquisition scores for choosing the next experiment.

`standardized_disagreement` is the between-hypothesis variance of predicted
force along the predicted path, divided by the mean predictive variance.
It measures how strongly the current hypotheses disagree relative to their
own stated uncertainty. It is a disagreement heuristic, not an expected
information gain, and is reported as such.
"""

from __future__ import annotations

import numpy as np

from kynovar.discovery.lab import ExperimentDesign
from kynovar.planning.forward import simulate_pair

ACQUISITION_SEMANTICS = "standardized disagreement: mean over path of Var_h[mean_h] / Mean_h[sigma_h^2]; a heuristic, not information gain"


def standardized_disagreement(models, design: ExperimentDesign, path_model=None) -> float:
    if len(models) < 2:
        return 0.0
    path = simulate_pair(path_model or models[0], design)
    if path is None or not len(path["data"]["r"]):
        return float("-inf")
    means, variances = [], []
    for model in models:
        data = {name: path["data"][name] for name in model.law.variables}
        mean, std = model.predict(data)
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            return float("-inf")
        means.append(mean)
        variances.append(std**2)
    means, variances = np.stack(means), np.stack(variances)
    return float(np.mean(means.var(axis=0) / np.maximum(variances.mean(axis=0), 1e-12)))
