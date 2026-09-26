"""Scientist-side forward simulation of a two-body design under a hypothesized law.

The hypothesis gives the signed central force F(m1, m2, r[, vr]) (positive
attracts). Integration is semi-implicit Euler with the design's timestep,
matching the laboratory's documented integrator. No hidden quantity is used.
"""

from __future__ import annotations

import numpy as np

from kynovar.discovery.lab import ExperimentDesign


def _force(model, m_self, m_other, r, vr):
    data = {"m1": np.array([m_self]), "m2": np.array([m_other]), "r": np.array([r]), "vr": np.array([vr])}
    variables = model.law.variables
    value = model.law.predict({name: data[name] for name in variables})
    return float(value[0])


def simulate_pair(model, design: ExperimentDesign, stride: int = 5) -> dict | None:
    """Predicted path, or None if the hypothesis produces invalid values.

    Returns sampled evidence rows (both bodies' perspectives) along the path
    plus the predicted maximum acceleration, minimum separation, and maximum
    position magnitude.
    """
    m = np.asarray(design.masses, dtype=np.float64)
    x = np.asarray(design.positions, dtype=np.float64).copy()
    v = np.asarray(design.velocities, dtype=np.float64).copy()
    steps = int(round(design.duration / design.dt))
    rows = {"m1": [], "m2": [], "r": [], "vr": []}
    positions = []
    max_acc, min_sep, max_pos = 0.0, float("inf"), float(np.abs(x).max())
    for step in range(steps + 1):
        delta = x[1] - x[0]
        r = float(np.linalg.norm(delta))
        if not np.isfinite(r) or r < 1e-9:
            return None
        unit = delta / r
        vr0 = float(v[0] @ unit)
        vr1 = float(v[1] @ -unit)
        f0 = _force(model, m[0], m[1], r, vr0)
        f1 = _force(model, m[1], m[0], r, vr1)
        if not (np.isfinite(f0) and np.isfinite(f1)):
            return None
        a = np.stack([f0 / m[0] * unit, -f1 / m[1] * unit])
        max_acc = max(max_acc, float(np.linalg.norm(a, axis=1).max()))
        min_sep = min(min_sep, r)
        max_pos = max(max_pos, float(np.abs(x).max()))
        if step % stride == 0:
            positions.append(x.copy())
            for mi, mj, vr in ((m[0], m[1], vr0), (m[1], m[0], vr1)):
                rows["m1"].append(mi)
                rows["m2"].append(mj)
                rows["r"].append(r)
                rows["vr"].append(vr)
        if max_acc > 1e4 or not np.isfinite(x).all():
            break
        v = v + a * design.dt
        x = x + v * design.dt
    return {
        "data": {key: np.asarray(value) for key, value in rows.items()},
        "positions": np.asarray(positions),
        "stride": stride,
        "max_acceleration": max_acc,
        "min_separation": min_sep,
        "max_position": max_pos,
    }
