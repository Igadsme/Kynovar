# Milestone 2 — Stabilization (stable-v1)

The original Milestone 2 results are preserved unchanged: `docs/milestone-2.md`, `results/benchmarks/development/`, `results/ood/development/`, `results/plots/development/`, `checkpoints/development/`. Nothing in this document modifies them.

The original run had 12 universes, 2 validation and 2 test universes, about half the body-frames outside the operating box, stiff near-singular interactions, and a GNN that behaved like constant velocity. The stabilization work below creates the `stable-v1` namespace (config `configs/experiments/stable-v1.yaml`) and the `controlled-2body` namespace.

## 2.1 Regime diagnosis

`scripts/diagnose_regimes.py` → `results/diagnostics/stable-v1/regimes.json` (marked evaluation-only because it tabulates hidden k and p). 300 trajectories were drawn from the original prior (`configs/simulator/power_law.yaml`, duration 3.0, dt 0.02). Per trajectory it measures max |position|, max speed, max acceleration, max force, minimum pairwise distance, overlap (collision) count, near-singular count, fraction of frames outside the operating box, and NaN/Inf count.

| Original prior, 300 trajectories | Value |
| --- | ---: |
| Unstable trajectories | 96.3% |
| Mean fraction of frames outside the box | 0.504 |
| Median / p95 max acceleration | 1561 / 2.0e5 |
| Median max position | 86.6 |
| Near-singular trajectories | 156 |
| Trajectories with overlapping bodies | 285 |
| Non-finite values | 0 |

Instability is almost independent of k and p within the prior (unstable fraction 0.75–1.0 in every k × p cell). It depends strongly on the initial geometry: 100% unstable for initial separations below 0.8, and 68% unstable even when the initial acceleration is below 1. The dominant cause is close encounters from overlapping or near-overlapping initial placements, not a particular exponent.

Plots: `results/plots/stable-v1/instability_k_p.png`, `max_acceleration_vs_separation.png`, `outside_vs_initial_acceleration.png`.

## 2.2 Regimes and rejection sampling

`configs/simulator/regimes.yaml`, `kynovar/data/regimes.py`.

| Regime | Prior | Acceptance on the observed trajectory |
| --- | --- | --- |
| stable (primary) | k ∈ [0.5, 4], p ∈ [1, 2.5], masses [0.5, 2], velocities ±0.3, min initial center distance 1.0 | \|x\| ≤ 6, speed ≤ 4, \|a\| ≤ 40, separation ≥ 0.3 |
| challenging (OOD) | default prior, min center distance 0.6 | \|x\| ≤ 12, speed ≤ 12, \|a\| ≤ 400, separation ≥ 0.12 |
| extreme (stress) | default prior | none; includes ejections |
| controlled_two_body | 2 bodies, one fixed hidden law | \|x\| ≤ 6, speed ≤ 4, \|a\| ≤ 40, separation ≥ 0.4 |

A rejected experiment is resampled with new initial conditions (up to 60 attempts). Rejection never edits or clips a trajectory and never reads a hidden parameter; it reads only the observed states. Without rejection, even the stable prior produces 80% unstable trajectories, so the rejection step is what makes the regime stable.

## 2.3 Integration check

Same file, `timestep_study`. Position error against a dt = 0.0005 reference over 3.0 time units:

| Case | dt | Semi-implicit Euler | RK4 |
| --- | ---: | ---: | ---: |
| Circular orbit, k=3, p=2, r=2 | 0.02 | 0.032 | 4.8e-9 |
| | 0.01 | 0.016 | 3.0e-10 |
| | 0.005 | 0.0081 | 1.9e-11 |
| Eccentric orbit, k=2, p=2.5, 60% circular speed | 0.02 | 0.52 | 0.66 |
| | 0.01 | 0.20 | 0.038 |
| | 0.005 | 0.057 | 0.0014 |
| Close pass, k=9, p=3.8, 30% circular speed | 0.02 / 0.01 / 0.005 | ~4.2e5 | ~4.1e5 |

Semi-implicit Euler is first order: its error halves with dt. It is adequate for smooth orbits at dt = 0.01 (error 0.016 over 3 time units). It is not adequate for eccentric orbits at dt = 0.02 (error 0.52), and no tested integrator or timestep resolves the close pass. That trajectory is chaotic after the encounter. stable-v1 uses dt = 0.01, and the stable regime rejects close encounters, so the close-pass case lies outside the primary data. RK4 is available in `kynovar/simulator/integrator.py` as a reference; the simulator default remains semi-implicit Euler, which the scientist-side models also assume.

## 2.4 Dataset and splits

`scripts/split_report.py` → `results/diagnostics/stable-v1/split_report.json`, plot `results/plots/stable-v1/split_distributions.png`.

80 universes × 8 experiments, duration 2.0, dt 0.01, split by universe with seed 101. Hidden k and p appear only in fields named `*_evaluation_only`.

| Split | Universes | Trajectories | 2/3/4 bodies | k mean | p mean | Median KE | p95 max accel | Near-singular | Outside box | Overlap |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 56 | 448 | 343/88/17 | 2.09 | 1.83 | 0.219 | 7.66 | 0 | 0 | 0.016 |
| validation | 12 | 96 | 79/12/5 | 2.01 | 1.70 | 0.267 | 4.41 | 0 | 0 | 0 |
| test | 12 | 96 | 65/25/6 | 1.99 | 1.68 | 0.269 | 13.6 | 0 | 0 | 0.021 |
| development train (original) | 8 | 32 | 10/10/12 | 4.46 | 2.42 | 592 | 7.1e4 | 0.47 | 0.46 | 0.88 |

Split leakage: 0 universes shared between any pair of splits, and 0 universe-ID mismatches between manifests and files. Rejection needed 4.6–5.5 attempts per accepted experiment on average.

## 2.5 GNN diagnosis

`scripts/diagnose_gnn.py` on the original development GNN → `results/diagnostics/stable-v1/gnn_development_gnn.json`, plot `results/plots/stable-v1/acceleration_scatter_development_gnn.png`.

| Quantity (development test targets, 1300 samples) | Value |
| --- | ---: |
| Actual \|a\|: median / mean / max | 3.3e-5 / 8.85 / 1870 |
| Predicted \|a\|: median / mean / max | 0.017 / 0.25 / 4.45 |
| Correlation of predicted and actual components | 0.037 |
| Acceleration RMSE / RMSE of predicting zero | 70.79 / 70.80 |
| Robust acceleration scale used by the normalizer | 0.175 |

The development GNN learned an almost-zero acceleration field. The targets are extremely heavy-tailed. Most frames are ejected bodies with near-zero acceleration, and a few near-singular frames sit 10³–10⁴ robust scales away. With a Huber loss on normalized targets, those frames contribute only linear gradients, so the fit that minimizes the loss is close to zero everywhere; its RMSE is identical to predicting zero. Gradient norms were healthy in every module (0.23–0.54), and activation magnitudes were not collapsed, so this is a data and loss problem, not a dead network.

Three changes followed:

- The stable regime removes the near-singular frames (section 2.2).
- The acceleration loss term is relative: a Huber loss on `(a_hat - a) / (|a| + s)`, where `s` is the training acceleration scale (`relative_acceleration: 1.0` in the stable-v1 and controlled-2body configs). A single large target no longer dominates the gradient, and small accelerations are not ignored.
- `interaction_gnn` (`kynovar/models/gnn/interaction.py`) uses observable relative features: displacement unit vector, log distance, squared log distance, radial relative velocity, and log masses of both bodies. Each directed edge emits a scalar along the unit vector, so the pairwise part is a central force by construction. A context encoder reads past observed frames of the same experiment to infer universe-specific strength. k and p are never provided.

## 2.6 Controlled interaction test

`configs/experiments/controlled-2body.yaml`: 30 universes sharing one hidden law, two bodies, no drag, gravity, springs, or collisions, dt 0.01, duration 2.0. Source `results/benchmarks/controlled-2body/benchmark.md`.

| Model | 1-step position RMSE | 150-step position RMSE | Acceleration R² |
| --- | ---: | ---: | ---: |
| constant velocity | 2.81e-5 | 0.179 | -0.0004 |
| linear | 2.11e-5 | 0.134 | 0.433 |
| MLP | 1.17e-5 | 0.0958 | 0.828 |
| GNN | 2.18e-6 | 0.00593 | 0.994 |
| interaction GNN (single frame) | 1.59e-6 | 0.00352 | see JSON |

On clean two-body data the GNN learns the interaction: acceleration R² = 0.994, and 150-step error 30× below constant velocity. The constant-velocity-like behavior on the development set was therefore a property of that dataset, not an inability of the architecture.

## 2.7 stable-v1 benchmark

PENDING — the benchmark and OOD evaluation were running when this section was drafted. This section will be filled from `results/benchmarks/stable-v1/benchmark.json` and `results/ood/stable-v1/ood.json` once they exist.

## Limitations

- Train-seconds recorded in the stable-v1 benchmark for models trained while other jobs were running are contention-affected and are not representative performance numbers.
- stable-v1 is dominated by two-body experiments (77% of training trajectories), a side effect of rejection sampling favoring simpler configurations.
- 1.6–2.1% of trajectories in train and test still have overlapping body radii at some frame. The acceptance rule checks center separation, not radii.
- The stable regime deliberately excludes close encounters; performance there is only measured through the challenging and extreme OOD regimes.
