# Milestone 2 — Dynamics Prediction

Status: implemented. The measurements below are from the development configuration on this Apple M1. Symbolic law discovery is not implemented.

The model inputs are observable trajectories. Training and inference do not call `World.internal_state()` and do not receive `k`, `p`, or the force-law configuration. Those values stay in the simulator. An operator file, `datasets/generated/<name>/operator/run.json`, records the dataset seed and the sampling ranges so a run can resume. The dataset loader does not open that file.

## Dataset

Each universe is one npz file. Each experiment inside it is one trajectory of observable states:

```text
object channels: x, y, vx, vy, ax, ay, mass, radius
```

`datasets/generated/<name>/manifest.json` stores the public universe id, the filename, the timestep, the duration, and the frame and body counts. It does not store a realized coefficient or exponent.

Generation is resumable. A universe that already has a `completed.jsonl` entry and an npz file is left in place. A directory whose operator signature does not match the requested settings is refused.

```bash
export COPYFILE_DISABLE=1
.venv/bin/python scripts/generate_dataset.py --config configs/experiments/development.yaml
```

Optional flags `--worlds`, `--experiments-per-world`, `--duration`, `--dt`, and `--seed` override the config. The development config is the default.

Universes are assigned as wholes. Trajectories from one universe stay in one split. The development assignment, seed 42, is 8 train, 2 validation, and 2 test universes. Normalization is fit on the training universes only. The center is the median and the scale is `1.4826` times the median absolute deviation, because a raw standard deviation is dominated by ejected bodies. Validation and test trajectories do not enter that fit. The map is reversed when a model emits acceleration.

The development set that was actually generated:

| Quantity | Value |
| --- | ---: |
| Universes | 12 |
| Experiments | 48 |
| Frames per experiment | 151 |
| Recorded frames | 7248 |
| Integration steps per experiment | 150 |
| `dt` | 0.02 |
| Duration | 3.0 |
| Two-body experiments | 13 |
| Three-body experiments | 19 |
| Four-body experiments | 16 |
| Train / validation / test universes | 8 / 2 / 2 |

About half of the training body-frames leave the box `|x| ≤ 8`, `|v| ≤ 8`, `|a| ≤ 80`. Pairwise forces with a large exponent and a small separation fling bodies out of the initial box within the first second. The benchmark reports the full error and, separately, the error on targets that remain inside that box.

## Models

All learned models predict acceleration from observable state, then take one semi-implicit Euler step:

```text
v <- v + a_hat dt
x <- x + v dt
```

| Model | What it uses | Trainable parameters |
| --- | --- | ---: |
| Constant velocity | `a = 0` | 0 |
| Linear | 12 observable features, ridge regression | 26 |
| MLP | the same 12 features, two GELU layers, width 48 | 3074 |
| GRU | 8 steps of those features, hidden size 48 | 9026 |
| GNN | one graph step, hidden size 48, 2 message-passing layers | 38066 |

Node features are `x, y, vx, vy, mass, radius`. Edge features are `dx, dy, distance, dvx, dvy`. Acceleration is a target, not an input. The graph is padded per batch, so the body count is not fixed. The linear fit ignores encoded acceleration targets beyond 8 robust scales so one ejected sample does not own the normal equations.

The loss is a Huber loss on acceleration, position, and velocity after dividing by the training scale. Weights are all 1 in the development config. `gnn_rollout` adds a 4-step rollout term. Reported RMSE stays in physical units.

## Training and evaluation

```bash
export COPYFILE_DISABLE=1
.venv/bin/python scripts/train_dynamics.py --model gnn
.venv/bin/python scripts/run_benchmark.py
.venv/bin/python scripts/run_ood.py
```

`make train` and `make evaluate` call the development config. Device selection is MPS, then CUDA, then CPU. This machine selected MPS (PyTorch 2.14.0). Data loaders use `num_workers=0`.

Three configs live in `configs/experiments/`:

| Config | Role |
| --- | --- |
| `smoke.yaml` | Small CPU run used by the tests |
| `development.yaml` | The M1 run recorded here |
| `research.yaml` | Larger settings. It was not executed |

An existing checkpoint is reused unless `--retrain` is passed. The checkpoint with the lowest validation one-step position RMSE is the one that is saved.

## Benchmark

Source file: `results/benchmarks/development/benchmark.md`, generated from `benchmark.json`. Test universes: 2.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 0.19733903 | 3.7781786 | 12.033341 | 29.092691 | 85.805312 | 0 | 2.945691 |
| linear | 0.19733908 | 3.7781859 | 12.033407 | 29.093166 | 85.808507 | 26 | 1.5222036 |
| mlp | 0.19734016 | 3.7783458 | 12.034473 | 29.096994 | 85.820064 | 3074 | 24.935297 |
| gru | 0.092000153 | 1.7342241 | 5.5072771 | 13.848312 | 43.575336 | 9026 | 31.912237 |
| gnn | 0.19734457 | 3.7785236 | 12.035738 | 29.099727 | 85.803272 | 38066 | 16.444945 |

On the full test trajectories the GRU has the lowest position RMSE at every listed horizon. The linear model, the MLP, and the GNN stay within a small fraction of constant velocity. At 100 steps the GNN position RMSE is 85.803272 and constant velocity is 85.805312. That gap is much smaller than the trajectory-to-trajectory variation expected from a two-universe test split.

The same models on targets that are still inside the bounded box:

| Model | 1-step | 10-step | 25-step | 50-step | 100-step |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 0.0022515427 | 0.028905954 | 0.079003849 | 0.18700306 | 0.54751519 |
| linear | 0.002252476 | 0.02895268 | 0.079234779 | 0.18718365 | 0.55000094 |
| mlp | 0.0022464752 | 0.028505961 | 0.076151082 | 0.17978603 | 0.5155427 |
| gru | 0.0019738613 | 0.022930315 | 0.083703587 | 0.22655048 | 0.74380646 |
| gnn | 0.0022418538 | 0.029987118 | 0.095472819 | 0.29573007 | 0.80106002 |

Here the GRU is lowest at 1 and 10 steps. The MLP is lowest at 25, 50, and 100 steps. The GNN is slightly below constant velocity at 1 step (0.0022418538 versus 0.0022515427) and above it after that.

One-step bounded acceleration RMSE on the same test targets: constant velocity 5.628855634, linear 5.631189844, MLP 5.616186525, GRU 4.934654373, GNN 5.604633075. The GRU is the only model with a clearly smaller acceleration error in that box. The GNN is 0.024222559 below constant velocity.

Validation one-step position RMSE, which selected the checkpoints, did not tell the same story as the test table. Constant velocity, the MLP, and the GNN sat near 0.0645. The GRU stayed near 31.03 for all 8 epochs, because the validation universes contain ejections and the GRU hidden state extrapolates there. The saved GRU is still the test-set leader on full trajectories. With two universes in each held-out split, that disagreement is a real limitation of this run.

The GNN ran 4 epochs and stopped because validation did not improve for 3 epochs. The MLP and GRU ran all 8. `train_seconds` includes those validation passes. It does not include the later multi-horizon test evaluation.

Rollout-aware training did not help. `gnn_rollout` uses the same network with `rollout_steps=4` and a rollout loss weight of 1. It trained for 21.84304 seconds and 4 epochs. Its test position RMSE matches the one-step GNN at 1 step (0.19734419 versus 0.19734457) and is slightly worse at 100 steps (85.814916 versus 85.803272).

Plots from this run:

- `results/plots/development/training_curves.png`
- `results/plots/development/rollout_error.png`
- `results/plots/development/benchmark_bars.png`
- `results/plots/development/trajectory_overlay.png`
- `results/plots/development/ood_exponent.png`
- `results/plots/development/ood_bars.png`

## OOD

Each scenario is 4 new universes and 8 trajectories. The trained development models are evaluated and not retrained. Position RMSE:

| Scenario | What changed | Constant velocity 1-step | Best 1-step | Best 50-step |
| --- | --- | ---: | --- | --- |
| exponent | `p` in `[4, 6)` | 4388.3661 | constant velocity, linear, MLP, and GNN, tied at 4388.3661 | GRU 645407.95, still far above a usable trajectory |
| body_count | 5 or 6 bodies | 131.22648 | constant velocity and linear, 131.22648 | GRU 25630.333 |
| mass | masses in `[6, 12)` | 0.92637708 | GRU 0.49466412 | GRU 59.861506 |
| velocity | each component in `[1.2, 2.2)` | 1.2715087 | GRU 0.39704443 | GRU 87.144946 |
| long_horizon | training prior, duration 6.0 | 0.0054145651 | GNN 0.0053217467 | GRU 2.2309452 |

On the stiff exponent shift, every model fails. On the extra-body shift, the learned models do not beat constant velocity at one step. On the mass and velocity shifts, the GRU is lower than constant velocity at 1 and 50 steps; the GNN is not. On the longer draw from the training prior, the GNN has the lowest 1-step error and the GRU has the lowest 50, 100, and 150-step errors. The full table is `results/ood/development/ood.md`.

## Reproduce

```bash
cd "/Volumes/T7 Shield/KYNOVAR"
export COPYFILE_DISABLE=1
make test
.venv/bin/python scripts/run_benchmark.py --config configs/experiments/development.yaml --retrain
.venv/bin/python scripts/run_ood.py --config configs/experiments/development.yaml
```

`make test` does not need the development dataset. The development benchmark reads and writes only under this repository. Do not run `research.yaml` on the M1 as a default; it is a larger configuration for a later GPU run.

## Limitations

- The test and validation splits contain two universes each. Model order can change if those universes change.
- Full-trajectory RMSE is dominated by ejected bodies. A model that emits a large acceleration on those frames is punished far more than constant velocity, which predicts zero acceleration.
- One-step position error inside the bounded box is about `0.002`. `dt` is `0.02`, so a modest acceleration error barely moves the position. Longer rollouts are the more informative comparison, and there the GNN falls behind constant velocity inside the box.
- Horizon-100 acceleration RMSE on the full test set collapses toward zero for constant velocity because separated bodies have almost no true acceleration left. That number is not evidence that a force law was recovered.
- The GNN did not win the development benchmark.
- No symbolic regression, experiment planner, backend, or frontend is included.
