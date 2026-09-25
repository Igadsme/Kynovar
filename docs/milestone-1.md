# Milestone 1 — Universe

Status: complete. Dynamics prediction is Milestone 2 and is documented in [milestone-2.md](milestone-2.md).

## What works

The repository is the external-drive checkout. Project paths resolve under that checkout. Startup creates the data, checkpoint, result, log, and cache directories, checks that they are writable, and points this process at `<repo>/.cache` for `HF_HOME`, `TORCH_HOME`, and `XDG_CACHE_HOME`.

The 2D simulator integrates bodies with semi-implicit Euler. Force laws that have closed-form tests:

| Law | Formula used by the code |
| --- | --- |
| Constant gravity | `F = m g` |
| Pairwise power law | `F_ij = k m_i m_j / r_eff^p`, attractive for `k > 0` |
| Linear drag | `F = -c v` |
| Quadratic drag | `F = -c \|v\| v` |
| Spring | magnitude `stiffness * (r - rest_length)` on every pair |
| Collision | restitution impulse after the drift step |

`r_eff = max(r, softening)`. The shipped softening is `1e-8`.

`generate_universe(seed, difficulty)` builds a hidden world.

- Difficulty 1 samples `k` from `[0.5, 10)` and sets `p = 2`.
- Difficulty 2 samples `k` from `[0.5, 10)` and `p` from `[0.5, 4)`.
- Difficulties 0 and 3–8 raise `UnsupportedDifficulty`.

The same seed and difficulty rebuild the same bodies and the same law. The public id for seed 4 is `K-D6F83A4E`. The id does not contain the seed.

`Laboratory.run(experiment)` takes objects, duration, and `dt`. Duration must be an integer multiple of `dt` (the tolerance is `1e-6` on the step ratio, so `5.0 / 0.01` is 500 steps). The return value is a trajectory of observations. The prototype world's clock and hidden parameters stay as they were. A subclass of `Experiment` is rejected. A mapping that includes `k`, `p`, `seed`, `restitution`, or an object field such as `ax` is rejected.

`scripts/simulate_world.py` writes that trajectory as JSON. `--write-ground-truth` writes a second file. The script checks the observation document before writing it.

## Information boundary

Measurements on each body are `id`, `x`, `y`, `vx`, `vy`, `ax`, `ay`, `mass`, and `radius`. Acceleration is `F / m` from the continuous forces at the current state. It is a measurement of the motion, not a copy of `k` or `p`.

`World.internal_state()` and `kynovar.evaluation.ground_truth` add the seed, difficulty, integrator name, restitution, and force parameters. Tests reject that record if it is passed through the public serializer. The laboratory module does not call `internal_state` or import the evaluation package.

`World.__repr__` and `Laboratory.__repr__` omit parameter values. A regression test uses `k = 847261.13579` and checks that this text is absent from the world repr and from the public JSON, while it is present on the internal state.

This is a software contract for later discovery code. Callers that already hold a `World` can invoke `internal_state()`. Discovery modules are required to use `observe()` and the laboratory.

## Tests executed

Command:

```bash
export COPYFILE_DISABLE=1
.venv/bin/python -m pytest
```

Latest result on this Apple M1, Python 3.12.4, NumPy 2.5.3, PyYAML 6.0.3, pytest 9.1.1:

```text
70 passed in 0.84s
```

Collected tests:

| File | Tests |
| --- | ---: |
| `tests/test_collisions.py` | 7 |
| `tests/test_determinism.py` | 2 |
| `tests/test_forces.py` | 8 |
| `tests/test_information_boundary.py` | 6 |
| `tests/test_integrator.py` | 5 |
| `tests/test_laboratory.py` | 8 |
| `tests/test_paths.py` | 9 |
| `tests/test_reproducibility.py` | 1 |
| `tests/test_simulate_script.py` | 3 |
| `tests/test_universe.py` | 11 |
| `tests/test_world.py` | 10 |

Representative checks:

- Discrete closed form `v_n = v_0 + n a dt`, `x_n = x_0 + n dt v_0 + a dt^2 n (n + 1) / 2`.
- Gap versus `x = x_0 + v_0 t + 0.5 a t^2` equals `0.5 a dt t` and shrinks with `dt`.
- Target law `F = 4 m1 m2 / r^3` with masses 2 and 3 at separation 2 has force magnitude 3. One step from rest with `dt = 0.1` moves the first body to `x = 0.015`, `vx = 0.15`.
- Equal-mass head-on collisions: `e = 1` exchanges velocities, `e = 0.5` leaves separation speed 1, `e = 0` stops both.
- Unequal masses `(1, 3)` with approach velocities `(1, -1)` and `e = 1` end at velocities `(-2, 0)`.
- Pairwise momentum is unchanged over 400 steps at `dt = 0.005`.
- Two calls with the same seed produce equal observations. Seed 42 and seed 43 diverge.
- Seed 0, difficulty 2 is locked to `k = 6.551136029553816`, `p = 1.4442534981735462`.

## Commands

```bash
export COPYFILE_DISABLE=1
make setup
make test
.venv/bin/python scripts/simulate_world.py --seed 4 --difficulty 2 \
  --duration 0.05 --dt 0.01 \
  --output results/demo/seed4_observations.json \
  --write-ground-truth results/evaluation/seed4_ground_truth.json
```

`make generate`, `make train`, `make discover`, `make evaluate`, and `make web` exit with an error. Those milestones are not implemented.

CI is `.github/workflows/tests.yml`. It installs the package and runs pytest. It does not train a model.

## Known limitations

- 2D only. Semi-implicit Euler only.
- Softening changes the power law for `r < 1e-8`. Bodies placed on the same point produce no pairwise force.
- Generator levels 3–8 are specified in `configs/simulator/power_law.yaml` and rejected by the code. The YAML cannot mark them implemented unless the allowlist changes.
- Generated universes disable collisions so the hidden law is only the power law.
- Simultaneous collisions are resolved in body-index order.
- There is no observation noise, no container, and no 3D state.
- The public universe id is an 8-hex-digit SHA-256 prefix of the seed. Small seeds can be searched by rerunning the generator. The seed itself is stored on the evaluation record.
- The external volume is ExFAT. The virtualenv must be created with `--copies`, and AppleDouble `._*` files have to be removed or pip can fail while reading package metadata.
- PyTorch, datasets, dynamics models, symbolic regression, uncertainty, active experiments, falsification, theory revision, the API server, and the UI are not implemented.
- No equation has been discovered, and no model error has been measured.

## Next milestone

Milestone 2 — Prediction.

Build dataset generation by universe, then constant-velocity, linear, MLP, and recurrent baselines, then the dynamics graph network. Success is a held-out trajectory error that is measured from a real training run and is lower than those baselines. The controlled recovery of `F = 4 m1 m2 / r^3` is Milestone 3, after that predictor exists.
