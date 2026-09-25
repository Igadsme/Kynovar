# KYNOVAR

Autonomous discovery of physical laws.

**Milestone 1 is implemented. Later milestones are not.** There are no trained models, recovered equations, or benchmark tables yet. Numbers in this repository come from the simulator and its tests.

## What is Kynovar?

Kynovar is a research system for a specific question: can a program observe an unknown dynamical system, learn its dynamics, propose equations, choose experiments, and revise those equations when the evidence turns against them?

This milestone builds the world those later stages will study. A deterministic 2D simulator integrates bodies under explicit force laws. A universe generator hides a pairwise power law behind an observation API. A laboratory runs experiments and returns trajectories. Evaluation code is the only path that reads the hidden law.

## Demo

Generate one hidden universe and write the measurements:

```bash
.venv/bin/python scripts/simulate_world.py \
  --seed 4 \
  --difficulty 2 \
  --duration 0.05 \
  --dt 0.01 \
  --output results/demo/seed4_observations.json \
  --write-ground-truth results/evaluation/seed4_ground_truth.json
```

On this machine that command printed:

```text
universe_id=K-D6F83A4E
frames=6
bodies=4
observations=/Volumes/T7 Shield/KYNOVAR/results/demo/seed4_observations.json
ground_truth_file=/Volumes/T7 Shield/KYNOVAR/results/evaluation/seed4_ground_truth.json
```

The observation file records positions, velocities, accelerations, masses, and radii. The ground-truth file is separate and is labeled evaluation-only. For seed 4 at difficulty 2, that file contains the sampled law `pairwise_power_law` with `k = 9.459033002937492` and `p = 2.2896464348502654`. Those values were read from the evaluation export after the run. They are the hidden parameters of the generator, saved for later scoring.

## Research question

Can an AI observe an unknown dynamical system, learn its dynamics, formulate mathematical hypotheses, select experiments that reduce uncertainty, recover governing equations, validate or falsify those equations, detect a change in the physics, and revise its theory?

Milestone 1 builds the environment in which that question can be tested.

## Architecture

Implemented now:

```text
kynovar/simulator/     2D state, semi-implicit Euler, force laws, collisions
kynovar/simulator/universe.py
                       difficulty 1–2 pairwise power-law worlds
kynovar/laboratory/    Experiment and Laboratory.run
kynovar/evaluation/    ground-truth export for scoring only
kynovar/utils/         path validation and run metadata
configs/simulator/     force-law list and sampling ranges
scripts/simulate_world.py
tests/
```

Reserved directories exist and contain no scientific code: `models/`, `discovery/`, `theory/`, `uncertainty/`, `planning/`, `falsification/`, `backend/`, `frontend/`, `paper/`.

`World.observe()` returns kinematics. `World.internal_state()` also returns hidden laws and is for evaluation. The laboratory accepts initial conditions, duration, and timestep. It does not accept force parameters. Details are in [docs/milestone-1.md](docs/milestone-1.md).

## How it works

Each body has an id, position, velocity, mass, and radius. Acceleration is computed from the active force laws and reported as a measurement. The integrator is semi-implicit Euler:

```text
v <- v + a dt
x <- x + v dt
```

The integrator receives an acceleration callback so a later multistage method can evaluate intermediate states. RK4 is not implemented.

Continuous forces:

- constant gravity, `F = m g`
- pairwise power law, `F_ij = k m_i m_j / r^p`, attractive when `k > 0`
- linear drag, `F = -c v`
- quadratic drag, `F = -c |v| v`
- springs, `F = stiffness (r - rest_length)` along the pair

Collisions are impulsive. The coefficient of restitution `e` is in `[0, 1]`. Overlap is removed along the contact normal, then an impulse is applied when the bodies are approaching. Generated power-law universes leave collisions off, so those worlds have one hidden continuous law.

Difficulty 1 samples `k` and fixes `p = 2`. Difficulty 2 samples both `k` and `p`. The same seed rebuilds the same world. The public id is `K-` plus a hash prefix. It is not the seed.

## Results

No discovery results exist.

The simulator checks that do exist are unit tests against closed-form updates, including:

- free motion and the semi-implicit Euler closed form under constant acceleration
- `F = 4 m1 m2 / r^3` for `m = (2, 3)` and `r = 2`, which has magnitude 3
- one semi-implicit step of that law from rest
- drag, springs, and gravity-plus-drag superposition
- elastic, partially elastic, and inelastic collisions
- momentum conservation under the pairwise law
- identical trajectories from identical seeds

The latest local run was **70 passed in 0.84s** (Python 3.12.4, NumPy 2.5.3, pytest 9.1.1) on an Apple M1.

## Benchmarks

No model benchmarks exist. `make evaluate` exits with an error for that reason.

## Installation

The project lives on the external drive. The virtual environment is created inside the repository. This volume is ExFAT, which does not support symlinks, so the environment is built with copies.

```bash
cd "/Volumes/T7 Shield/KYNOVAR"
export COPYFILE_DISABLE=1
make setup
make test
```

`make setup` creates `.venv` with `python3 -m venv --copies`, removes AppleDouble sidecar files, and installs the package in editable mode.

Dependencies for this milestone: NumPy, PyYAML, and pytest. PyTorch is not installed. Device selection arrives with the dynamics model.

Copy `.env.example` to `.env` only if you need to override paths. Empty values use directories under the repository. Startup rejects project paths outside the repository unless `KYNOVAR_DATA_ROOT` is set, and it points `HF_HOME`, `TORCH_HOME`, and `XDG_CACHE_HOME` at `<repo>/.cache` for the process.

## Generate worlds

```bash
.venv/bin/python scripts/simulate_world.py \
  --seed 4 \
  --difficulty 2 \
  --duration 1.0 \
  --dt 0.01 \
  --output results/demo/observations.json
```

Add `--write-ground-truth results/evaluation/ground_truth.json` when an evaluation file is required. The two paths must differ.

## Train dynamics model

Not implemented. `make train` exits with an error.

## Run discovery

Not implemented. `make discover` exits with an error.

## Run evaluation

Hidden laws can be exported with `kynovar.evaluation.ground_truth.ground_truth_record` or the script flag above. There is no discovery score to evaluate.

## Web interface

Not implemented. `make web` exits with an error.

## Reproducing results

Tests:

```bash
export COPYFILE_DISABLE=1
make test
```

A seeded universe is fully determined by the seed, the difficulty, `configs/simulator/power_law.yaml`, and the NumPy PCG64 stream. Seed 0 at difficulty 2 currently samples `k = 6.551136029553816` and `p = 1.4442534981735462`. A unit test locks those values.

`scripts/simulate_world.py` attaches a run record to each file: project version, git commit when one exists, timestamp, Python, NumPy, platform, and machine. The universe seed is written only in the evaluation file.

## Paper

Not started. The `paper/` directory is a placeholder path for a later scaffold.

## Limitations

- The simulator is 2D. A 3D request raises an error.
- The only integrator is semi-implicit Euler. Constant acceleration is not exact in continuous time; the local tests check the discrete closed form and the known `O(dt)` gap.
- Pairwise forces use `r_eff = max(r, softening)` with `softening = 1e-8`. Separations below that threshold do not match the written formula. Coincident bodies contribute no force.
- Sampling intervals are half-open on the right, matching NumPy's `Generator.uniform`.
- Universe difficulties 3–8 raise `UnsupportedDifficulty`. Drag, springs, gravity, and collisions can be constructed directly in tests. The generator does not randomize them.
- Collision resolution visits pairs in index order. The result is deterministic and can depend on that order when several pairs overlap.
- Reported acceleration is the continuous-force acceleration. A collision shows up as a velocity jump.
- Space has no walls.
- The observation boundary is an API contract. A Python caller who already holds a `World` can call `internal_state()`. Discovery code must not do that. The public id is a short hash of the seed, so a caller who brute-forces small seeds and reruns the generator could recover the law. The laboratory return value does not include the seed.
- ExFAT creates `._*` AppleDouble files. `make setup` deletes them inside `.venv`. Set `COPYFILE_DISABLE=1` in the shell used for development.
- No datasets, models, symbolic regression, uncertainty estimates, experiment planner, falsifier, backend, or frontend are implemented.
