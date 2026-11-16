# Kynovar Research Protocol v1 — Interpretable ODE Discovery With Active Experiments

**Status:** design proposal / not executed. Freezing this document before the final evaluation is part of M9.
**Primary research question:** In a class of damped and nonlinear oscillators, can an agent using dimensionally constrained symbolic system identification and cost-aware active interventions recover governing differential equations more reliably or with fewer observations than fair passive/random strategies under realistic measurement noise?

## Primary scientific system

Start with controlled single pendulum:
```text
dtheta/dt = omega
domega/dt = -(g/L) sin(theta) - beta * omega + u(t)/(m*L^2)
```
Here L is measured length, beta is effective angular damping, u is external applied torque, and g is gravitational acceleration. The mathematical form above is the **evaluator/reference model**, not a hardcoded discovery oracle. Control interfaces may allow initial angle, length, and permitted drive frequency within physically realizable bounds.

Progression: simple harmonic oscillator -> damped nonlinear pendulum -> driven pendulum -> coupled/double pendulum. Do not treat derived velocities/accelerations as directly observed when raw data only contain angles/positions.

## Hypotheses to pre-register

- **H1:** cost-aware active experiments use fewer accepted experimental trials than random selection to reach a pre-defined identification/forecast threshold on held-out synthetic conditions.
- **H2:** dimensional constraints reduce invalid candidate equations and improve structural recovery at the same compute budget.
- **H3:** uncertainty-aware revision reduces change-detection delay or post-change forecast error relative to a no-revision baseline, without an unacceptable false-alarm rate.
- **H4 (external validation):** the pipeline accurately forecasts instrumentally measured pendulum data across held-out sessions; equation plausibility is separately reviewed.

All four are empirical hypotheses, **not** completed achievements. Publish null outcomes.

## Data sources and provenance

1. **Synthetic controlled:** seeded physically modeled systems with accessible evaluator-only ground truth; training/testing split by environment parameters and system family.
2. **Instrumentally measured external:** IBM Double Pendulum Chaotic Dataset https://ibm.github.io/double-pendulum-chaotic-dataset/ and/or open multi-arm pendulum dataset https://doi.org/10.5281/zenodo.6633719. Review dataset licenses and original papers; exclude video if heavy and use time-series positions initially.
3. **Own physical measurements (recommended for true interventions):** fixed camera/IMU pendulum rig, ruler-based calibration, recorded sampling frequency, multiple lengths and initial angles. Record independent trials before evaluating.
4. **Future domain transfer:** NASA PCoE measured battery aging signals; introduce a separate scientific model and evaluation protocol.

Dataset metadata must include source DOI/URL, license, exact file checksum, measurement process, original units, measurement uncertainty, consent/privacy notes if relevant, preprocessing decisions, and dataset version. Do not redistribute raw data contrary to source terms.

## Splits and concealment

- Synthetic: different seeds **and** disjoint environment parameter draws for train/dev/test; hold out at least one law-family/shift condition as OOD; freeze before final analysis.
- Real: split by complete recording/session and physical condition (e.g. pendulum length), never randomly split adjacent frames. Avoid fitting normalization on held-out data.
- The scientist is allowed observations and intervention limits only. The evaluator holds the exact generating equation, coefficients, and experiment scores. All decisions logged.
- Record failed and unusable trials, not just successful ones.

## Methods and fair baselines

- Existing Kynovar symbolic discovery + current planning heuristic.
- PySINDy and weak-form SINDy for noisy derivatives.
- PySR with physics/unit constraints.
- Least-squares fixed-form pendulum fit as a strong physics-informed baseline.
- Constant/linear dynamic baseline and neural GRU/GNN where appropriate.
- Interventions: cost-aware active, disagreement-only, random, grid/prespecified and passive observation.

Methods receive matching measurements, control rights, time budgets, data budgets and access to reference constants. If a baseline is given special structural prior knowledge, disclose it prominently. Do not compare full-information baselines and partial-information methods as if equivalent.

## Independent variables

- Measurement noise: zero, low, moderate, high; specify levels in units after pilot calibration and freeze them.
- Missingness: none, 10%, and a contiguous sensor dropout scenario.
- Observation type: clean angle, noisy angle, video-derived position, IMU time series.
- Intervention budget: 5, 10 and 20 trials (or cost-normalized equivalent).
- Distribution shift: change damping/length/drive between sessions, blinded to scientist.
- Seeds: target 30 independent synthetic seeds per feasible primary condition; real world uses distinct recorded sessions rather than pretending repeated code seeds are independent measurements.

## Primary outcomes

1. **Synthetic structural recovery:** exact or equivalence-class equation recovery, normalized coefficient error, dimensional consistency.
2. **Held-out predictive utility:** one-step RMSE and free rollout horizon RMSE on unseen initial conditions/physical settings, compared at common units and horizons.
3. **Experiment efficiency:** number and cost of trials to threshold and success rate by budget, against paired random/grid controls.
4. **Reliability:** nominal 90/95% prediction interval coverage; change-detection false alarms, delay, and recovery quality; rejected/unusable interventions.
5. **Computational cost:** CPU/GPU hours, peak memory, wall-clock latency and timeout rate.

For synthetic evaluations, define success **before** inspecting test runs. Real data usually lack exact ground truth equations; report predictive comparison and domain-expert physical plausibility, not invented equation-recovery accuracy.

## Statistics and reporting

Report median, mean where useful, confidence intervals, and per-seed outcomes. Use paired comparisons when the seed/system design is matched. Correct for multiple planned comparisons or clearly mark exploratory analyses. Keep an immutable record of all attempted configurations, including failed or timed-out methods. Do not assert an improvement from a single cherry-picked rollout.

## Strict run evidence

Every accepted scientific artifact should record:
```yaml
git_sha: <commit>
dataset:
  source: <DOI_or_URL_or_simulator_commit>
  sha256: <checksum>
  split_manifest: <path>
config_sha256: <checksum>
dependencies_lock: <path_or_hash>
seed: <number>
hardware: <CPU_GPU_RAM>
run_started_utc: <timestamp>
budget: {experiments: 10, wall_clock_seconds: 3600}
status: COMPLETED|FAILED|TIMED_OUT
metrics: <file>
traces: <file>
```

Use a new results namespace `results/research/v1/` instead of overwriting `results/acceptance/full.json`. A stored old artifact cannot satisfy new-run acceptance merely by being present. Separate smoke tests from expensive replicates; scientific results can be generated offline.

## First minimum experiment (M9/M10 pilot)

- **Task:** identify damped nonlinear pendulum `domega/dt = -a sin(theta) - b omega` from angles sampled over time, with `a,b` unknown and variable across worlds.
- **Pilot:** 3 system settings, 3 seeds, 2 noise levels; compare fixed-form least squares, PySINDy, and current Kynovar (if its model class can express this law). Label *pilot*, do not claim final significance.
- **Next frozen evaluation:** at least 30 synthetic seeds, held-out settings, 3 intervention strategies and 3 discovery engines under a controlled compute budget.
- **Real-data follow-up:** fit on one instrument-measured pendulum session and evaluate on independent sessions, with observation error and synchronization details recorded.
- **Exit gate:** publish the exact command, raw per-run tables, all unsuccessful attempts, plots, and a limitations/interpretation paragraph.

## Risks / interpretive cautions

- Near-linear small-angle observations cannot distinguish `sin(theta)` from `theta`; larger safe-angle interventions may be necessary for identifiability.
- Differentiating noisy angle measurements can create spurious dynamics; compare smoothing/weak formulations.
- Chaotic double pendulum trajectories diverge rapidly; compare appropriate short-horizon and statistical metrics, not only long-rollout RMSE.
- An LLM-generated hypothesis is not validated evidence. Scientific claims are derived from held-out interventions and independent evaluation.
