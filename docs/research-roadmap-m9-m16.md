# Kynovar 2.0: Scientific Research Roadmap (M9–M16)

**Status:** proposed; no claims of passing these milestones.
**Research thesis:** Can a closed-loop system infer interpretable dynamical laws from noisy, partial observations, choose informative interventions efficiently, and detect/recover from changing dynamics?

## Research principles

1. A model's *predictive fit* is not evidence that the equation is identifiable or scientifically correct.
2. Ground truth from a simulator must remain inaccessible to the scientist and all preprocessing/hyperparameter selection. Only the evaluator may read it.
3. Distinguish **synthetic observations**, **numerically generated scientific reference data** (e.g. JPL Horizons/JHTDB), and **instrumentally measured real-world data**.
4. Lock evaluation tasks and splits before running final experiments. Separate development, validation, and held-out evaluation.
5. Compare methods at equivalent intervention, observation, compute and wall-clock budgets.
6. Report negative results, failed experiments, calibration, sensitivity analyses and uncertainty, not just successes.
7. Never claim new physical laws if the system is rediscovering already-known relationships.

## Current baseline (M1–M8, October 2026)

The current system has a 2D/3D simulated lab, neural dynamics models, candidate symbolic theories, active experiment selection, counterexample search, change detection, theory revision, and a browser interface.

The current M8 UI is restricted to pairwise force families plus monomial sums; it does not discover drag or spring laws there. The M3 accepted law-recovery experiment used noise-free simulated law families. The stored `results/acceptance/full.json` had `reuse_artifacts: true` and therefore does not itself demonstrate an independent full reproduction from source. Keep all M1–M8 baseline reports as historical results; do not overwrite them with M9 data.

## Milestones

### M9 — Reproducibility, protocol, and claim audit (first)
**Objective:** establish trustworthy, fresh-run research infrastructure.

**Build**
- Add a new `research/` namespace for versioned protocols, dataset cards, experiment manifests, and results.
- Produce clean-environment commands that re-run the scientific benchmarks, explicitly without `--reuse-artifacts`, and capture fresh environment fingerprints and artifact hashes.
- Record seeds, configuration, dependency versions, git SHA, CPU/GPU, elapsed time, raw metrics, and failure traces.
- Keep separate smoke CI (fast), offline scientific acceptance (slow), and optional nightly runs.
- Fix packaging/repository license discrepancy (pyproject says MIT while root LICENSE was deleted); document supported Python/Node versions.
- Publish a claim table: exact dataset, N, seeds, budget, failure policy, confidence interval and evidence path for every headline metric.

**Acceptance:** a new clean run yields complete artifacts, all provenance fields, no test leakage, and an auditable summary; failed runs remain visible rather than discarded.

### M10 — Research-grade dynamics environments
**Objective:** broaden from narrow pairwise laws to representative dynamical systems.

**Build** independent environment adapters for (i) damped and driven harmonic oscillator; (ii) nonlinear single pendulum; (iii) coupled/double pendulum; (iv) 3-body gravitational dynamics; and later (v) 1-D Burgers or wave PDE.

Each exposes state, observable sensors, physical units, interventions, noise process, cost/safety limits and the evaluator-only reference law. Prefer established adaptive solvers and convergence checks; for conservative Hamiltonian systems, compare long-term energy behavior against a symplectic baseline.

**Acceptance:** tests for integrator convergence, physical invariants where applicable, controlled unit conversions, seed determinism, sensor noise, and evaluator/scientist isolation. No global claim that one model works on every physics domain.

### M11 — Competing symbolic system-identification engines
**Objective:** remove search-family dependency.

**Build**
- Common model interface for existing symbolic engine, PySINDy, PySR, and restricted equation-template baselines.
- Explicit variable dimensions, unit constraints, equation canonicalization, complexity–error Pareto fronts, parameter uncertainty, and identifiability diagnostics.
- Handle derivatives estimated from noisy data; compare smoothing, weak-form SINDy, and direct integral/trajectory fitting.
- Track when two models are observationally indistinguishable.

**Acceptance:** report symbolic structure recovery, coefficient error, units validity, forecasting, negative log likelihood, noise sensitivity, and compute cost over pre-registered held-out worlds.

### M12 — Cost-aware and intervention-aware experimental design
**Objective:** measure whether autonomous experimentation actually helps.

**Build**
- A task-specific intervention schema (e.g. pendulum release angle, length, mass, drive frequency), with feasibility bounds and cost.
- Candidate selection by predictive disagreement, approximate expected information gain, and uncertainty reduction, plus random/grid/uncertainty baselines.
- Explicit stop rules, abstention, failed experiment logging, pre-registered intervention budgets.

**Acceptance:** across fixed held-out families, compare discovery rate vs number of experiments, total observations, compute-seconds and cost. Demonstrate benefit with confidence intervals and paired seeds, or report no benefit. The current disagreement heuristic is not exact information gain.

### M13 — Partial observability, measurement noise and system shift
**Objective:** make data imperfect on purpose.

**Build** camera/sensor observation operators; time jitter, occlusion, missing channels, correlated and heteroscedastic noise; latent velocity handling; controlled and unannounced changes in friction, damping, coupling, actuation or boundary conditions.

**Acceptance:** report calibration coverage, failure-to-detect and false-alarm rates, change-detection delay, time to identify replacement law, and performance at multiple held-out noise levels.

### M14 — Real scientific measurements
**Objective:** show the pipeline operates on observed physics.

**First target:** real pendulum motion. Ingest an instrument-measured open dataset and also record independent, controlled phone-camera/sensor trials if possible. Extract angles and timing with uncertainty; document apparatus, calibration and provenance.

**Next targets:** IBM's experimental double-pendulum videos/positions; NASA PCoE measured battery aging signals, as a separate scientific domain. Treat JPL Horizons ephemerides and JHTDB as modeled/numerical references, **not** raw physical measurement.

**Acceptance:** reproduce conventional baselines, use condition/session-level held-out splits, explain whether Kynovar merely fits known equations or learns a transferable model, and report uncertainty and identifiable limitations. Active experimentation requires a controllable physical rig or a logged intervention bank with explicit constraints.

### M15 — Bounded autonomous research agent
**Objective:** connect discovery into a repeatable, inspectable decision cycle.

**Build** a typed scientist state machine: observe → propose candidate models → select experiment → run within limits → compare posterior predictions → challenge → revise / abstain. Keep an append-only evidence log, budget and timeouts. An LLM may explain and orchestrate only; numerical solvers and evaluators determine the evidence.

**Acceptance:** zero hidden-answer access; deterministic replay of a saved campaign; safe stop; finite budgets; explicit abstention on insufficient evidence; agent-vs-non-agent ablation.

### M16 — Reproducible benchmark and research paper
**Objective:** create a publishable, falsifiable research contribution.

**Build**
- Evaluation suite and public benchmark runner with fixed train/validation/test families.
- Minimum 30 random seeds for feasible synthetic conditions and multiple independent physical recording sessions when possible; include intervals and paired tests.
- Ablations: without dimensional constraints, without active experiments, without uncertainty, without revision, without neural dynamics.
- Data cards, model cards, runnable scripts, artifact index, notebooks, figures, limitations, and threat-to-validity section.
- Manuscript in `paper/` centered on **one** verified contribution, not a broad claim of general artificial scientific discovery.

**Acceptance:** another researcher can re-run headline tables under documented hardware and budget; the chosen method improves an appropriate baseline on a pre-registered task, or the paper clearly presents a negative finding.

## Suggested sequence / rough solo-development planning

- Phase I (4–8 weeks): M9 + initial M10 damped oscillator/pendulum.
- Phase II (8–12 weeks): M11–M12 and paired synthetic benchmarks.
- Phase III (8–12 weeks): M13 and initial M14 real pendulum evaluation.
- Phase IV (8–12 weeks): M15–M16, ablations and manuscript.

These are planning estimates, not delivery commitments. Prioritize depth and replication over adding many physics families.

## Scientific boundaries

- **Prediction ≠ explanation:** low error alone does not establish a law.
- **Discovery ≠ rediscovery:** known-form equation recovery is system identification; call it that.
- **Generated reference ≠ measurement:** JPL Horizons and JHTDB are useful but not physical measurement datasets.
- **Dataset selection ≠ intervention:** choosing pre-existing trajectories is offline selection, not autonomous control of a live experiment.
- **Closed-world evidence ≠ unrestricted scientific autonomy:** keep domain and observation assumptions explicit.

## Relevant prior work/resources

- PySINDy: https://pysindy.readthedocs.io/en/stable/
- PySR dimensional constraints: https://pysr.ai/v2.7.0/examples/physics
- SRBench: https://github.com/cavalab/srbench
- Experimental double pendulum dataset: https://ibm.github.io/double-pendulum-chaotic-dataset/
- Laboratory-grade multi-link pendulum and data: https://pmc.ncbi.nlm.nih.gov/articles/PMC10458325/
- NASA PCoE measured battery data: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- JPL Horizons API: https://ssd-api.jpl.nasa.gov/doc/horizons.html
- Johns Hopkins Turbulence Database: https://turbulence.idies.jhu.edu/database
- GWOSC: https://gwosc.org/api/
- FIRE-Bench science agent evaluation: https://proceedings.mlr.press/v306/wang26d.html

See `docs/research-protocol-v1.md` for the first pre-registered experiment.
