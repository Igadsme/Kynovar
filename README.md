# KYNOVAR

Autonomous discovery and revision of physical laws in unknown simulated universes.

Kynovar is an experimental research system that observes sealed dynamical systems, learns trajectory dynamics, proposes symbolic laws, manages competing hypotheses with uncertainty, selects experiments, searches for counterexamples, detects silent changes in the governing physics, revises theories, and exposes the process through an interactive 3D laboratory.

> **Evidence status:** M1 is accepted. M2-M8 are implemented. Historical M4/M6/M7 and browser evidence are preserved under `results/`; the repository includes a sequential acceptance runner for fresh, uncontended validation. Do not interpret implementation or a historical artifact as a fresh PASS unless `results/acceptance/full.json` records it for the current commit.

## System

```text
unknown universe
  -> sealed observation / experiment API
  -> trajectory datasets and learned dynamics
  -> symbolic law discovery
  -> uncertain competing hypotheses
  -> active experiment selection
  -> adversarial falsification
  -> residual change detection + theory revision
  -> FastAPI + WebSocket API
  -> Next.js / React Three Fiber 3D laboratory
```

The scientist-facing code is intentionally separated from hidden ground truth. Hidden laws are available only to evaluation code and the backend universe registry; tests enforce the information boundary.

## Milestones

| Milestone | Capability | Repository status |
| --- | --- | --- |
| M1 | deterministic simulator, hidden universes, laboratory boundary | accepted |
| M2 | dynamics datasets, CV/linear/MLP/GRU/GNN models, stable-v1, OOD evaluation | implemented; fresh stable-v1 acceptance required |
| M3 | symbolic regression and law-recovery evaluation | implemented; final rerun required |
| M4 | hypotheses, uncertainty, evidence lifecycle, scientific notebook | implemented and measured |
| M5 | passive/random/grid/active experiment campaigns | implemented; clean comparison required |
| M6 | challenger, prequential tests, counterexample store | implemented and measured |
| M7 | residual monitoring, change-point estimation, theory versioning | implemented; total-uncertainty validation fix requires fresh seeds |
| M8 | 3D laboratory, FastAPI/WebSocket backend, Next.js/R3F frontend | implemented; browser evidence preserved |

Detailed reports live in `docs/`. Existing results are retained rather than overwritten so preliminary and accepted evidence remain distinguishable.

## Installation

Python 3.11+ and Node 22+ are recommended.

```bash
cd "/Volumes/T7 Shield/KYNOVAR"
export COPYFILE_DISABLE=1
make setup
make setup-web
```

The project drive is ExFAT, so `make setup` creates the virtual environment with copies instead of symlinks.

## Verify the application

Run the cheap structural acceptance first:

```bash
make smoke
```

This runs the Python/API tests, TypeScript typecheck, and Next.js production build and writes `results/acceptance/smoke.json`.

Run the full scientific acceptance **uncontended**:

```bash
make acceptance
```

The full profile additionally executes the stable-v1 dynamics benchmark and OOD suite, final M3 law-recovery run, M4 theory competition, clean M5 strategy comparison, M6 falsification, and M7 revision on a fresh seed range. It stops at the first failed stage and writes `results/acceptance/full.json` with the git commit, runtime, command, artifacts, and PASS/FAIL state.

A successful process exit alone does not make a scientific result strong; inspect each milestone's measured metrics and report before making a research claim.

The current automated suite contains 125 passing tests across simulation, discovery, theory management, leakage boundaries, the API, and reproducibility.

## Common commands

```bash
make test       # Python + API tests
make verify     # Python tests + frontend typecheck/build
make evaluate   # stable-v1 dynamics benchmark
make ood        # stable-v1 OOD evaluation
make discover   # M3 law recovery
make plan       # M5 active-experiment comparison
make falsify    # M6 challenger evaluation
make revise     # M7 change/revision evaluation
make backend    # FastAPI on :8000
make web        # Next.js development server on :3000
make web-build  # TypeScript + production Next.js build
make dev        # backend and frontend together
```

## Interactive laboratory

For local development, run both services with:

```bash
make dev
```

Then open `http://localhost:3000`. You can also run `make backend` and `make web` in separate terminals. The browser uses the same-origin `/api` proxy by default; set `NEXT_PUBLIC_KYNOVAR_API` and optionally `NEXT_PUBLIC_KYNOVAR_WS` only when serving the API from a different public origin. Configure cross-origin API clients with the comma-separated `KYNOVAR_CORS_ORIGINS` environment variable.

The interface supports hidden-universe creation, live autonomous discovery, 3D trajectories, theory exploration, user-designed challenges, adversarial experiment design, reveal/evaluation, and the scientific notebook. Backend discovery events stream over WebSocket, with polling fallback when a proxy does not support WebSockets.

## Deployment

Docker Compose provides a reproducible single-origin deployment with WebSocket proxying, health checks, persistent artifact storage, non-root application processes, and restart policies:

```bash
export COPYFILE_DISABLE=1
docker compose up --build -d
```

Open `http://localhost:8080`. Set `KYNOVAR_PORT` to publish a different host port. Terminate TLS at your load balancer or ingress and forward HTTP/WebSocket traffic to the gateway on port 8080.

The API image uses `requirements-api.txt`, which excludes training-only dependencies such as PyTorch. Run one API worker: discovery sessions currently live in process memory and are intentionally not shared between workers. The browser recreates its interactive world cleanly if the API process restarts.

## Research safeguards

- Discovery receives observations and experiment controls, not hidden force parameters.
- Dataset splits are assigned by universe, with disjointness checks.
- Hypothesis scores are predictive scores, **not probabilities**.
- A hypothesis can be supported, challenged, contradicted, superseded, rejected, or archived; Kynovar does not label a theory “proven.”
- Counterexample tests are prequential: predictions and intervals are recorded before the challenge result is revealed.
- M7 replacement validation uses total predictive uncertainty on fresh validation evidence rather than treating fitted absolute and relative noise components as independently identifiable quantities.
- Ground-truth comparisons are evaluation-only.

## Repository layout

```text
kynovar/simulator/       2D/3D simulation, forces, integration, universes
kynovar/laboratory/      sealed experiment interface
kynovar/data/            generation, stable regimes, splits, normalization, OOD
kynovar/models/          baselines, MLP/GRU/GNN/interaction GNN training
kynovar/discovery/       evidence extraction, symbolic search, law discovery
kynovar/uncertainty/     parametric uncertainty and predictive intervals
kynovar/theory/          hypotheses, manager, notebook, revision loop
kynovar/planning/        experiment design and acquisition strategies
kynovar/falsification/   challenger and counterexample storage
kynovar/evaluation/      benchmarks, law recovery, reports and plots
backend/kynovar_api/     FastAPI + WebSocket interactive API
frontend/                Next.js 15 + React Three Fiber laboratory
scripts/                 reproducible experiment and acceptance entry points
tests/                   simulator, leakage, discovery, theory, backend, 3D tests
results/                 measured research and browser evidence
```

## Current limitations

Kynovar is a research prototype, not a general-purpose physics engine. The interactive preset family is intentionally constrained, scientific acceptance currently focuses on simulated worlds, the API keeps session state in process memory, discovery work runs in the API process, and browser evidence is not a substitute for broad responsive/device testing. Nested model families can also be difficult to distinguish when a richer law collapses toward a simpler one.

## Documentation

- `docs/milestone-1.md` — simulator and information boundary
- `docs/milestone-2.md` and `docs/milestone-2-stabilization.md` — dynamics and stable-v1
- `docs/milestone-4-scientific-reasoning.md` — hypotheses and uncertainty
- `docs/milestone-6-falsification.md` — counterexample search
- `docs/milestone-7-theory-revision.md` — change detection and revision
- `docs/milestone-8-interactive-lab.md` — 3D/API/frontend evidence

## License

MIT. See `LICENSE`.
