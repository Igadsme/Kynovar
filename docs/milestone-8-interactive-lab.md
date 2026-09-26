# Milestone 8 — Interactive Laboratory

Status: implemented. The browser acceptance evidence is in `results/interactive/m8/` (screenshots `01`–`09` and the raw API state `e2e_run.json`).

## 3D physics

`kynovar/simulator/world3d.py` adds `World3D`, `Laboratory3D`, `BodyInit3D`, and `observation_array_3d` (11 channels: x, y, z, vx, vy, vz, ax, ay, az, mass, radius). The 2D simulator is unchanged. Discovery code detects 8- vs 11-column layouts (`kynovar/discovery/evidence.py::state_layout`), so pairwise evidence, feasibility checks, forward simulation, and the instrument work in both. Tests (`tests/test_world3d.py`) check that:

- 3D momentum is conserved and accelerations match the law;
- a 3D world restricted to a plane matches the 2D world;
- boundaries and validation behave correctly;
- discovery works from 3D observations;
- design dimension is detected correctly.

`tests/test_backend.py` covers the world lifecycle (no hidden keys in public responses), discovery + challenge + metrics, WebSocket event streaming, and background start/stop.

## Backend

`backend/kynovar_api/app.py` (FastAPI). Run with `make backend` or

```bash
.venv/bin/uvicorn backend.kynovar_api.app:app --host 127.0.0.1 --port 8000
```

| Endpoint | Purpose |
| --- | --- |
| `POST /worlds`, `GET /worlds`, `GET /worlds/{id}` | Create or list sealed universes. The public view says `governing_laws: UNKNOWN`. |
| `POST/GET /worlds/{id}/experiments` | Run a user-designed 3D experiment; list all experiments with observed positions |
| `POST /worlds/{id}/discovery/start`, `/stop`, `GET /status` | Background discovery loop (`kynovar/discovery/session.py`) |
| `GET /worlds/{id}/theories` | Ranked hypotheses, counterexamples, ledger |
| `POST /worlds/{id}/challenge` | With a design: the leader's prediction (positions and 95% force interval), recorded before the universe runs. Without a design: the challenger's adversarial design for a chosen criterion. |
| `POST /worlds/{id}/challenge/{cid}/reveal` | Runs the experiment, judges the prediction, records counterexamples |
| `GET /worlds/{id}/metrics` | Ledger plus an EVALUATION ONLY comparison with the hidden law |
| `GET /worlds/{id}/notebook` | Notebook entries |
| `WS /ws/{id}` | Stream of experiment, theory, notebook, and status events |

`backend/kynovar_api/universes.py` is the only backend module that knows hidden laws. The discovery session receives an `ExperimentClient` and nothing else. The leakage test (`tests/test_leakage.py`) scans the scientist packages. The evaluation endpoint computes its comparison server-side and never passes it to the session.

## Frontend

`frontend/` — Next.js 14, React 18, TypeScript, React Three Fiber / drei / three, KaTeX. Run with `make frontend` (production build + start on port 3000) after `make backend`.

- **Landing screen**: KYNOVAR / AUTONOMOUS SCIENTIFIC DISCOVERY / UNIVERSE K-0042 / GOVERNING LAWS UNKNOWN / EXPERIMENTS 0 / THEORY STATE UNINITIALIZED / [BEGIN DISCOVERY]. Every value is read from the backend. If the backend is unreachable the page says so; it never shows placeholder numbers.
- **Discovery view**: live 3D replay of the latest experiment, loop controls, ledger, the leading theory in KaTeX with status, score (labelled "not a probability"), and parameter bootstrap intervals, competing hypotheses, and recent notebook entries.
- **Theories**: ranked hypotheses, per-experiment verdicts, permanent counterexamples.
- **Challenge Kynovar**: the user edits masses, 3D positions, and velocities, then PREDICT FUTURE shows the dashed predicted paths and REVEAL REALITY overlays the real paths with the verdict, the fraction inside the 95% interval, and trajectory RMSE.
- **Break the theory**: the challenger designs an attack for a chosen criterion (uncertainty, disagreement, extrapolation, weakly observed), followed by the same predict/reveal flow.
- **Notebook**: the full event log, filterable by kind.
- **Evaluation**: hidden ground truth vs Kynovar, behind an explicit EVALUATION ONLY banner.

Visual style: near-black background, off-white monospace text, one amber accent, status colors only for SUPPORTED/CHALLENGED/CONTRADICTED.

## Browser acceptance run

Real browser session against the live backend and production frontend. No mocks. The steps, in order:

1. Landing screen showed K-0042, 0 experiments, UNINITIALIZED (`01_landing.png`).
2. BEGIN DISCOVERY started the loop, and experiments streamed over the WebSocket (`02_discovery_running.png`).
3. The loop converged after 10 experiments on `F = 3.201 m1 m2 / r^2.401`, status SUPPORTED (`03_discovery_converged.png`).
4. The theory explorer showed 3 hypotheses and 5 supporting verdicts for the leader, with 91–99% of observations inside its 95% interval (`04_theory_explorer.png`).
5. For a user-designed challenge (masses 1.2 / 0.9, separation 2.4), the prediction gave an initial force of 0.418 [0.398, 0.438]. The reveal: supports, 94% inside, median |z| 0.72, trajectory RMSE 4.84e-5 (`05`, `06`).
6. Break the theory, extrapolation criterion: supports, 96% inside, median |z| 0.79, RMSE 3.42e-5 (`07_break_the_theory.png`).
7. The notebook held 51 real entries (`08_notebook.png`).
8. Evaluation against the hidden law 3.2 m1 m2 / r^2.4: structural match, coefficient relative error 6.25e-4, exponent error 0.002, held-out trajectory RMSE 5.7e-5 to 7.3e-5, first correct leader after 4 experiments (`09_evaluation.png`).

During an earlier session, a user challenge with masses 1.8 / 0.9 at separation 1.6 produced a close encounter. The outcome was reported as "EXPERIMENT NOT USABLE · acceleration above instrument range", and the theory was not tested on it. After that the reveal endpoint was changed to return the real trajectory for unusable outcomes as well, and the default design's separation was widened to 2.4.

## Limitations

- The preset K-0042 is a pure pairwise power law. The interactive loop searches only pairwise families plus a monomial sum; it would not discover drag or a spring in this UI.
- The session is seeded, so repeated runs of K-0042 reproduce the same experiment sequence.
- The discovery loop runs in a thread inside the API process; one process serves all universes, and nothing persists across restarts.
- The WebSocket polls an in-memory event list every 0.2 s.
- The acceptance run was performed at a 1920 × 1080 viewport. Smaller screens were not tested.
