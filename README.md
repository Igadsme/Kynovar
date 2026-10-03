# Kynovar

### Autonomous Scientific Discovery Engine

Kynovar is an AI-powered scientific discovery platform that creates simulated universes, learns their hidden physical laws from observations, designs experiments to test competing theories, detects when existing theories fail, and proposes revised explanations.

Instead of training a model to make predictions inside one fixed environment, Kynovar explores a harder question:

> **Can an AI system discover the rules governing an unfamiliar world — and recognize when those rules change?**

Kynovar combines simulation, machine learning, graph neural networks, symbolic discovery, active experimentation, falsification, and an interactive 3D laboratory into one end-to-end system.

---

## Overview

Traditional machine learning systems are typically trained and evaluated under similar data distributions. Kynovar creates controlled synthetic environments where the underlying laws can vary between universes.

Each universe contains hidden parameters that determine how objects interact. Kynovar only receives observations available through its laboratory interface and must infer the underlying behavior from data.

The system can:

- Generate controlled simulated universes with hidden physical laws
- Collect observational and experimental datasets
- Train and compare multiple machine learning architectures
- Discover candidate mathematical relationships
- Maintain competing theories and uncertainty estimates
- Select experiments that distinguish between competing hypotheses
- Detect when an existing theory stops explaining observations
- Propose replacement theories after environmental changes
- Evaluate models on unseen and out-of-distribution universes
- Visualize experiments through an interactive 3D web interface

---

## Key Results

Kynovar's current experimental pipeline includes **80 simulated universes** divided into:

- **56 training universes**
- **12 validation universes**
- **12 test universes**

The split is constructed to prevent universe-level data leakage.

Selected experimental results include:

| Experiment | Result |
|---|---:|
| Simulated universes | 80 |
| Train / Validation / Test | 56 / 12 / 12 |
| Universe-level leakage | 0 |
| Controlled GNN acceleration R² | **0.994** |
| Theory recovery evaluation | **5 / 5** |
| Targeted wrong-theory elimination | **8 / 8** |
| Random experiment baseline | **5 / 8** |
| Environmental change detection | **20 / 20** |
| Successful theory revision | **19 / 20** |

In one end-to-end experiment, Kynovar recovered a relationship approximately equivalent to:

`F = 3.201 × m₁m₂ / r^2.401`

from a universe whose hidden relationship was approximately:

`F = 3.2 × m₁m₂ / r^2.4`

The system was not directly given those hidden parameters.

---

## How Kynovar Works

Kynovar follows a scientific-discovery loop:

```text
                 ┌─────────────────────┐
                 │ Simulated Universe  │
                 │   Hidden Physics    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Observation     │
                 │      Laboratory     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Dataset Builder   │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │     ML Model Training     │
              │ Ridge / MLP / GRU / GNN  │
              └─────────────┬─────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Symbolic Discovery  │
                 │ Candidate Theories  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Theory Competition  │
                 │    + Uncertainty    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Active Experiment   │
                 │      Selection      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Falsification    │
                 └──────────┬──────────┘
                            │
                   ┌────────┴────────┐
                   │                 │
                 Pass              Fail
                   │                 │
                   ▼                 ▼
             Keep Theory      Revise Theory
                   │                 │
                   └────────┬────────┘
                            │
                            ▼
                       Repeat
```

The goal is not simply to find a model with the lowest prediction error. Kynovar attempts to construct, challenge, and revise explanations of the simulated environment.

---

## Machine Learning Models

Kynovar supports multiple modeling approaches so different inductive biases can be compared under the same experimental conditions.

### Ridge / Linear Models

Linear baselines establish whether a universe can be modeled using relatively simple relationships.

### Multilayer Perceptron

MLPs provide nonlinear function approximation without explicit temporal or relational structure.

### GRU

Gated Recurrent Units model sequential behavior and allow Kynovar to learn from temporal trajectories.

### Graph Neural Networks

Objects are represented as nodes and interactions as edges.

This allows the system to model pairwise relationships between objects directly and provides a natural representation for multi-object physical systems.

A controlled GNN experiment achieved an acceleration **R² of 0.994**.

---

## Symbolic Discovery

Prediction alone does not necessarily provide an interpretable explanation.

Kynovar therefore includes a symbolic discovery stage that attempts to convert observed relationships into candidate mathematical laws.

For example, rather than only learning:

```text
state → neural network → predicted acceleration
```

Kynovar can attempt to recover relationships of the form:

```text
F ∝ m₁m₂ / rᵖ
```

and estimate parameters such as the interaction constant and exponent.

This produces hypotheses that can subsequently be tested rather than treating a neural network as the final explanation.

---

## Theory Competition

Kynovar can maintain multiple candidate theories simultaneously.

Instead of immediately selecting one hypothesis, the system compares how well competing theories explain available observations.

Conceptually:

```text
Theory A ─┐
Theory B ─┼──► Evidence ──► Theory Scores
Theory C ─┘
```

When multiple theories explain existing data, Kynovar can search for an experiment where their predictions diverge.

---

## Active Experimentation

One of Kynovar's core capabilities is choosing **what experiment to run next**.

Rather than collecting observations randomly, the system searches for experimental conditions that maximize disagreement between competing theories.

This turns data collection into an optimization problem.

In controlled evaluation:

```text
Targeted experiment selection: 8 / 8
Random experiment baseline:    5 / 8
```

The targeted strategy eliminated incorrect theories more consistently than random experimentation in this evaluation.

---

## Falsification

Scientific theories must survive attempts to disprove them.

Kynovar continuously compares predictions from its current theory against new observations.

If prediction error exceeds expected uncertainty, the system can flag the current theory as potentially invalid.

```text
Theory
   │
   ▼
Prediction
   │
   ▼
Experiment
   │
   ▼
Observed Result
   │
   ├── Consistent ──► Retain Theory
   │
   └── Conflict ────► Falsification / Revision
```

---

## Theory Revision

The simulated environment can change after Kynovar has already learned a valid theory.

This creates a form of scientific distribution shift.

Kynovar must recognize that the existing explanation no longer fits the evidence and search for a replacement.

In controlled change experiments, the system:

- Detected **20 / 20** injected environmental changes
- Successfully recovered replacement theories in **19 / 20** cases

This allows Kynovar to study adaptation beyond conventional static train/test evaluation.

---

## Out-of-Distribution Evaluation

Kynovar separates universes at the environment level rather than randomly mixing observations from the same universes.

```text
80 Universes

├── 56 Training
├── 12 Validation
└── 12 Test
```

A test universe therefore represents an environment the system did not train on.

This makes it possible to evaluate whether learned representations and discovered relationships generalize beyond the environments used during training.

---

## Interactive Laboratory

Kynovar includes a browser-based interface for observing experiments and interacting with the scientific discovery system.

The application combines:

- **Next.js**
- **TypeScript**
- **React**
- **React Three Fiber / Three.js**
- **FastAPI**
- **WebSockets**

The frontend communicates with the scientific backend in real time, allowing experiments and theory updates to be visualized as they occur.

---

## System Architecture

```text
┌──────────────────────────────────────────────┐
│                 Web Interface                │
│                                              │
│     Next.js + TypeScript + React / R3F       │
└──────────────────────┬───────────────────────┘
                       │
                REST / WebSocket
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                 FastAPI API                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Scientific Discovery Engine         │
│                                              │
│  Theory Competition │ Active Experiments     │
│  Falsification      │ Theory Revision        │
│  Symbolic Discovery │ Uncertainty            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              ML / Modeling Layer             │
│                                              │
│       Ridge │ MLP │ GRU │ GNN                │
│              PyTorch                         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              Simulation Engine               │
│                                              │
│  Controlled Universes + Hidden Parameters    │
└──────────────────────────────────────────────┘
```

---

## Tech Stack

### Machine Learning & Scientific Computing

- **Python**
- **PyTorch**
- **NumPy**
- **Graph Neural Networks**
- **GRUs**
- **MLPs**
- **Ridge Regression**
- Symbolic discovery
- Statistical evaluation

### Backend

- **FastAPI**
- REST APIs
- **WebSockets**
- Python scientific pipeline

### Frontend

- **Next.js**
- **React**
- **TypeScript**
- **React Three Fiber**
- **Three.js**

### Infrastructure

- **Docker**
- **PostgreSQL**
- Production web deployment
- Runtime API proxying
- Secure WebSocket communication

---

## Engineering Challenges

Kynovar combines several problems that are often treated independently:

**Simulation engineering** — creating deterministic environments with controlled hidden parameters.

**Machine learning** — training models that generalize across environments rather than memorizing one universe.

**Graph learning** — representing interactions between multiple objects.

**Scientific inference** — transforming observations into candidate explanations.

**Experiment design** — deciding which observations provide the most information.

**Falsification** — identifying when a previously successful explanation stops working.

**Distribution shift** — evaluating models in environments not represented during training.

**Full-stack engineering** — exposing the research pipeline through APIs, WebSockets, and an interactive 3D frontend.

---

## Project Milestones

Kynovar was developed incrementally around a series of scientific and engineering milestones.

### M1 — Simulation & Laboratory

Deterministic simulation environment, hidden universe parameters, observation boundaries, and laboratory API.

### M2 — Machine Learning

Dataset generation and baseline models including Ridge, MLP, GRU, and GNN architectures.

### M3 — Symbolic Discovery

Generation and evaluation of interpretable candidate physical relationships.

### M4 — Theory Competition

Comparison of competing explanations against experimental evidence.

### M5 — Uncertainty

Represent uncertainty around predictions and candidate theories.

### M6 — Active Experimentation

Select experiments designed to maximally distinguish competing hypotheses.

### M7 — Falsification & Revision

Detect theory failure and search for replacement explanations when environments change.

### M8 — Interactive Laboratory

FastAPI backend, real-time WebSocket communication, and a Next.js/3D browser interface.

---

## Why I Built Kynovar

Most machine learning projects ask:

> "Can the model predict the correct answer?"

Kynovar started from a different question:

> **"Can a machine discover why something happens, determine what experiment would test its explanation, and recognize when that explanation is wrong?"**

The project gave me a way to explore the intersection of **software engineering, machine learning, graph neural networks, simulation, scientific reasoning, and autonomous AI systems** while building the complete system from the simulation layer to a deployed interactive interface.

---

## Future Work

Potential extensions include:

- Larger and more diverse universe families
- More complex multi-body environments
- Improved symbolic regression
- Probabilistic theory representations
- Bayesian experiment selection
- Stronger uncertainty calibration
- More advanced graph architectures
- Automated experiment scheduling
- Expanded OOD benchmarks
- Real-world scientific datasets
- Vision-based observation
- Autonomous multi-stage discovery loops

---

## Repository

**GitHub:** [github.com/Igadsme/Kynovar](https://github.com/Igadsme/Kynovar)

---

## Author

**Imani Gad**

Computer Science — Kennesaw State University  
Software Engineering • Machine Learning • AI Systems • Backend Engineering

[Portfolio](https://imanigad.com) • [LinkedIn](https://www.linkedin.com/in/igad/) • [GitHub](https://github.com/Igadsme)

---

## Disclaimer

Kynovar is an experimental software and machine-learning research project. Its synthetic-universe results demonstrate behavior within controlled simulation environments and should not be interpreted as evidence of autonomous scientific discovery in unrestricted real-world scientific settings.

## Current limitations

Kynovar is a research prototype, not a general-purpose physics engine. The interactive preset family is intentionally constrained, scientific acceptance currently focuses on simulated worlds, the API keeps session state in process memory, discovery work runs in the API process, and browser evidence is not a substitute for broad responsive/device testing. Nested model families can also be difficult to distinguish when a richer law collapses toward a simpler one.

## Documentation

- `docs/milestone-1.md` — simulator and information boundary
- `docs/milestone-2.md` and `docs/milestone-2-stabilization.md` — dynamics and stable-v1
- `docs/milestone-3-law-discovery.md` — symbolic law recovery
- `docs/milestone-4-scientific-reasoning.md` — hypotheses and uncertainty
- `docs/milestone-5-active-selection.md` — active experiment selection
- `docs/milestone-6-falsification.md` — counterexample search
- `docs/milestone-7-theory-revision.md` — change detection and revision
- `docs/milestone-8-interactive-lab.md` — 3D/API/frontend evidence
