# Milestone 4 — Scientific Reasoning

Status: implemented and measured. Source: `results/theory/m4/theory_competition.json`, example notebook `results/theory/m4/notebook_seed0.txt`.

## Components

| Module | Role |
| --- | --- |
| `kynovar/theory/hypothesis.py` | `Hypothesis` objects, `Status` enum, `judge()` |
| `kynovar/theory/manager.py` | `TheoryManager`: propose, adopt, test, refit, merge, resolve nesting, rank |
| `kynovar/theory/notebook.py` | Append-only research notebook; every entry is rendered from a structured payload |
| `kynovar/theory/families.py` | Candidate families: power law, exponential screening, linear in r, Yukawa |
| `kynovar/uncertainty/parametric.py` | Parametric laws, experiment-level bootstrap, noise model, calibration utilities |

A hypothesis records its equation, LaTeX, variables, parameters with bootstrap standard deviations and 95% intervals, fit error (weighted NMSE, definition stored beside it), complexity, source, supporting and contradicting experiments, per-experiment verdicts, timestamps, status, and score.

The score is the mean log predictive density minus `0.01 × complexity`. It is stored together with the sentence "not a probability". No hypothesis carries a probability.

Status values are `PROPOSED`, `SUPPORTED`, `CHALLENGED`, `CONTRADICTED`, `SUPERSEDED`, `REJECTED`, `ARCHIVED`. There is no `PROVEN`.

Rules in `TheoryManager._apply_rules`:

- A test *supports* when at least 80% of the new observations fall inside the hypothesis' 95% predictive interval; it *contradicts* below 50%; in between it is inconclusive.
- PROPOSED → SUPPORTED after 2 supporting experiments.
- The first contradiction moves the hypothesis to CHALLENGED. A second contradiction, or any single experiment with less than 20% inside the interval, moves it to CONTRADICTED.
- CHALLENGED → SUPPORTED after 3 consecutive supports.

## Uncertainty

Parameter uncertainty is an experiment-level bootstrap: whole experiments are resampled, not individual frames, because frames inside one trajectory are correlated. Predictive variance is

```text
sigma^2 = var_bootstrap(f) + a^2 + (b f)^2
```

where `a` and `b` are fitted by maximum likelihood on the fitting residuals. An earlier version used a wider residual-based noise model. Its nominal 50% intervals covered 81% of held-out data, so it was replaced before the acceptance run below.

## Acceptance experiment

World `inverse_square_k2.5` (hidden to the scientist), instrument noise 3% relative + 0.005 absolute on acceleration, 5 seeds.

```bash
.venv/bin/python scripts/theory_competition.py --seeds 5
```

| Metric | Value (n = 5) |
| --- | ---: |
| Acceptance rate | 1.0 |
| Early evidence leaves at least two distinct plausible structures | 1.0 |
| Leader has the hidden structure at the end | 1.0 |
| Strict acceptance (every wrong structure contradicted, not only pruned by parsimony) | 0.4 |
| Wrong structure still active before the parsimony rule | 1.0 |
| Runtime | 50.1 s |

Acceptance definition (stored in the JSON): at least two distinct structures are non-contradicted after the early held-out tests, and afterwards at least one early-plausible wrong structure is CONTRADICTED and the leader has the hidden structure.

Calibration of the 50/80/90/95% predictive intervals on held-out experiments:

| Nominal | Empirical mean | std | n |
| ---: | ---: | ---: | ---: |
| 0.50 | 0.562 | 0.055 | 5 |
| 0.80 | 0.842 | 0.042 | 5 |
| 0.90 | 0.922 | 0.033 | 5 |
| 0.95 | 0.961 | 0.019 | 5 |

Intervals are slightly conservative at every level.

## Limitations

- The strict criterion passes in only 2 of 5 seeds. Exponential-screening and linear-in-r hypotheses are contradicted by wide-range data, but Yukawa-type hypotheses (`r^-p e^{-λr}`) contain the power law as the limit λ → 0. They fit the data equally well and are removed by `resolve_nested`, a parsimony rule, not by contradiction. That is reported separately as `wrong_structure_active_before_parsimony_rate = 1.0`.
- `merge_equivalents` compares term-wise bootstrap intervals; it can merge hypotheses whose differences are below the noise level.
- Calibration is measured on one world and one noise level.
- The complexity penalty 0.01 was chosen by hand, not tuned on test data.
