# Milestone 6 — Falsification

Status: implemented and measured. Source: `results/falsification/m6/falsification.json`.

## Components

`kynovar/falsification/challenger.py`

- `Challenger` scores candidate designs under four criteria: `uncertainty` (width of the leader's predictive interval), `disagreement` (standardized disagreement between active hypotheses), `extrapolation` (distance from the observed region in normalized design space), and `weakly_observed` (regions with few prior observations inside the observed bounding box). The scores are heuristics. None of them is called information gain.
- `challenge()` is prequential: the prediction and its 95% interval are recorded before the experiment runs. Only after that does the client run the design. The leader and all rival hypotheses are then judged, and every active hypothesis is refitted.
- `CounterexampleStore` is append-only JSONL. A counterexample records the hypothesis, challenge, criterion, design, fraction inside the interval, median and max |z|. Entries are never edited or deleted.
- Statuses come from the Milestone 4 manager: PROPOSED / SUPPORTED / CHALLENGED / CONTRADICTED / SUPERSEDED. There is no PROVEN.

## Acceptance experiment

World `inverse_square_k2.5`, noise 10% relative + 0.02 absolute, budget 8 challenges per run, 8 seeds, challenger vs random designs.

```bash
.venv/bin/python scripts/falsification.py --seeds 8 --budget 8
```

| Metric (n = 8 per mode) | Challenger | Random |
| --- | ---: | ---: |
| Runs where every materially wrong hypothesis ended CONTRADICTED | 8 | 5 |
| Materially wrong hypotheses left active (total) | 0 | 3 |
| Runs where the true structure was falsely contradicted | 0 | 0 |
| Final leader has the hidden structure | 8 | 8 |
| Counterexamples recorded (total) | 23 | 17 |
| Runs where every *structurally* wrong hypothesis was contradicted | 0 | 0 |

"Materially wrong" means that the final fitted hypothesis deviates from the hidden law by more than 5% relative RMS over the full design range. It is an evaluation-only quantity computed after the run. Runtime: 216 s.

## Interpretation

Under this setting the challenger removed every materially wrong hypothesis in 8 of 8 runs; random designs did so in 5 of 8. Neither mode contradicted the true structure.

Neither mode contradicts every *structurally* wrong hypothesis. The Yukawa family `c m1 m2 e^{-λr} / r^p` contains the true power law as λ → 0. After refitting, λ shrinks toward 0, and the hypothesis becomes numerically indistinguishable from the true law; there is nothing to contradict. That is why the "materially wrong" metric was added. The structural metric is still reported, and it is 0 in both modes.

## Limitations

- One world, one noise level, 8 seeds. The 8/8 vs 5/8 difference is not tested for significance. With n = 8, the 95% Wilson intervals overlap.
- The materially-wrong metric was introduced after an earlier run showed Yukawa converging to the power law. Only the 5% threshold was evaluated; sensitivity to that threshold was not measured.
- The `extrapolation` criterion is bounded by the feasibility filters. The challenger cannot propose designs that the scientist-side forward simulation predicts to be unstable.
