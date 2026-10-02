# Milestone 3 — Symbolic Law Discovery

Status: accepted. Final evidence: `results/discovery/m3-final/law_recovery.json`.

The discovery engine compares symbolic regression, SINDy, dimensional templates, and forward-selected power sums on held-out evidence. The final campaign used seeds 200–204 across eight worlds and 36 pairwise experiment attempts for each world. Ground truth is used only by evaluation.

All 40 runs recovered the correct structure and coefficients. The minimum per-world structural recovery rate and full recovery rate are both 1.0. Worst normalized held-out RMSE is `5.30e-9`; maximum coefficient relative error is `5.30e-9`, and maximum exponent absolute error is `9.11e-14`.

The composite gravity-plus-drag case exposed a nested complexity penalty in power-sum forward selection: an exact second term could be rejected even though the outer discovery engine already penalized completed-law complexity. Forward selection now uses held-out error improvement only; final model ranking continues to apply complexity once.

## Limitations

- The accepted catalog contains eight deterministic simulated law families; it is not evidence of universal symbolic discovery.
- The final campaign uses noise-free law-recovery worlds. Instrument-noise behavior is evaluated separately in the theory milestones.
- Search cost grows quickly with larger expression spaces and additional variables.
