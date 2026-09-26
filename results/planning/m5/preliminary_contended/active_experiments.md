# Milestone 5 experiment selection

Budget 10 experiments, 8 seeds, censored at budget+1 when never identified.
Acquisition: standardized disagreement: mean over path of Var_h[mean_h] / Mean_h[sigma_h^2]; a heuristic, not information gain.

| World | Strategy | Identified | Mean exps (censored) | Std | Obs. at id. | Wall s at id. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| inverse_power_k1.5_p2.6 | passive | 8/8 | 5.62 | 2.26 | 349 | 14.9 |
| inverse_power_k1.5_p2.6 | random | 8/8 | 4.50 | 2.07 | 279 | 9.7 |
| inverse_power_k1.5_p2.6 | grid | 8/8 | 4.62 | 2.77 | 287 | 9.4 |
| inverse_power_k1.5_p2.6 | active | 8/8 | 3.50 | 0.53 | 217 | 9.7 |
| inverse_square_k2.5 | passive | 8/8 | 5.00 | 1.41 | 310 | 8.2 |
| inverse_square_k2.5 | random | 8/8 | 3.50 | 0.76 | 217 | 4.6 |
| inverse_square_k2.5 | grid | 8/8 | 3.50 | 0.76 | 217 | 5.9 |
| inverse_square_k2.5 | active | 8/8 | 3.25 | 0.71 | 202 | 7.1 |
