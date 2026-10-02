# Milestone 5 experiment selection

Budget 10 experiments, 10 seeds, censored at budget+1 when never identified.
Acquisition: standardized disagreement: mean over path of Var_h[mean_h] / Mean_h[sigma_h^2]; a heuristic, not information gain.

| World | Strategy | Identified | Mean exps (censored) | Std | Obs. at id. | Wall s at id. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| inverse_power_k1.5_p2.6 | passive | 10/10 | 5.60 | 2.01 | 347 | 6.5 |
| inverse_power_k1.5_p2.6 | random | 10/10 | 4.60 | 2.07 | 285 | 5.9 |
| inverse_power_k1.5_p2.6 | grid | 10/10 | 4.30 | 2.54 | 267 | 4.4 |
| inverse_power_k1.5_p2.6 | active | 10/10 | 3.50 | 0.53 | 217 | 5.0 |
| inverse_square_k2.5 | passive | 10/10 | 5.10 | 1.29 | 316 | 5.1 |
| inverse_square_k2.5 | random | 10/10 | 3.40 | 0.70 | 211 | 3.7 |
| inverse_square_k2.5 | grid | 10/10 | 3.60 | 0.70 | 223 | 5.3 |
| inverse_square_k2.5 | active | 10/10 | 3.20 | 0.63 | 198 | 4.1 |
