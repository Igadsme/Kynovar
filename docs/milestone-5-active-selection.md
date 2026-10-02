# Milestone 5 — Active Experiment Selection

Status: accepted. Final evidence: `results/planning/m5-final/active_experiments.json`.

Passive, random, grid, and active campaigns were compared for 10 seeds in each of two inverse-power worlds, with a budget of 10 experiments. The acquisition score is standardized predictive disagreement; it is a practical heuristic, not a claim of exact information gain.

Active selection recovered the law in 20 / 20 runs. Averaged across the two worlds, it required 3.35 experiments, versus 4.00 for random and 5.35 for passive. The active/random ratio is 0.8375, below the 0.95 acceptance threshold, and active never recovered fewer laws than random in either world.

## Limitations

- The comparison covers two related inverse-power families and a small controlled design space.
- Wall-clock time is not the optimization target; acquisition itself has computational cost.
- Broader law families may require different acquisition functions or larger budgets.
