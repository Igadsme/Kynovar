# Kynovar dynamics benchmark (controlled-2body)

Device: `mps`

Dataset: `controlled-2body`

Position RMSE is the root mean square error of the x and y components at that horizon.
Values are copied from the measured JSON record.

The acceleration term is a Huber loss on (a_hat - a) / (|a| + s), where s is the training acceleration scale; position and velocity terms use the median absolute deviation. Reported RMSE stays in physical units.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | 150-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 2.8078809e-05 | 0.0014632419 | 0.0070659224 | 0.023701261 | 0.08288926 | 0.179257 | 0 | 3.1681318 |
| linear | 2.1137343e-05 | 0.0010873321 | 0.0048945367 | 0.015895065 | 0.057820549 | 0.1344904 | 26 | 1.9346344 |
| mlp | 1.1653759e-05 | 0.00060916792 | 0.0030165151 | 0.011012236 | 0.042144956 | 0.095793389 | 5122 | 92.509378 |
| gnn | 2.1774989e-06 | 0.00010909487 | 0.0003467315 | 0.00090782215 | 0.0027693061 | 0.0059314723 | 67138 | 92.841312 |
| interaction_gnn_single | 1.5900883e-06 | 8.7006451e-05 | 0.00032745288 | 0.00075052402 | 0.0017415507 | 0.0035223896 | 117252 | 181.0481 |

## Bounded targets

Position RMSE on target states with true position norm ≤ 8, velocity norm ≤ 8, and acceleration norm ≤ 80.
Frames outside that box are stiff ejections and are omitted from this table only.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | 150-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 2.8078809e-05 | 0.0014632419 | 0.0070659224 | 0.023701261 | 0.08288926 | 0.179257 | 0 | 3.1681318 |
| linear | 2.1137343e-05 | 0.0010873321 | 0.0048945367 | 0.015895065 | 0.057820549 | 0.1344904 | 26 | 1.9346344 |
| mlp | 1.1653759e-05 | 0.00060916792 | 0.0030165151 | 0.011012236 | 0.042144956 | 0.095793389 | 5122 | 92.509378 |
| gnn | 2.1774989e-06 | 0.00010909487 | 0.0003467315 | 0.00090782215 | 0.0027693061 | 0.0059314723 | 67138 | 92.841312 |
| interaction_gnn_single | 1.5900883e-06 | 8.7006451e-05 | 0.00032745288 | 0.00075052402 | 0.0017415507 | 0.0035223896 | 117252 | 181.0481 |

## One-step acceleration

| Model | Acceleration RMSE | Acceleration R² | Relative RMSE | Bounded acceleration RMSE |
| --- | ---: | ---: | ---: | ---: |
| constant_velocity | 0.28078369 | -0.00035035464 | 1 | 0.28078369 |
| linear | 0.21137352 | 0.43309597 | 0.75279844 | 0.21137352 |
| mlp | 0.11653512 | 0.82768542 | 0.41503521 | 0.11653512 |
| gnn | 0.021766685 | 0.99398836 | 0.077521187 | 0.021766685 |
| interaction_gnn_single | 0.015894253 | 0.99679455 | 0.056606753 | 0.015894253 |

## Rollout-aware training

interaction_gnn_single is trained to predict the next step. interaction_gnn_single_rollout uses the same architecture with rollout_steps=4 and lambda_rollout=1.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | 150-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interaction_gnn_single | 1.5900883e-06 | 8.7006451e-05 | 0.00032745288 | 0.00075052402 | 0.0017415507 | 0.0035223896 | 117252 | 181.0481 |
| interaction_gnn_single_rollout | 1.2879735e-06 | 6.9037212e-05 | 0.0002588207 | 0.00065049548 | 0.0016448433 | 0.0032157334 | 117252 | 311.23523 |
