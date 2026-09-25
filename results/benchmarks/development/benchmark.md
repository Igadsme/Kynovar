# Kynovar dynamics benchmark (development)

Device: `mps`

Dataset: `development`

Position RMSE is the root mean square error of the x and y components at that horizon.
Values are copied from the measured JSON record.

Learned models minimize a Huber loss on residuals divided by the training-set median absolute deviation. Reported RMSE stays in physical units.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 0.19733903 | 3.7781786 | 12.033341 | 29.092691 | 85.805312 | 0 | 2.945691 |
| linear | 0.19733908 | 3.7781859 | 12.033407 | 29.093166 | 85.808507 | 26 | 1.5222036 |
| mlp | 0.19734016 | 3.7783458 | 12.034473 | 29.096994 | 85.820064 | 3074 | 24.935297 |
| gru | 0.092000153 | 1.7342241 | 5.5072771 | 13.848312 | 43.575336 | 9026 | 31.912237 |
| gnn | 0.19734457 | 3.7785236 | 12.035738 | 29.099727 | 85.803272 | 38066 | 16.444945 |

## Bounded targets

Position RMSE on target states with true position norm ≤ 8, velocity norm ≤ 8, and acceleration norm ≤ 80.
Frames outside that box are stiff ejections and are omitted from this table only.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 0.0022515427 | 0.028905954 | 0.079003849 | 0.18700306 | 0.54751519 | 0 | 2.945691 |
| linear | 0.002252476 | 0.02895268 | 0.079234779 | 0.18718365 | 0.55000094 | 26 | 1.5222036 |
| mlp | 0.0022464752 | 0.028505961 | 0.076151082 | 0.17978603 | 0.5155427 | 3074 | 24.935297 |
| gru | 0.0019738613 | 0.022930315 | 0.083703587 | 0.22655048 | 0.74380646 | 9026 | 31.912237 |
| gnn | 0.0022418538 | 0.029987118 | 0.095472819 | 0.29573007 | 0.80106002 | 38066 | 16.444945 |

## Rollout-aware training

gnn is trained to predict the next step. gnn_rollout uses the same architecture with rollout_steps=4 and lambda_rollout=1.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gnn | 0.19734457 | 3.7785236 | 12.035738 | 29.099727 | 85.803272 | 38066 | 16.444945 |
| gnn_rollout | 0.19734419 | 3.7786263 | 12.036112 | 29.100122 | 85.814916 | 38066 | 21.84304 |
