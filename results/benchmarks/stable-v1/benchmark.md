# Kynovar dynamics benchmark (stable-v1)

Device: `cpu`

Dataset: `stable-v1`

Position RMSE is the root mean square error of the x and y components at that horizon.
Values are copied from the measured JSON record.

The acceleration term is a Huber loss on (a_hat - a) / (|a| + s), where s is the training acceleration scale; position and velocity terms use the median absolute deviation. Reported RMSE stays in physical units.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | 150-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 8.5510725e-05 | 0.0045433355 | 0.022967503 | 0.070409623 | 0.19110592 | 0.3888486 | 0 | 3.9926189 |
| linear | 7.3847491e-05 | 0.0039824556 | 0.020335768 | 0.06073687 | 0.1364316 | 0.29057694 | 26 | 3.4534653 |
| mlp | 7.1964163e-05 | 0.003921156 | 0.020363922 | 0.061478201 | 0.13071869 | 0.26167931 | 5122 | 96.081775 |
| gru | 7.61249e-05 | 0.0039191198 | 0.020129543 | 0.06021032 | 0.13321804 | 0.26982455 | 15106 | 167.90504 |
| gnn | 6.7740387e-05 | 0.0037010807 | 0.01917011 | 0.058425564 | 0.11745159 | 0.24002194 | 67138 | 90.874346 |
| interaction_gnn_single | 3.6761684e-05 | 0.0019944017 | 0.010556076 | 0.033340566 | 0.10165007 | 0.23101978 | 117252 | 170.21052 |
| interaction_gnn | 2.6751745e-05 | 0.0013484437 | 0.0058983526 | 0.020107165 | 0.056978475 | 0.14335529 | 155332 | 335.27912 |

## Bounded targets

Position RMSE on target states with true position norm ≤ 8, velocity norm ≤ 8, and acceleration norm ≤ 80.
Frames outside that box are stiff ejections and are omitted from this table only.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | 150-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity | 8.5510725e-05 | 0.0045433355 | 0.022967503 | 0.070409623 | 0.19110592 | 0.3888486 | 0 | 3.9926189 |
| linear | 7.3847491e-05 | 0.0039824556 | 0.020335768 | 0.06073687 | 0.1364316 | 0.29057694 | 26 | 3.4534653 |
| mlp | 7.1964163e-05 | 0.003921156 | 0.020363922 | 0.061478201 | 0.13071869 | 0.26167931 | 5122 | 96.081775 |
| gru | 7.61249e-05 | 0.0039191198 | 0.020129543 | 0.06021032 | 0.13321804 | 0.26982455 | 15106 | 167.90504 |
| gnn | 6.7740387e-05 | 0.0037010807 | 0.01917011 | 0.058425564 | 0.11745159 | 0.24002194 | 67138 | 90.874346 |
| interaction_gnn_single | 3.6761684e-05 | 0.0019944017 | 0.010556076 | 0.033340566 | 0.10165007 | 0.23101978 | 117252 | 170.21052 |
| interaction_gnn | 2.6751745e-05 | 0.0013484437 | 0.0058983526 | 0.020107165 | 0.056978475 | 0.14335529 | 155332 | 335.27912 |

## One-step acceleration

| Model | Acceleration RMSE | Acceleration R² | Relative RMSE | Bounded acceleration RMSE |
| --- | ---: | ---: | ---: | ---: |
| constant_velocity | 0.85510945 | -9.3090602e-06 | 1 | 0.85510945 |
| linear | 0.73847435 | 0.25418439 | 0.86360214 | 0.73847435 |
| mlp | 0.71964476 | 0.29173306 | 0.84158205 | 0.71964476 |
| gru | 0.76125087 | 0.27641001 | 0.85063779 | 0.76125087 |
| gnn | 0.67740464 | 0.37243752 | 0.79218473 | 0.67740464 |
| interaction_gnn_single | 0.36761547 | 0.81518033 | 0.42990458 | 0.36761547 |
| interaction_gnn | 0.26751649 | 0.91064105 | 0.29892858 | 0.26751649 |

## Rollout-aware training

interaction_gnn is trained to predict the next step. interaction_gnn_rollout uses the same architecture with rollout_steps=4 and lambda_rollout=1.

| Model | 1-step position RMSE | 10-step position RMSE | 25-step position RMSE | 50-step position RMSE | 100-step position RMSE | 150-step position RMSE | Parameters | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interaction_gnn | 2.6751745e-05 | 0.0013484437 | 0.0058983526 | 0.020107165 | 0.056978475 | 0.14335529 | 155332 | 335.27912 |
| interaction_gnn_rollout | 1.7353559e-05 | 0.0009758829 | 0.0058809098 | 0.019816912 | 0.055368734 | 0.11880371 | 155332 | 1665.3903 |
