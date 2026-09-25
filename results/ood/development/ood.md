# Kynovar OOD evaluation (development)

Device: `mps`

Each cell is position RMSE at the stated horizon.

## exponent

Pairwise exponent sampled from [4, 6), above the training prior [0.5, 4).

| Model | 1-step | 10-step | 50-step |
| --- | ---: | ---: | ---: |
| constant_velocity | 4388.3661 | 88811.538 | 851484.68 |
| linear | 4388.3661 | 88811.538 | 851484.69 |
| mlp | 4388.3661 | 88811.538 | 851484.69 |
| gru | 4494.4887 | 87480.296 | 645407.95 |
| gnn | 4388.3661 | 88811.538 | 851484.68 |

## body_count

Five or six bodies. Training worlds use the default count range of two to four.

| Model | 1-step | 10-step | 50-step |
| --- | ---: | ---: | ---: |
| constant_velocity | 131.22648 | 2655.742 | 28834.049 |
| linear | 131.22648 | 2655.742 | 28834.049 |
| mlp | 131.22649 | 2655.7429 | 28834.066 |
| gru | 134.39994 | 2724.2253 | 25630.333 |
| gnn | 131.22649 | 2655.7431 | 28834.059 |

## mass

Masses sampled from [6, 12), above the training prior [0.5, 5).

| Model | 1-step | 10-step | 50-step |
| --- | ---: | ---: | ---: |
| constant_velocity | 0.92637708 | 14.711215 | 111.42564 |
| linear | 0.92637709 | 14.711218 | 111.4258 |
| mlp | 0.92637708 | 14.711293 | 111.43029 |
| gru | 0.49466412 | 8.810714 | 59.861506 |
| gnn | 0.92637463 | 14.711213 | 111.43425 |

## velocity

Each initial velocity component is sampled from [1.2, 2.2), outside the training prior [-0.5, 0.5).

| Model | 1-step | 10-step | 50-step |
| --- | ---: | ---: | ---: |
| constant_velocity | 1.2715087 | 25.042967 | 199.15522 |
| linear | 1.2715085 | 25.042938 | 199.15412 |
| mlp | 1.2715086 | 25.042977 | 199.15997 |
| gru | 0.39704443 | 7.9944539 | 87.144946 |
| gnn | 1.2715094 | 25.043041 | 199.16726 |

## long_horizon

New universes from the training prior, integrated for twice as long.

| Model | 1-step | 50-step | 100-step | 150-step |
| --- | ---: | ---: | ---: | ---: |
| constant_velocity | 0.0054145651 | 2.4643903 | 5.1190694 | 7.9637502 |
| linear | 0.0054145606 | 2.46487 | 5.1196406 | 7.9699083 |
| mlp | 0.0053900929 | 2.3935566 | 4.7987559 | 7.1832629 |
| gru | 0.0053370055 | 2.2309452 | 4.4080337 | 6.4778388 |
| gnn | 0.0053217467 | 2.2993807 | 4.5955577 | 6.8926189 |
