# Milestone 3 law recovery

Seeds per world: 3. Generations: 25, population 160, restarts 2.
Recovered means identical term structure, every coefficient within 10% relative error, and every exponent within 0.1 absolute error.
RMSE is in force units on fresh experiments; extrapolation uses larger masses, separations, and speeds than training.

| World | Method | Structural | Recovered | Test RMSE | Extrap. RMSE | Complexity | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inverse_cube_k4 | selected | 1.00 | 1.00 | 1.89e-16 | 2.8e-16 | 9 | 13.6 |
| inverse_cube_k4 | log_linear | 1.00 | 1.00 | 1.48e-15 | 2.31e-15 | 11 | 0.0 |
| inverse_cube_k4 | polynomial | 0.00 | 0.00 | 0.213 | 2.01 | 108 | 0.0 |
| inverse_cube_k4 | power_sum | 1.00 | 1.00 | 1.89e-16 | 2.8e-16 | 11 | 0.3 |
| inverse_cube_k4 | sindy | 0.00 | 0.00 | 3.14e-05 | 0.000367 | 16 | 0.0 |
| inverse_cube_k4 | symbolic | 1.00 | 1.00 | 1.89e-16 | 2.8e-16 | 9 | 13.6 |
| inverse_power_k1.5_p2.6 | selected | 1.00 | 1.00 | 2.25e-16 | 1.37e-16 | 9 | 14.0 |
| inverse_power_k1.5_p2.6 | log_linear | 1.00 | 1.00 | 3e-15 | 5.63e-15 | 11 | 0.0 |
| inverse_power_k1.5_p2.6 | polynomial | 0.00 | 0.00 | 0.661 | 4.55 | 108 | 0.0 |
| inverse_power_k1.5_p2.6 | power_sum | 1.00 | 1.00 | 3.08e-16 | 1.37e-16 | 11 | 0.4 |
| inverse_power_k1.5_p2.6 | sindy | 0.00 | 0.00 | 0.000109 | 0.000186 | 12 | 0.0 |
| inverse_power_k1.5_p2.6 | symbolic | 1.00 | 1.00 | 2.25e-16 | 1.37e-16 | 9 | 14.0 |
| inverse_power_k3_p1.5 | selected | 1.00 | 1.00 | 6.81e-16 | 9.95e-16 | 9 | 13.8 |
| inverse_power_k3_p1.5 | log_linear | 1.00 | 1.00 | 2.3e-15 | 8.96e-15 | 11 | 0.0 |
| inverse_power_k3_p1.5 | polynomial | 0.00 | 0.00 | 1.7 | 10.9 | 108 | 0.0 |
| inverse_power_k3_p1.5 | power_sum | 1.00 | 1.00 | 6.81e-16 | 9.95e-16 | 11 | 0.2 |
| inverse_power_k3_p1.5 | sindy | 0.00 | 0.00 | 9.54e-05 | 0.00197 | 20 | 0.0 |
| inverse_power_k3_p1.5 | symbolic | 1.00 | 1.00 | 6.81e-16 | 9.95e-16 | 9 | 13.8 |
| inverse_square_k2.5 | selected | 1.00 | 1.00 | 5.73e-16 | 5.85e-16 | 9 | 13.8 |
| inverse_square_k2.5 | log_linear | 1.00 | 1.00 | 1.32e-15 | 1.58e-15 | 11 | 0.0 |
| inverse_square_k2.5 | polynomial | 0.00 | 0.00 | 0.402 | 3.35 | 108 | 0.0 |
| inverse_square_k2.5 | power_sum | 1.00 | 1.00 | 5.73e-16 | 5.85e-16 | 11 | 0.3 |
| inverse_square_k2.5 | sindy | 0.00 | 0.00 | 0.00194 | 0.000838 | 28 | 0.0 |
| inverse_square_k2.5 | symbolic | 1.00 | 1.00 | 5.73e-16 | 5.85e-16 | 9 | 13.8 |
| linear_drag_c0.6 | selected | 1.00 | 1.00 | 1.54e-16 | 3.44e-16 | 3 | 5.7 |
| linear_drag_c0.6 | log_linear | 1.00 | 1.00 | 1.57e-16 | 3.76e-16 | 8 | 0.0 |
| linear_drag_c0.6 | polynomial | 1.00 | 1.00 | 2.25e-09 | 4.48e-09 | 45 | 0.0 |
| linear_drag_c0.6 | power_sum | 1.00 | 1.00 | 5.64e-16 | 5.6e-15 | 8 | 0.0 |
| linear_drag_c0.6 | sindy | 1.00 | 1.00 | 1.58e-09 | 4.19e-09 | 4 | 0.0 |
| linear_drag_c0.6 | symbolic | 1.00 | 1.00 | 1.54e-16 | 3.44e-16 | 3 | 5.7 |
| power_plus_drag | selected | 0.67 | 0.67 | 4.7e-16 | 4.39e-16 | 18 | 5.6 |
| power_plus_drag | log_linear | 0.00 | 0.00 | 0.247 | 0.282 | 11 | 0.0 |
| power_plus_drag | polynomial | 0.00 | 0.00 | 0.597 | 1.8 | 108 | 0.0 |
| power_plus_drag | power_sum | 0.67 | 0.67 | 4.7e-16 | 4.39e-16 | 18 | 4.0 |
| power_plus_drag | sindy | 0.00 | 0.00 | 0.000584 | 0.000924 | 24 | 0.0 |
| power_plus_drag | symbolic | 0.00 | 0.00 | 0.114 | 0.254 | 15 | 35.8 |
| quadratic_drag_c0.3 | selected | 1.00 | 1.00 | 1.15e-09 | 5.28e-09 | 4 | 0.0 |
| quadratic_drag_c0.3 | log_linear | 1.00 | 1.00 | 2.33e-16 | 1.34e-15 | 8 | 0.0 |
| quadratic_drag_c0.3 | polynomial | 1.00 | 1.00 | 4.64e-11 | 2.95e-10 | 45 | 0.0 |
| quadratic_drag_c0.3 | power_sum | 1.00 | 1.00 | 2.39e-16 | 7.58e-16 | 8 | 0.0 |
| quadratic_drag_c0.3 | sindy | 1.00 | 1.00 | 1.15e-09 | 5.28e-09 | 4 | 0.0 |
| quadratic_drag_c0.3 | symbolic | 1.00 | 1.00 | 1.68e-16 | 5.89e-16 | 5 | 7.2 |
| spring_k2_L1.5 | selected | 1.00 | 1.00 | 1.27e-16 | 4.64e-16 | 5 | 7.7 |
| spring_k2_L1.5 | polynomial | 1.00 | 1.00 | 1.56e-06 | 8.65e-07 | 108 | 0.0 |
| spring_k2_L1.5 | power_sum | 0.33 | 0.33 | 0.676 | 69.8 | 13 | 4.1 |
| spring_k2_L1.5 | sindy | 0.00 | 0.00 | 0.000421 | 0.012 | 164 | 0.0 |
| spring_k2_L1.5 | symbolic | 1.00 | 1.00 | 1.27e-16 | 4.64e-16 | 5 | 7.7 |

## Selected law per run

- inverse_cube_k4 seed 0: `4.0*m1*m2/r**3.0` via symbolic (recovered=True)
- inverse_cube_k4 seed 1: `4.0*m1*m2/r**3.0` via symbolic (recovered=True)
- inverse_cube_k4 seed 2: `4.0*m1*m2/r**3.0` via symbolic (recovered=True)
- inverse_square_k2.5 seed 0: `2.5*m1*m2/r**2.0` via symbolic (recovered=True)
- inverse_square_k2.5 seed 1: `2.5*m1*m2/r**2.0` via symbolic (recovered=True)
- inverse_square_k2.5 seed 2: `2.5*m1*m2/r**2.0` via symbolic (recovered=True)
- inverse_power_k3_p1.5 seed 0: `3.0*m1**1.0*m2/r**1.5` via symbolic (recovered=True)
- inverse_power_k3_p1.5 seed 1: `3.0*m1*m2/r**1.5` via symbolic (recovered=True)
- inverse_power_k3_p1.5 seed 2: `3.0*m1*m2/r**1.5` via symbolic (recovered=True)
- inverse_power_k1.5_p2.6 seed 0: `1.5*m1*m2/r**2.6` via symbolic (recovered=True)
- inverse_power_k1.5_p2.6 seed 1: `1.5*m1*m2/r**2.6` via symbolic (recovered=True)
- inverse_power_k1.5_p2.6 seed 2: `1.5*m1*m2/r**2.6` via symbolic (recovered=True)
- linear_drag_c0.6 seed 0: `-0.6*s` via symbolic (recovered=True)
- linear_drag_c0.6 seed 1: `-0.6*s` via symbolic (recovered=True)
- linear_drag_c0.6 seed 2: `-0.6*s` via symbolic (recovered=True)
- quadratic_drag_c0.3 seed 0: `-0.3*s**2` via sindy (recovered=True)
- quadratic_drag_c0.3 seed 1: `-0.3*s**2` via sindy (recovered=True)
- quadratic_drag_c0.3 seed 2: `-0.3*s**2` via sindy (recovered=True)
- spring_k2_L1.5 seed 0: `2.0*r - 3.0` via symbolic (recovered=True)
- spring_k2_L1.5 seed 1: `2*r - 3.0` via symbolic (recovered=True)
- spring_k2_L1.5 seed 2: `2.0*r - 3.0` via symbolic (recovered=True)
- power_plus_drag seed 0: `2.0*m1**1.0*m2**1.0/r**2.0 - 0.5*vr/r**1.83e-16` via power_sum (recovered=True)
- power_plus_drag seed 1: `1.784*m1*m2/r**1.996` via symbolic (recovered=False)
- power_plus_drag seed 2: `-0.5*vr/(m1**1.824e-16*m2**5.9e-16*r**2.919e-16) + 2.0*m1**1.0*m2**1.0/r**2.0` via power_sum (recovered=True)
