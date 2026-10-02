# Milestone 3 law recovery

Seeds per world: 5. Generations: 25, population 160, restarts 2.
Recovered means identical term structure, every coefficient within 10% relative error, and every exponent within 0.1 absolute error.
RMSE is in force units on fresh experiments; extrapolation uses larger masses, separations, and speeds than training.

| World | Method | Structural | Recovered | Test RMSE | Extrap. RMSE | Complexity | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inverse_cube_k4 | selected | 1.00 | 1.00 | 2.9e-16 | 2.14e-16 | 7 | 10.3 |
| inverse_cube_k4 | log_linear | 1.00 | 1.00 | 8.35e-16 | 1.81e-15 | 11 | 0.0 |
| inverse_cube_k4 | polynomial | 0.00 | 0.00 | 0.285 | 1.64 | 166 | 0.0 |
| inverse_cube_k4 | power_sum | 1.00 | 1.00 | 2.9e-16 | 2.14e-16 | 11 | 0.3 |
| inverse_cube_k4 | sindy | 0.40 | 0.40 | 8.17e-06 | 0.000364 | 50 | 0.0 |
| inverse_cube_k4 | symbolic | 1.00 | 1.00 | 2.9e-16 | 2.14e-16 | 7 | 10.3 |
| inverse_power_k1.5_p2.6 | selected | 1.00 | 1.00 | 1.35e-16 | 1.14e-16 | 7 | 10.4 |
| inverse_power_k1.5_p2.6 | log_linear | 1.00 | 1.00 | 6.95e-16 | 1.02e-15 | 11 | 0.0 |
| inverse_power_k1.5_p2.6 | polynomial | 0.00 | 0.00 | 0.0392 | 1.35 | 157 | 0.0 |
| inverse_power_k1.5_p2.6 | power_sum | 1.00 | 1.00 | 1.35e-16 | 1.14e-16 | 11 | 0.3 |
| inverse_power_k1.5_p2.6 | sindy | 0.00 | 0.00 | 0.000138 | 0.000169 | 22 | 0.0 |
| inverse_power_k1.5_p2.6 | symbolic | 1.00 | 1.00 | 1.35e-16 | 1.14e-16 | 7 | 10.4 |
| inverse_power_k3_p1.5 | selected | 1.00 | 1.00 | 6.05e-16 | 1.3e-15 | 7 | 11.2 |
| inverse_power_k3_p1.5 | log_linear | 1.00 | 1.00 | 3.82e-15 | 2.48e-14 | 11 | 0.0 |
| inverse_power_k3_p1.5 | polynomial | 0.00 | 0.00 | 0.643 | 4.01 | 166 | 0.0 |
| inverse_power_k3_p1.5 | power_sum | 1.00 | 1.00 | 6.05e-16 | 1.3e-15 | 11 | 0.2 |
| inverse_power_k3_p1.5 | sindy | 0.60 | 0.60 | 4.64e-08 | 8.34e-08 | 7 | 0.0 |
| inverse_power_k3_p1.5 | symbolic | 1.00 | 1.00 | 6.05e-16 | 1.3e-15 | 7 | 11.2 |
| inverse_square_k2.5 | selected | 1.00 | 1.00 | 2.78e-16 | 4.64e-16 | 7 | 11.0 |
| inverse_square_k2.5 | log_linear | 1.00 | 1.00 | 4.21e-16 | 4.64e-16 | 11 | 0.0 |
| inverse_square_k2.5 | polynomial | 0.00 | 0.00 | 0.215 | 3.05 | 156 | 0.0 |
| inverse_square_k2.5 | power_sum | 1.00 | 1.00 | 2.75e-16 | 4.33e-16 | 11 | 0.3 |
| inverse_square_k2.5 | sindy | 0.60 | 0.60 | 1.91e-08 | 2.72e-08 | 7 | 0.0 |
| inverse_square_k2.5 | symbolic | 1.00 | 1.00 | 2.75e-16 | 4.33e-16 | 7 | 11.0 |
| linear_drag_c0.6 | selected | 1.00 | 1.00 | 1.55e-16 | 3.3e-16 | 3 | 3.8 |
| linear_drag_c0.6 | log_linear | 1.00 | 1.00 | 1.6e-16 | 3.64e-16 | 5 | 0.0 |
| linear_drag_c0.6 | polynomial | 1.00 | 1.00 | 3.69e-09 | 8.61e-09 | 3 | 0.0 |
| linear_drag_c0.6 | power_sum | 1.00 | 1.00 | 1.55e-16 | 3.55e-16 | 5 | 0.0 |
| linear_drag_c0.6 | sindy | 1.00 | 1.00 | 1.89e-09 | 4.4e-09 | 3 | 0.0 |
| linear_drag_c0.6 | symbolic | 1.00 | 1.00 | 1.55e-16 | 3.3e-16 | 3 | 3.8 |
| power_plus_drag | selected | 1.00 | 1.00 | 2.18e-16 | 3.22e-16 | 13 | 10.7 |
| power_plus_drag | log_linear | 0.00 | 0.00 | 0.135 | 0.32 | 11 | 0.0 |
| power_plus_drag | polynomial | 0.00 | 0.00 | 0.149 | 1.71 | 166 | 0.0 |
| power_plus_drag | power_sum | 1.00 | 1.00 | 2.18e-16 | 3.22e-16 | 15 | 1.2 |
| power_plus_drag | sindy | 0.00 | 0.00 | 2.53e-05 | 0.000644 | 60 | 0.0 |
| power_plus_drag | symbolic | 0.60 | 0.60 | 2.99e-16 | 4.25e-16 | 11 | 13.2 |
| quadratic_drag_c0.3 | selected | 1.00 | 1.00 | 1.49e-16 | 6.1e-16 | 5 | 6.3 |
| quadratic_drag_c0.3 | log_linear | 1.00 | 1.00 | 1.49e-16 | 6.1e-16 | 5 | 0.0 |
| quadratic_drag_c0.3 | polynomial | 1.00 | 1.00 | 8.76e-11 | 3.54e-10 | 5 | 0.0 |
| quadratic_drag_c0.3 | power_sum | 1.00 | 1.00 | 1.49e-16 | 6.1e-16 | 5 | 0.0 |
| quadratic_drag_c0.3 | sindy | 1.00 | 1.00 | 1.45e-09 | 5.69e-09 | 5 | 0.0 |
| quadratic_drag_c0.3 | symbolic | 1.00 | 1.00 | 1.49e-16 | 6.1e-16 | 5 | 6.3 |
| spring_k2_L1.5 | selected | 1.00 | 1.00 | 1.2e-16 | 4.51e-16 | 5 | 5.7 |
| spring_k2_L1.5 | polynomial | 1.00 | 1.00 | 1.81e-07 | 1.19e-07 | 5 | 0.0 |
| spring_k2_L1.5 | power_sum | 0.80 | 0.80 | 1.38e-15 | 1.55e-15 | 7 | 4.1 |
| spring_k2_L1.5 | sindy | 0.00 | 0.00 | 1.89e-05 | 0.00619 | 206 | 0.0 |
| spring_k2_L1.5 | symbolic | 1.00 | 1.00 | 1.2e-16 | 4.51e-16 | 5 | 5.7 |

## Selected law per run

- inverse_cube_k4 seed 200: `4.0*m1*m2**1.0/r**3.0` via symbolic (recovered=True)
- inverse_cube_k4 seed 201: `4.0*m1*m2/r**3.0` via symbolic (recovered=True)
- inverse_cube_k4 seed 202: `4.0*m1*m2/r**3.0` via symbolic (recovered=True)
- inverse_cube_k4 seed 203: `4.0*m1*m2/r**3.0` via symbolic (recovered=True)
- inverse_cube_k4 seed 204: `4.0*m1*m2/r**3.0` via symbolic (recovered=True)
- inverse_square_k2.5 seed 200: `2.5*m1*m2/r**2` via symbolic (recovered=True)
- inverse_square_k2.5 seed 201: `2.5*m1*m2/r**2.0` via symbolic (recovered=True)
- inverse_square_k2.5 seed 202: `2.5*m1*m2/r**2.0` via sindy (recovered=True)
- inverse_square_k2.5 seed 203: `2.5*m1*m2/r**2.0` via symbolic (recovered=True)
- inverse_square_k2.5 seed 204: `2.5*m1*m2/r**2.0` via symbolic (recovered=True)
- inverse_power_k3_p1.5 seed 200: `3.0*m1*m2/r**1.5` via symbolic (recovered=True)
- inverse_power_k3_p1.5 seed 201: `3.0*m1*m2/r**1.5` via symbolic (recovered=True)
- inverse_power_k3_p1.5 seed 202: `3.0*m1*m2/r**1.5` via symbolic (recovered=True)
- inverse_power_k3_p1.5 seed 203: `3.0*m1*m2/r**1.5` via symbolic (recovered=True)
- inverse_power_k3_p1.5 seed 204: `3.0*m1*m2/r**1.5` via symbolic (recovered=True)
- inverse_power_k1.5_p2.6 seed 200: `1.5*m1*m2/r**2.6` via symbolic (recovered=True)
- inverse_power_k1.5_p2.6 seed 201: `1.5*m1*m2/r**2.6` via symbolic (recovered=True)
- inverse_power_k1.5_p2.6 seed 202: `1.5*m1*m2/r**2.6` via symbolic (recovered=True)
- inverse_power_k1.5_p2.6 seed 203: `1.5*m1*m2/r**2.6` via symbolic (recovered=True)
- inverse_power_k1.5_p2.6 seed 204: `1.5*m1*m2/r**2.6` via symbolic (recovered=True)
- linear_drag_c0.6 seed 200: `-0.6*s` via symbolic (recovered=True)
- linear_drag_c0.6 seed 201: `-0.6*s` via symbolic (recovered=True)
- linear_drag_c0.6 seed 202: `-0.6*s` via symbolic (recovered=True)
- linear_drag_c0.6 seed 203: `-0.6*s` via symbolic (recovered=True)
- linear_drag_c0.6 seed 204: `-0.6*s` via symbolic (recovered=True)
- quadratic_drag_c0.3 seed 200: `-0.3*s**2` via symbolic (recovered=True)
- quadratic_drag_c0.3 seed 201: `-0.3*s**2` via symbolic (recovered=True)
- quadratic_drag_c0.3 seed 202: `-0.3*s**2` via symbolic (recovered=True)
- quadratic_drag_c0.3 seed 203: `-0.3*s**2` via symbolic (recovered=True)
- quadratic_drag_c0.3 seed 204: `-0.3*s**2` via symbolic (recovered=True)
- spring_k2_L1.5 seed 200: `2.0*r - 3.0` via symbolic (recovered=True)
- spring_k2_L1.5 seed 201: `2.0*r - 3.0` via symbolic (recovered=True)
- spring_k2_L1.5 seed 202: `2.0*r - 3.0` via symbolic (recovered=True)
- spring_k2_L1.5 seed 203: `2.0*r - 3.0` via symbolic (recovered=True)
- spring_k2_L1.5 seed 204: `2.0*r - 3.0` via symbolic (recovered=True)
- power_plus_drag seed 200: `2.0*m1*m2/r**2.0 - 0.5*vr` via symbolic (recovered=True)
- power_plus_drag seed 201: `2.0*m1*m2**1.0/r**2.0 - 0.5*vr` via symbolic (recovered=True)
- power_plus_drag seed 202: `2.0*m1*m2/r**2.0 - 0.5*vr` via symbolic (recovered=True)
- power_plus_drag seed 203: `2.0*m1**1.0*m2**1.0/r**2.0 - 0.5*vr` via power_sum (recovered=True)
- power_plus_drag seed 204: `2.0*m1**1.0*m2**1.0/r**2.0 - 0.5*vr` via power_sum (recovered=True)
