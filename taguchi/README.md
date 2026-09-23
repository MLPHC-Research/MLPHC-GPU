# Taguchi parameter study

The paper configuration was selected with a Taguchi `L9(3^3)` study over 12 calibration instances.

- Hidden width: `2, 4, 8`
- Population size: `10, 20, 30`
- Parameter scale: `0.5, 1.0, 2.0`
- Response: smaller-the-better signal-to-noise ratio of the optimality gap
- Evaluation budget: `50 × TargetNum`

The configuration used by the public runner is `H=8`, `N=20`, parameter range `[-2,2]^40`.
