# Taguchi parameter study

The paper configuration was selected with a Taguchi `L9(3^3)` study over 12 calibration instances.

- Hidden width: `2, 4, 8`
- Population size: `10, 20, 30`
- Parameter bound: `2, 4, 6`
- Response: smaller-the-better signal-to-noise ratio of the optimality gap
- Evaluation budget: `50 × TargetNum`

The quality-oriented main effects favor `H=8`, `N=30`, and `kappa=4`. Because increasing `H` from 4 to 8 raises the mean GPU optimization time from 42.78 s to 53.96 s, the paper adopts `H=4` as a quality--time compromise. The public runner therefore uses `H=4`, `N=30`, and `kappa=4`, corresponding to a 3-4-1 MLP with 20 parameters searched in `[-4,4]^20`.
