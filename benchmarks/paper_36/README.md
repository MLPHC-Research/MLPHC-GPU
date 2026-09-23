# The 36 SITA-TW Paper Instances

This directory contains the 36 instances used in the final experiments of **Tensorized GPU-Accelerated Hyper-Construction for Large-Scale Sensor--Interceptor--Target Assignment With Time Windows**. Files `case_01.mat` through `case_36.mat` correspond directly to Cases 1 through 36 in the paper.

## Instance groups

| Paper cases | Source and setting |
|---|---|
| Cases 1-12 | SITA-TW time windows derived from the 12 MOP5 benchmark instances, with `m_j=1` |
| Cases 13-24 | Instances paired with Cases 1-12, with `m_j=2` |
| Cases 25-30 | Six high-conflict instances generated with the MOP5 instance generator, with `m_j=1` |
| Cases 31-36 | Instances paired with Cases 25-30, with `m_j=2` |

Here, `m_j` is the maximum number of interceptions assignable to target `j`. `CS` and `DS` denote centralized and decentralized layouts, respectively. The high-conflict instances use tighter sensor-channel capacities and longer guidance intervals to increase resource contention and temporal conflicts.

`manifest.csv` reports the paper case number, original global identifier, source family, layout, dimensions, candidate-quad count, reference type, and SHA-256 checksum for every file. `Reference=gurobi_optimal` indicates a Gurobi-certified optimum. `Reference=best_known` indicates the best-known reference value used in the paper.

## Data format

Each instance is stored as a MATLAB `.mat` file with the following main fields:

- `weapon_pr_matrix`: interceptor success probabilities;
- `weapon_vp_matrix`: value-weighted interceptor-target launch scores used to identify active launch candidates;
- `radar_pr_cell`: sensor detection probabilities;
- `radar_tbegin_cell` and `radar_tend_cell`: guidance start and end times;
- `danyao_constraint`: interceptor ammunition capacities;
- `strike_constraint`: target-specific interception limits `m_j`;
- `radar_constraint`: concurrent guidance-channel capacities of the sensors;
- `V_matrix`: target values.

## Usage

Run one instance from the repository root.

```bash
python scripts/run_paper_configuration.py benchmarks/paper_36/case_01.mat
```

Run all 36 instances once.

```bash
python scripts/run_paper_suite.py --cases 1-36 --runs 1
```

Run 25 independent trials per instance, following the paper protocol.

```bash
python scripts/run_paper_suite.py --cases 1-36 --runs 25
```

Run a selected subset, such as Cases 20, 22, and 24.

```bash
python scripts/run_paper_suite.py --cases 20,22,24 --runs 25
```

The default configuration uses a GPU, a 3-8-1 MLP, 20 particles, and an evaluation budget of `50 × TargetNum`. Results are organized by case and independent trial under `results/paper_36/`. Add `--device cpu --disable-cuda-graph` only for a CPU functional check.
