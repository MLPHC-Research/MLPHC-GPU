# MLPHC-GPU

This repository provides the reproducible implementation of MLPHC-GPU from the paper **Tensorized GPU-Accelerated Hyper-Construction for Large-Scale Sensor--Interceptor--Target Assignment With Time Windows**. The implementation tensorizes candidate-quad scoring, constraint-state updates, feasibility masking, HCLPSO population updates, and global-best reduction with PyTorch.

> **Paper benchmark data:** [Download the 36 SITA-TW instances from the GitHub Release](https://github.com/MaWeijie0908/MLPHC-GPU/releases/tag/benchmarks-v1.0).

## Paper configuration

`scripts/run_paper_configuration.py` uses the configuration selected by the Taguchi experiment in the paper.

| Setting | Default |
|---|---:|
| MLP architecture | 3-8-1 |
| Hidden units | 8 |
| MLP parameter dimension | 40 |
| Parameter range | `[-2, 2]^40` |
| HCLPSO population size | 20 |
| Exploration/exploitation subpopulations | 8 / 12 |
| Maximum function evaluations | `50 × TargetNum` |
| Exemplar refresh | After six consecutive non-improving generations |
| Numerical precision | `float64` |

The defaults are defined in `mlphc_gpu/paper_config.py`. Command-line options support diagnostic and ablation runs, but no overrides are required to reproduce the paper configuration.

## Environment

The reported experiments used Windows 11, an NVIDIA RTX 4090, CUDA 11.8, Python 3.9, PyTorch 2.0, and NumPy 1.23. The implementation also runs in other PyTorch 2.x environments with a compatible CUDA installation.

Install the PyTorch build that matches your CUDA environment by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install this project.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Quick validation

Generate a small demonstration instance and run MLPHC-GPU with the paper defaults.

```bash
python scripts/generate_demo_instance.py
python scripts/run_paper_configuration.py examples/demo_instance.mat
```

The result is written to `results/demo_instance_paper_configuration.json`. It contains the best objective value, the 40 MLP parameters, the selected quads, convergence records, the evaluation count, and the optimization time.

## Paper benchmark

1. Open the [benchmark Release](https://github.com/MaWeijie0908/MLPHC-GPU/releases/tag/benchmarks-v1.0).
2. Download `MLPHC-GPU-paper-36-instances.zip`.
3. Extract the archive into the repository root. The instances will be placed in `benchmarks/paper_36/`.

Run one instance with the paper configuration.

```bash
python scripts/run_paper_configuration.py benchmarks/paper_36/case_01.mat
```

Run all 36 instances once.

```bash
python scripts/run_paper_suite.py --cases 1-36 --runs 1
```

Run 25 independent trials per instance, following the experimental protocol in the paper.

```bash
python scripts/run_paper_suite.py --cases 1-36 --runs 25
```

The case selector accepts individual numbers, comma-separated lists, and ranges. For example, the following command runs Cases 20, 22, and 24.

```bash
python scripts/run_paper_suite.py --cases 20,22,24 --runs 25
```

The runner uses `cuda` by default and assigns an independent random seed to each trial. Results are stored under `results/paper_36/case_XX/`. Each JSON file records the best objective value, MLP parameters, selected quads, convergence history, evaluation count, and optimization time.

For a CPU-only functional check, use:

```bash
python scripts/run_paper_configuration.py examples/demo_instance.mat --device cpu --disable-cuda-graph
```

The MATLAB `.mat` input format is documented in `benchmarks/paper_36/README.md`. The complete case mapping, instance dimensions, candidate-quad counts, reference types, and SHA-256 checksums are reported in `benchmarks/paper_36/manifest.csv`.

## Repository structure

- `mlphc_gpu/hclpso_gpu.py`: tensorized HCLPSO.
- `mlphc_gpu/optimized_dynamic_mlp_gpu.py`: the 3-8-1 MLP scoring model.
- `mlphc_gpu/optimized_dynamic_mmrra_rbf_gpu.py`: static-shape GPU construction and CUDA Graph execution.
- `mlphc_gpu/dynamic_mmrra_rbf_gpu.py`: candidate quads, time windows, and constraint tensors.
- `mlphc_gpu/instance_io.py`: MATLAB instance loading.
- `benchmarks/paper_36/`: the 36 paper instances, documentation, and manifest after extracting the Release archive.
- `scripts/run_paper_suite.py`: batch runner for selected paper cases.
- `taguchi/`: Taguchi `L9(3^3)` settings and summary results.
- `tests/`: configuration and runner tests.

## Tests

```bash
python -m unittest discover -s tests
```

## Method overview

Each HCLPSO particle encodes the 40 parameters of a 3-8-1 MLP construction rule. For all particles in a generation, the GPU evaluates the three features and MLP scores of the candidate quads in batches. Each particle then selects its highest-scoring feasible quad.

After each selection, indexed tensor operations update the remaining target assignments, interceptor ammunition, and sensor-channel capacity over time. A Boolean feasibility mask then disables quads that violate the updated constraints. This score-select-update-mask procedure continues until no feasible quad remains, producing a complete feasible solution and its objective value.

HCLPSO uses this objective value as particle fitness. Row-index tensors divide the exploration and exploitation subpopulations for parallel updates. Bound checks, personal-best updates, stagnation detection, and global-best reduction also run as tensor operations. Construction is parallel across particles, while the successive quad selections within each construction remain sequential because of their state dependence.

## Citation

If you use this implementation or the benchmark instances, please cite the associated paper. The final volume, page range, and DOI will be added after publication.
