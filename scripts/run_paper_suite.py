#!/usr/bin/env python
"""Run one or more of the 36 paper instances with the reported configuration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIR = REPOSITORY_ROOT / "benchmarks" / "paper_36"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "paper_36"
DEFAULT_SEED = 2026091101


def parse_case_specification(specification: str) -> list[int]:
    """Parse values such as ``1``, ``1,3,5`` or ``1-12,25-30``."""
    selected: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    if not selected:
        raise ValueError("No cases were selected")
    invalid = sorted(case for case in selected if not 1 <= case <= 36)
    if invalid:
        raise ValueError(f"Paper case numbers must be in 1..36: {invalid}")
    return sorted(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public 36-instance paper benchmark sequentially."
    )
    parser.add_argument(
        "--cases",
        default="1-36",
        help="Paper case numbers, for example 1, 1-12, or 1-12,25-30",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Independent runs per instance (use 25 for the paper protocol)",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device")
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA Graph replay for diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    cases = parse_case_specification(args.cases)
    single_runner = REPOSITORY_ROOT / "scripts" / "run_paper_configuration.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(cases) * args.runs
    completed = 0
    for case_number in cases:
        instance = args.benchmark_dir / f"case_{case_number:02d}.mat"
        if not instance.is_file():
            raise FileNotFoundError(f"Missing benchmark instance: {instance}")
        case_output_dir = args.output_dir / f"case_{case_number:02d}"
        case_output_dir.mkdir(parents=True, exist_ok=True)

        for run_index in range(1, args.runs + 1):
            seed = args.seed_base + run_index - 1
            output = case_output_dir / f"run_{run_index:02d}_seed_{seed}.json"
            command = [
                sys.executable,
                str(single_runner),
                str(instance),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--output",
                str(output),
            ]
            if args.disable_cuda_graph:
                command.append("--disable-cuda-graph")
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            completed += 1
            print(
                f"[{completed}/{total}] case={case_number:02d} "
                f"run={run_index:02d} seed={seed}"
            )

    print(f"Completed {completed} runs. Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
