#!/usr/bin/env python
"""Run MLPHC-GPU with the parameter setting reported in the paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mlphc_gpu import (  # noqa: E402
    HCLPSOConfig,
    OptimizedDynamicMLPGPUEvaluator,
    PAPER_CONFIG,
    run_hclpso,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tensorized GPU HCLPSO optimization of the paper's 3-4-1 MLP rule."
    )
    parser.add_argument("instance", type=Path, help="SITA-TW MATLAB instance (.mat)")
    parser.add_argument("--output", type=Path, help="JSON result path")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    parser.add_argument("--seed", type=int, default=PAPER_CONFIG.seed)
    parser.add_argument("--hidden-neurons", type=int, default=PAPER_CONFIG.hidden_neurons)
    parser.add_argument("--population-size", type=int, default=PAPER_CONFIG.population_size)
    parser.add_argument("--parameter-scale", type=float, default=PAPER_CONFIG.parameter_scale)
    parser.add_argument(
        "--evaluations-per-target",
        type=int,
        default=PAPER_CONFIG.evaluations_per_target,
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        help="Optional explicit budget; otherwise 50 times the target count",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA Graph replay for diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    evaluator = OptimizedDynamicMLPGPUEvaluator(
        args.instance,
        device=str(device),
        mlp_hidden_neurons=args.hidden_neurons,
        parameter_scale=args.parameter_scale,
        use_cuda_graph=not args.disable_cuda_graph,
        dtype=torch.float64,
    )
    max_evaluations = (
        int(args.max_evaluations)
        if args.max_evaluations is not None
        else int(args.evaluations_per_target) * evaluator.inst.target_num
    )
    config = HCLPSOConfig(
        population_size=args.population_size,
        max_evaluations=max_evaluations,
        schedule_evaluations=max_evaluations,
        seed=args.seed,
        exploration_fraction=PAPER_CONFIG.exploration_fraction,
        refresh_gap=PAPER_CONFIG.refresh_gap,
        velocity_fraction=PAPER_CONFIG.velocity_fraction,
    )

    def evaluate_population(
        population: torch.Tensor, *, record_selection: bool = False
    ):
        """Keep a fixed CUDA Graph batch while masking padded particle rows."""
        active_count = int(population.shape[0])
        if active_count > args.population_size:
            raise ValueError("Objective batch exceeds the configured population size")
        if active_count == args.population_size:
            padded = population
        else:
            padded = torch.zeros(
                (args.population_size, population.shape[1]),
                device=population.device,
                dtype=population.dtype,
            )
            padded[:active_count] = population
        enabled = torch.arange(
            args.population_size, device=population.device
        ) < active_count
        decoded_population = evaluator.evaluate(
            padded,
            record_selection=record_selection,
            rule_enabled=enabled,
        )
        return decoded_population, active_count

    result = run_hclpso(
        objective=lambda population: evaluate_population(population)[0].fitness[
            : population.shape[0]
        ],
        dimension=evaluator.rule_dimension,
        device=device,
        config=config,
        population_initializer=evaluator.initialize_search_population,
    )
    best_normalized = result["best_rule_tensor"].detach()
    best_signed = (2.0 * best_normalized - 1.0) * args.parameter_scale
    decoded, _ = evaluate_population(
        best_normalized[None, :], record_selection=True
    )

    payload = {
        "paper_configuration": {
            "architecture": f"3-{args.hidden_neurons}-1",
            "hidden_neurons": args.hidden_neurons,
            "rule_dimension": evaluator.rule_dimension,
            "population_size": args.population_size,
            "parameter_range": [-args.parameter_scale, args.parameter_scale],
            "evaluation_budget": max_evaluations,
            "evaluation_budget_rule": f"{args.evaluations_per_target}*TargetNum",
            "seed": args.seed,
        },
        "instance": str(args.instance.resolve()),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "evaluator": evaluator.metadata(),
        "optimization": result,
        "best_rule_parameters": best_signed,
        "best_selected_count": int(decoded.selected_count[0].item()),
        "best_selection_history": decoded.selection_history[0],
    }
    output = args.output or (
        REPOSITORY_ROOT / "results" / f"{args.instance.stem}_paper_configuration.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Best objective: {result['best_fitness']:.10f}")
    print(f"Evaluations: {result['evaluation_count']}")
    print(f"Optimization time: {result['optimization_sec']:.6f} s")
    print(f"Result: {output.resolve()}")


if __name__ == "__main__":
    main()
