"""Tensorized GPU HCLPSO for bounded black-box maximization.

The update equations and exemplar construction follow the authors' MATLAB
implementation ``HCLPSO_PS_15_35.m`` released with Lynn and Suganthan (2015):

* the swarm is split into exploration and exploitation subpopulations;
* both groups use dimension-wise comprehensive-learning exemplars;
* the exploration group learns only from its exemplars;
* the exploitation group additionally learns from the global best;
* exemplars are refreshed after six consecutive non-improving generations;
* linearly varying inertia and acceleration coefficients are retained.

The original code uses 15 exploration and 25 exploitation particles.  When a
different population size is requested, this implementation preserves the
15:25 ratio by default.  Objective calls are batched on the selected device,
while every candidate row is counted as one function evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Callable

import torch


BatchObjective = Callable[[torch.Tensor], torch.Tensor]
PopulationInitializer = Callable[[torch.Tensor, torch.Generator], None]


@dataclass(frozen=True)
class HCLPSOConfig:
    population_size: int = 30
    max_evaluations: int = 5000
    schedule_evaluations: int | None = None
    seed: int = 2026091101
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    exploration_fraction: float = 15.0 / 40.0
    exploration_size: int | None = None
    refresh_gap: int = 5
    velocity_fraction: float = 0.2
    synchronize_timing: bool = True
    time_limit_seconds: float | None = None
    max_generations_without_evaluation: int = 10000


def _rand(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def _build_exemplars(
    particle_ids: torch.Tensor,
    friend_pool_size: int,
    pbest: torch.Tensor,
    pbest_fitness: torch.Tensor,
    learning_probability: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build HCLPSO's dimension-wise comprehensive-learning exemplars."""

    count = int(particle_ids.numel())
    dimension = int(pbest.shape[1])
    if count == 0:
        return torch.empty((0, dimension), device=pbest.device, dtype=pbest.dtype)

    friend1 = torch.randint(
        friend_pool_size,
        (count, dimension),
        device=pbest.device,
        generator=generator,
    )
    friend2 = torch.randint(
        friend_pool_size,
        (count, dimension),
        device=pbest.device,
        generator=generator,
    )
    friend = torch.where(
        pbest_fitness[friend1] > pbest_fitness[friend2], friend1, friend2
    )
    learn = _rand(
        (count, dimension),
        device=pbest.device,
        dtype=pbest.dtype,
        generator=generator,
    ) < learning_probability[particle_ids, None]

    # The source code forces at least one dimension to learn from a friend.
    empty_rows = torch.nonzero(~learn.any(dim=1), as_tuple=False).flatten()
    if empty_rows.numel():
        forced_dimensions = torch.randint(
            dimension,
            (int(empty_rows.numel()),),
            device=pbest.device,
            generator=generator,
        )
        learn[empty_rows, forced_dimensions] = True

    own = particle_ids[:, None].expand(-1, dimension)
    source = torch.where(learn, friend, own)
    dimensions = torch.arange(dimension, device=pbest.device)[None, :]
    return pbest[source, dimensions]


def run_hclpso(
    objective: BatchObjective,
    dimension: int,
    device: torch.device | str,
    config: HCLPSOConfig,
    population_initializer: PopulationInitializer | None = None,
) -> dict[str, object]:
    """Run HCLPSO on a bounded maximization objective.

    ``objective`` accepts an ``N x dimension`` tensor and returns one fitness
    value per row.  As in the official MATLAB implementation, positions that
    leave the prescribed bounds are not evaluated; their velocities continue
    to evolve and may bring them back into the feasible parameter box.
    """

    device = torch.device(device)
    dtype = torch.float64
    population_size = int(config.population_size)
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if population_size < 2:
        raise ValueError("HCLPSO requires at least two particles")
    if config.max_evaluations < population_size:
        raise ValueError("max_evaluations is too small for HCLPSO initialization")
    schedule_evaluations = (
        config.max_evaluations
        if config.schedule_evaluations is None
        else int(config.schedule_evaluations)
    )
    if schedule_evaluations < population_size:
        raise ValueError("schedule_evaluations must be at least population_size")
    if not config.lower_bound < config.upper_bound:
        raise ValueError("lower_bound must be less than upper_bound")
    if not 0.0 < config.exploration_fraction < 1.0:
        raise ValueError("exploration_fraction must be in (0, 1)")
    if config.refresh_gap < 0:
        raise ValueError("refresh_gap must be nonnegative")
    if not 0.0 < config.velocity_fraction <= 1.0:
        raise ValueError("velocity_fraction must be in (0, 1]")
    if config.time_limit_seconds is not None and config.time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive when provided")

    if config.exploration_size is None:
        exploration_size = math.floor(
            population_size * config.exploration_fraction + 0.5
        )
    else:
        exploration_size = int(config.exploration_size)
    exploration_size = max(1, min(population_size - 1, exploration_size))
    exploitation_size = population_size - exploration_size

    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    lower = torch.full(
        (1, dimension), config.lower_bound, device=device, dtype=dtype
    )
    upper = torch.full(
        (1, dimension), config.upper_bound, device=device, dtype=dtype
    )
    interval = upper - lower
    velocity_max = config.velocity_fraction * interval
    velocity_min = -velocity_max

    evaluations = 0
    evaluation_wall_sec = 0.0
    convergence: list[float] = []
    generation_rows: list[dict[str, float | int]] = []
    start = time.perf_counter()

    def time_expired() -> bool:
        return (
            config.time_limit_seconds is not None
            and time.perf_counter() - start >= config.time_limit_seconds
        )

    def evaluate(candidates: torch.Tensor) -> torch.Tensor:
        nonlocal evaluations, evaluation_wall_sec
        if candidates.ndim != 2 or candidates.shape[1] != dimension:
            raise ValueError("objective candidates have an invalid shape")
        remaining = config.max_evaluations - evaluations
        if remaining <= 0 or candidates.shape[0] == 0 or time_expired():
            return torch.empty(0, device=device, dtype=dtype)
        candidates = candidates[:remaining]
        if device.type == "cuda" and config.synchronize_timing:
            torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        values = objective(candidates)
        if device.type == "cuda" and config.synchronize_timing:
            torch.cuda.synchronize(device)
        evaluation_wall_sec += time.perf_counter() - wall_start
        values = values.to(device=device, dtype=dtype).reshape(-1).clone()
        if values.shape[0] != candidates.shape[0]:
            raise ValueError("objective returned an invalid number of values")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("objective returned a non-finite fitness value")
        evaluations += int(candidates.shape[0])
        return values

    positions = lower + interval * _rand(
        (population_size, dimension),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if population_initializer is not None:
        population_initializer(positions, generator)
    velocity = velocity_min + (velocity_max - velocity_min) * _rand(
        (population_size, dimension),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    fitness = evaluate(positions)
    if fitness.numel() != population_size:
        raise RuntimeError("HCLPSO initialization exhausted its budget")
    pbest = positions.clone()
    pbest_fitness = fitness.clone()
    best_index = torch.argmax(pbest_fitness)
    global_best = pbest[best_index].clone()
    global_best_fitness = pbest_fitness[best_index].clone()
    convergence.append(float(global_best_fitness.item()))
    generation_rows.append(
        {
            "generation": 0,
            "evaluations": evaluations,
            "evaluated_particles": population_size,
            "best_fitness": float(global_best_fitness.item()),
            "population_improvements": population_size,
            "optimization_elapsed_sec": time.perf_counter() - start,
            "best_rule_tensor": global_best.clone(),
        }
    )

    exploration_ids = torch.arange(exploration_size, device=device)
    exploitation_ids = torch.arange(
        exploration_size, population_size, device=device
    )
    stagnation = torch.zeros(population_size, device=device, dtype=torch.int64)

    # Exact probability curve from the authors' HCLPSO MATLAB source.
    probability_axis = torch.linspace(
        0.0, 10.0, population_size, device=device, dtype=dtype
    )
    learning_probability = 0.25 * (
        torch.exp(probability_axis) - torch.exp(probability_axis[0])
    ) / (torch.exp(probability_axis[-1]) - torch.exp(probability_axis[0]))

    exemplars = pbest.clone()
    exemplars[exploration_ids] = _build_exemplars(
        exploration_ids,
        exploration_size,
        pbest,
        pbest_fitness,
        learning_probability,
        generator,
    )
    exemplars[exploitation_ids] = _build_exemplars(
        exploitation_ids,
        population_size,
        pbest,
        pbest_fitness,
        learning_probability,
        generator,
    )

    generation = 0
    generations_without_evaluation = 0
    nominal_generations = max(1, math.ceil(schedule_evaluations / population_size))
    stop_reason = "evaluation_limit"

    while evaluations < config.max_evaluations and not time_expired():
        generation += 1
        schedule_step = min(generation, nominal_generations)
        progress = schedule_step / nominal_generations
        inertia = 0.99 - 0.79 * progress
        exploration_acceleration = 3.0 - 1.5 * progress
        cognitive_acceleration = 2.5 - 2.0 * progress
        social_acceleration = 0.5 + 2.0 * progress

        random_exploration = _rand(
            (exploration_size, dimension),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        velocity[exploration_ids] = (
            inertia * velocity[exploration_ids]
            + exploration_acceleration
            * random_exploration
            * (exemplars[exploration_ids] - positions[exploration_ids])
        ).clamp(velocity_min, velocity_max)
        positions[exploration_ids] += velocity[exploration_ids]

        random_cognitive = _rand(
            (exploitation_size, dimension),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        random_social = _rand(
            (exploitation_size, dimension),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        velocity[exploitation_ids] = (
            inertia * velocity[exploitation_ids]
            + cognitive_acceleration
            * random_cognitive
            * (exemplars[exploitation_ids] - positions[exploitation_ids])
            + social_acceleration
            * random_social
            * (global_best[None, :] - positions[exploitation_ids])
        ).clamp(velocity_min, velocity_max)
        positions[exploitation_ids] += velocity[exploitation_ids]

        in_bounds = ((positions >= lower) & (positions <= upper)).all(dim=1)
        valid_ids = torch.nonzero(in_bounds, as_tuple=False).flatten()
        remaining = config.max_evaluations - evaluations
        valid_ids = valid_ids[:remaining]
        values = evaluate(positions[valid_ids])
        evaluated_count = int(values.numel())
        if evaluated_count:
            generations_without_evaluation = 0
        else:
            generations_without_evaluation += 1

        improved = torch.zeros(population_size, device=device, dtype=torch.bool)
        if evaluated_count:
            evaluated_ids = valid_ids[:evaluated_count]
            better = values > pbest_fitness[evaluated_ids]
            improved[evaluated_ids] = better
            better_ids = evaluated_ids[better]
            if better_ids.numel():
                pbest[better_ids] = positions[better_ids]
                pbest_fitness[better_ids] = values[better]

        stagnation = torch.where(improved, torch.zeros_like(stagnation), stagnation + 1)
        best_index = torch.argmax(pbest_fitness)
        candidate_best_fitness = pbest_fitness[best_index]
        if bool(candidate_best_fitness > global_best_fitness):
            global_best = pbest[best_index].clone()
            global_best_fitness = candidate_best_fitness.clone()

        refresh_exploration = exploration_ids[
            stagnation[exploration_ids] > config.refresh_gap
        ]
        if refresh_exploration.numel():
            exemplars[refresh_exploration] = _build_exemplars(
                refresh_exploration,
                exploration_size,
                pbest,
                pbest_fitness,
                learning_probability,
                generator,
            )
            stagnation[refresh_exploration] = 0
        refresh_exploitation = exploitation_ids[
            stagnation[exploitation_ids] > config.refresh_gap
        ]
        if refresh_exploitation.numel():
            exemplars[refresh_exploitation] = _build_exemplars(
                refresh_exploitation,
                population_size,
                pbest,
                pbest_fitness,
                learning_probability,
                generator,
            )
            stagnation[refresh_exploitation] = 0

        best_value = float(global_best_fitness.item())
        convergence.append(best_value)
        generation_rows.append(
            {
                "generation": generation,
                "evaluations": evaluations,
                "evaluated_particles": evaluated_count,
                "best_fitness": best_value,
                "population_improvements": int(improved.sum().item()),
                "optimization_elapsed_sec": time.perf_counter() - start,
                "best_rule_tensor": global_best.clone(),
            }
        )
        if generations_without_evaluation >= config.max_generations_without_evaluation:
            stop_reason = "no_in_bounds_particles"
            break

    if device.type == "cuda" and config.synchronize_timing:
        torch.cuda.synchronize(device)
    optimization_sec = time.perf_counter() - start
    if evaluations < config.max_evaluations and time_expired():
        stop_reason = "time_limit"
    elif evaluations >= config.max_evaluations:
        stop_reason = "evaluation_limit"

    time_snapshot_row = None
    if config.time_limit_seconds is not None:
        eligible = [
            item for item in generation_rows
            if float(item["optimization_elapsed_sec"])
            <= float(config.time_limit_seconds)
        ]
        time_snapshot_row = eligible[-1] if eligible else generation_rows[0]

    return {
        "best_fitness": float(global_best_fitness.item()),
        "best_rule_tensor": global_best,
        "evaluation_count": evaluations,
        "evaluation_wall_sec": evaluation_wall_sec,
        "optimization_sec": optimization_sec,
        "generations": generation,
        "stop_reason": stop_reason,
        "convergence": convergence,
        "generation_rows": generation_rows,
        "time_snapshot_fitness": (
            None if time_snapshot_row is None
            else float(time_snapshot_row["best_fitness"])
        ),
        "time_snapshot_evaluation_count": (
            None if time_snapshot_row is None
            else int(time_snapshot_row["evaluations"])
        ),
        "time_snapshot_elapsed_sec": (
            None if time_snapshot_row is None
            else float(time_snapshot_row["optimization_elapsed_sec"])
        ),
        "time_snapshot_best_rule_tensor": (
            None if time_snapshot_row is None
            else time_snapshot_row["best_rule_tensor"]
        ),
        "time_snapshot_limit_reached": (
            config.time_limit_seconds is not None
            and optimization_sec >= float(config.time_limit_seconds)
        ),
        "config": asdict(config),
        "optimizer": "HCLPSO-GPU",
        "source_algorithm": "HCLPSO",
        "source_doi": "10.1016/j.swevo.2015.05.002",
        "exploration_population_size": exploration_size,
        "exploitation_population_size": exploitation_size,
        "adaptations": [
            "maximization comparisons",
            "15:25 subpopulation ratio scaled to the requested swarm size",
            "GPU tensor-batched objective evaluation",
            "true objective-call accounting",
        ],
    }
