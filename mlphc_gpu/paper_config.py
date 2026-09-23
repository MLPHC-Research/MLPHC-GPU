"""The parameter setting used in the paper's reported experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperConfig:
    hidden_neurons: int = 8
    population_size: int = 20
    parameter_scale: float = 2.0
    evaluations_per_target: int = 50
    seed: int = 2026091101
    exploration_fraction: float = 15.0 / 40.0
    refresh_gap: int = 5
    velocity_fraction: float = 0.2

    @property
    def rule_dimension(self) -> int:
        return 5 * self.hidden_neurons

    def max_evaluations(self, target_num: int) -> int:
        return self.evaluations_per_target * int(target_num)


PAPER_CONFIG = PaperConfig()
