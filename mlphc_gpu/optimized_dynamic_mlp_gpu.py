"""Tiny-MLP scorer for the exact dynamic-continuous GPU constructor.

Only the candidate scoring model differs from the optimized RBF evaluator.
All feasibility masks, continuous channel transitions, greedy selection, and
CUDA Graph replay are inherited unchanged.
"""

from __future__ import annotations

import torch

from .dynamic_mmrra_rbf_gpu import DecodedRules
from .optimized_dynamic_mmrra_rbf_gpu import OptimizedDynamicMMRRBFGPUEvaluator


class OptimizedDynamicMLPGPUEvaluator(OptimizedDynamicMMRRBFGPUEvaluator):
    """Evaluate a 3-H-1 tanh MLP whose parameters are evolved by HCLPSO.

    The output bias is omitted because adding the same scalar to every
    candidate logit cannot alter the greedy ranking. The paper configuration
    uses H=8 and therefore has 3H + H + H = 40 parameters.
    """

    def __init__(
        self,
        *args,
        mlp_hidden_neurons: int = 8,
        parameter_scale: float = 2.0,
        **kwargs,
    ) -> None:
        if mlp_hidden_neurons < 1:
            raise ValueError("mlp_hidden_neurons must be positive")
        if parameter_scale <= 0:
            raise ValueError("parameter_scale must be positive")
        kwargs.setdefault("score_chunk", 131072)
        super().__init__(*args, hidden_neurons=mlp_hidden_neurons, **kwargs)
        if self.rbf_input_mode != "standard" or self.center_dimension != 3:
            raise ValueError(
                "The tiny MLP currently requires the standard three-feature input"
            )
        self.mlp_hidden_neurons = int(mlp_hidden_neurons)
        self.parameter_scale = float(parameter_scale)
        self.score_model = "mlp"
        self.rule_dimension = 5 * self.mlp_hidden_neurons

    def initialize_search_population(
        self, population: torch.Tensor, generator: torch.Generator
    ) -> None:
        """Keep an unbiased uniform population for signed MLP parameters."""
        del generator
        if population.shape[1] != self.rule_dimension:
            raise ValueError("Unexpected MLP population shape")

    def decode_rules(self, weights: torch.Tensor) -> DecodedRules:
        if weights.ndim != 2 or weights.shape[1] != self.rule_dimension:
            raise ValueError(
                f"Expected a population N x {self.rule_dimension}, "
                f"found {tuple(weights.shape)}"
            )
        weights = weights.to(device=self.device, dtype=self.dtype)
        signed = (2.0 * weights - 1.0) * self.parameter_scale
        hidden = self.mlp_hidden_neurons
        input_stop = 3 * hidden
        bias_stop = input_stop + hidden

        input_weights = signed[:, :input_stop].reshape(-1, hidden, 3)
        hidden_biases = signed[:, input_stop:bias_stop]
        output_weights = signed[:, bias_stop : bias_stop + hidden]

        # Reuse the static CUDA Graph buffer contract. The RBF-named fields
        # carry MLP tensors, while unused fields receive inert values.
        batch = weights.shape[0]
        return DecodedRules(
            centers=input_weights,
            inv_two_spread2=torch.zeros(
                batch, device=self.device, dtype=self.dtype
            ),
            output_weights=output_weights,
            secondary_centers=torch.zeros(
                (batch, hidden, 4), device=self.device, dtype=self.dtype
            ),
            secondary_inv_two_spread2=torch.ones(
                batch, device=self.device, dtype=self.dtype
            ),
            secondary_output_weights=hidden_biases,
            use_secondary=torch.zeros(
                batch, device=self.device, dtype=torch.bool
            ),
        )

    def _score_pairs(
        self,
        rules: DecodedRules,
        rule_rows: torch.Tensor,
        candidate_ids: torch.Tensor,
        survival: torch.Tensor,
    ) -> torch.Tensor:
        if rules.secondary_output_weights is None:
            raise RuntimeError("Decoded MLP rules are missing hidden biases")
        target = self.target[candidate_ids]
        threat = self.target_base[target] * survival[rule_rows, target]
        probability = self.probability[candidate_ids]
        duration = self.third_feature[candidate_ids]
        logits = torch.zeros_like(probability)

        for hidden_id in range(self.mlp_hidden_neurons):
            input_weights = rules.centers[rule_rows, hidden_id]
            hidden_value = torch.tanh(
                input_weights[:, 0] * threat
                + input_weights[:, 1] * probability
                + input_weights[:, 2] * duration
                + rules.secondary_output_weights[rule_rows, hidden_id]
            )
            logits.add_(
                hidden_value * rules.output_weights[rule_rows, hidden_id]
            )

        return self._transform_logits(logits)

    def _transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Sigmoid is strictly monotone, so it preserves the MLP ranking while
        # satisfying the constructor's positive-score contract.
        return torch.sigmoid(logits)

    def _initial_scores(
        self, rules: DecodedRules, survival: torch.Tensor
    ) -> torch.Tensor:
        """Score B x Q candidates without repeated B*Q rule-index tensors."""
        if rules.secondary_output_weights is None:
            raise RuntimeError("Decoded MLP rules are missing hidden biases")
        batch = rules.centers.shape[0]
        candidate_count = self.candidates.count
        scores = torch.empty(
            (batch, candidate_count), dtype=self.dtype, device=self.device
        )
        for start in range(0, candidate_count, self.score_chunk):
            stop = min(candidate_count, start + self.score_chunk)
            target = self.target[start:stop]
            threat = (
                self.target_base[target][None, :] * survival[:, target]
            )
            probability = self.probability[start:stop][None, :]
            duration = self.third_feature[start:stop][None, :]
            logits = torch.zeros(
                (batch, stop - start), dtype=self.dtype, device=self.device
            )
            for hidden_id in range(self.mlp_hidden_neurons):
                input_weights = rules.centers[:, hidden_id]
                hidden_value = torch.tanh(
                    input_weights[:, 0, None] * threat
                    + input_weights[:, 1, None] * probability
                    + input_weights[:, 2, None] * duration
                    + rules.secondary_output_weights[:, hidden_id, None]
                )
                logits.add_(
                    hidden_value * rules.output_weights[:, hidden_id, None]
                )
            scores[:, start:stop] = self._transform_logits(logits)
        return scores

    def _refresh_target_dense(
        self,
        rules: DecodedRules,
        rows: torch.Tensor,
        target_ids: torch.Tensor,
        apply: torch.Tensor,
        survival: torch.Tensor,
        active: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        """Refresh one target per rule using B x K broadcasting."""
        if rules.secondary_output_weights is None:
            raise RuntimeError("Decoded MLP rules are missing hidden biases")
        ids = self.target_table[target_ids]
        row_grid = rows[:, None].expand_as(ids)
        valid = ids != self.dummy_candidate
        threat = self.target_base[target_ids][:, None] * survival[rows, target_ids][:, None]
        probability = self.probability[ids]
        duration = self.third_feature[ids]
        logits = torch.zeros_like(probability)
        for hidden_id in range(self.mlp_hidden_neurons):
            input_weights = rules.centers[rows, hidden_id]
            hidden_value = torch.tanh(
                input_weights[:, 0, None] * threat
                + input_weights[:, 1, None] * probability
                + input_weights[:, 2, None] * duration
                + rules.secondary_output_weights[rows, hidden_id, None]
            )
            logits.add_(
                hidden_value * rules.output_weights[rows, hidden_id, None]
            )
        values = self._transform_logits(logits)
        old_scores = scores[row_grid, ids]
        refresh = apply[:, None] & valid & active[row_grid, ids]
        scores[row_grid, ids] = torch.where(refresh, values, old_scores)

    def structure_signature(self) -> dict[str, object]:
        return {
            "mode": "fixed_tiny_mlp",
            "input_dimension": 3,
            "hidden_neurons": self.mlp_hidden_neurons,
            "output_dimension": 1,
            "output_bias": False,
            "rule_dimension": self.rule_dimension,
            "parameter_scale": self.parameter_scale,
        }

    def metadata(self) -> dict[str, object]:
        result = super().metadata()
        result.update(
            {
                "score_model": "tiny_mlp",
                "score_architecture": f"3-{self.mlp_hidden_neurons}-1",
                "activation": "tanh",
                "output_transform": "sigmoid",
                "output_bias": False,
                "rule_dimension": self.rule_dimension,
                "parameter_decode": (
                    f"theta=(2*w-1)*{self.parameter_scale:g}"
                ),
            }
        )
        return result


class AutoStructureDynamicMLPGPUEvaluator(OptimizedDynamicMLPGPUEvaluator):
    """GPU MLP evaluator with optimizer-controlled hidden-unit gates.

    A maximum-width ``3-H-1`` network is materialized for a fixed CUDA shape.
    The final H genes are continuous gates in [0, 1]; a unit is active when
    its gate reaches ``gate_threshold``.  Disabled units have zero output
    weight, so the same kernel evaluates all candidate architectures without
    rebuilding the graph.  The chromosome therefore searches both the MLP
    parameters and the effective hidden-unit count.
    """

    def __init__(
        self,
        *args,
        max_hidden_neurons: int = 5,
        gate_threshold: float = 0.5,
        parameter_scale: float = 2.0,
        **kwargs,
    ) -> None:
        if max_hidden_neurons < 1:
            raise ValueError("max_hidden_neurons must be positive")
        if not 0.0 < gate_threshold < 1.0:
            raise ValueError("gate_threshold must be in (0, 1)")
        super().__init__(
            *args,
            mlp_hidden_neurons=max_hidden_neurons,
            parameter_scale=parameter_scale,
            **kwargs,
        )
        self.max_hidden_neurons = int(max_hidden_neurons)
        self.gate_threshold = float(gate_threshold)
        self.gate_start = 5 * self.max_hidden_neurons
        self.rule_dimension = self.gate_start + self.max_hidden_neurons

    def initialize_search_population(
        self, population: torch.Tensor, generator: torch.Generator
    ) -> None:
        """Stratify the initial population over one through H active units."""
        if population.shape[1] != self.rule_dimension:
            raise ValueError("Unexpected automatic-MLP population shape")
        hidden = self.max_hidden_neurons
        population_size = population.shape[0]
        order = torch.rand(
            (population_size, hidden),
            device=self.device,
            dtype=population.dtype,
            generator=generator,
        ).argsort(dim=1)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(hidden, device=self.device, dtype=torch.long).expand(
                population_size, -1
            ),
        )
        active_counts = 1 + (
            torch.arange(population_size, device=self.device, dtype=torch.long)
            % hidden
        )
        active = ranks < active_counts[:, None]
        low = self.gate_threshold * torch.rand(
            (population_size, hidden),
            device=self.device,
            dtype=population.dtype,
            generator=generator,
        )
        high = self.gate_threshold + (1.0 - self.gate_threshold) * torch.rand(
            (population_size, hidden),
            device=self.device,
            dtype=population.dtype,
            generator=generator,
        )
        population[:, self.gate_start :] = torch.where(active, high, low)

    def decode_rules(self, weights: torch.Tensor) -> DecodedRules:
        if weights.ndim != 2 or weights.shape[1] != self.rule_dimension:
            raise ValueError(
                f"Expected a population N x {self.rule_dimension}, "
                f"found {tuple(weights.shape)}"
            )
        weights = weights.to(device=self.device, dtype=self.dtype)
        hidden = self.max_hidden_neurons
        signed = (2.0 * weights[:, : self.gate_start] - 1.0) * self.parameter_scale
        input_stop = 3 * hidden
        bias_stop = input_stop + hidden
        output_stop = bias_stop + hidden
        input_weights = signed[:, :input_stop].reshape(-1, hidden, 3)
        hidden_biases = signed[:, input_stop:bias_stop]
        output_weights = signed[:, bias_stop:output_stop]

        gates = weights[:, self.gate_start :]
        active = gates >= self.gate_threshold
        empty = ~active.any(dim=1)
        fallback = gates.argmax(dim=1, keepdim=True)
        active = active.scatter(
            1,
            fallback,
            active.gather(1, fallback) | empty[:, None],
        )
        output_weights = output_weights * active.to(self.dtype)
        batch = weights.shape[0]
        return DecodedRules(
            centers=input_weights,
            inv_two_spread2=torch.zeros(
                batch, device=self.device, dtype=self.dtype
            ),
            output_weights=output_weights,
            secondary_centers=torch.zeros(
                (batch, hidden, 4), device=self.device, dtype=self.dtype
            ),
            secondary_inv_two_spread2=torch.ones(
                batch, device=self.device, dtype=self.dtype
            ),
            secondary_output_weights=hidden_biases,
            use_secondary=torch.zeros(
                batch, device=self.device, dtype=torch.bool
            ),
        )

    def active_hidden_counts(self, weights: torch.Tensor) -> torch.Tensor:
        gates = weights[:, self.gate_start : self.gate_start + self.max_hidden_neurons]
        return (gates >= self.gate_threshold).sum(dim=1).clamp_min(1)

    def architecture_summary(self, weights: torch.Tensor) -> dict[str, int]:
        count = int(self.active_hidden_counts(weights[:1])[0].item())
        return {
            "depth": 1,
            "layer1_active_neurons": count,
            "layer2_active_neurons": 0,
            "total_active_neurons": count,
        }

    def extra_parameter_groups(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return (
            (
                "mlp_hidden_unit_gates",
                tuple(range(self.gate_start, self.rule_dimension)),
            ),
        )

    def structure_signature(self) -> dict[str, object]:
        return {
            "mode": "automatic_gated_mlp",
            "input_dimension": 3,
            "max_hidden_neurons": self.max_hidden_neurons,
            "gate_threshold": self.gate_threshold,
            "output_dimension": 1,
            "output_bias": False,
            "rule_dimension": self.rule_dimension,
            "parameter_scale": self.parameter_scale,
        }

    def metadata(self) -> dict[str, object]:
        result = super().metadata()
        result.update(
            {
                "score_model": "automatic_mlp",
                "score_architecture": f"3-{self.max_hidden_neurons}-1_gated",
                "automatic_structure_search": True,
                "gate_threshold": self.gate_threshold,
                "max_hidden_neurons": self.max_hidden_neurons,
                "rule_dimension": self.rule_dimension,
            }
        )
        return result
