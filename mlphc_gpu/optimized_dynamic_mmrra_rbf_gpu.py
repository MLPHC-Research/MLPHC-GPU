"""Fixed-shape, compiled dynamic-continuous MMRRA-RBF evaluator.

The baseline implementation remains available for semantic validation. This
variant removes CPU decisions from the per-instance bounded construction loop
and pads all candidate tables with one inert sentinel, allowing every update
to retain a static tensor shape suitable for CUDA Graph replay.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .dynamic_mmrra_rbf_gpu import (
    DecodedRules,
    DynamicMMRRBFGPUEvaluator,
    EvaluationResult,
)


class OptimizedDynamicMMRRBFGPUEvaluator(DynamicMMRRBFGPUEvaluator):
    def __init__(
        self, *args, use_cuda_graph: bool = True, graph_unroll: int = 1, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.real_candidate_count = self.candidates.count
        self.dummy_candidate = self.real_candidate_count
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self.graph_capture_sec = 0.0
        self._graph_batch_size: int | None = None
        self._step_graph: torch.cuda.CUDAGraph | None = None
        self._graph_buffers: dict[str, torch.Tensor] = {}
        max_accepts = int(min(self.inst.ammo.sum(), self.inst.strike.sum()))
        requested_unroll = int(graph_unroll)
        if requested_unroll == 0:
            requested_unroll = max_accepts
        if requested_unroll < 1:
            raise ValueError("graph_unroll must be nonnegative")
        requested_unroll = min(max(1, requested_unroll), max(1, max_accepts))
        while max_accepts > 0 and max_accepts % requested_unroll != 0:
            requested_unroll -= 1
        self.graph_unroll = max(1, requested_unroll)

        def append_scalar(tensor: torch.Tensor, value: int | float | bool) -> torch.Tensor:
            suffix = torch.as_tensor([value], device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, suffix], dim=0)

        self.radar = append_scalar(self.radar, 0)
        self.weapon = append_scalar(self.weapon, 0)
        self.target = append_scalar(self.target, 0)
        self.identity = append_scalar(self.identity, 0)
        self.probability = append_scalar(self.probability, 0.0)
        self.demand = append_scalar(self.demand, 1)
        self.time_norm = append_scalar(self.time_norm, 0.0)
        self.third_feature = append_scalar(self.third_feature, 0.0)
        self.fourth_feature = append_scalar(self.fourth_feature, 0.0)
        self.launch = append_scalar(self.launch, 0)
        self.launch_gap = append_scalar(self.launch_gap, 0)
        self.interval_valid = append_scalar(self.interval_valid, False)
        self.segment_start = append_scalar(self.segment_start, 0)
        self.segment_stop = append_scalar(self.segment_stop, 0)

        for name in (
            "target_table",
            "weapon_table",
            "radar_table",
            "radar_target_table",
            "identity_table",
            "launch_resource_table",
        ):
            table = getattr(self, name)
            setattr(
                self,
                name,
                torch.where(table >= 0, table, torch.full_like(table, self.dummy_candidate)),
            )

        self._step_runner = self._decode_step_static

    def _select_best(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact row-wise maximum and its lowest candidate index."""
        return scores.max(dim=1)

    def _build_cuda_graph(self, batch: int) -> None:
        if not self.use_cuda_graph:
            return
        if self._step_graph is not None and self._graph_batch_size == batch:
            return
        if self._step_graph is not None:
            raise RuntimeError(
                "CUDA Graph batch size changed; keep the population tensor fixed"
            )

        capture_start = time.perf_counter()
        hidden = self.hidden_neurons
        max_accepts = int(min(self.inst.ammo.sum(), self.inst.strike.sum()))
        buffers: dict[str, torch.Tensor] = {
            "centers": torch.zeros(
                (batch, hidden, self.center_dimension),
                device=self.device,
                dtype=self.dtype,
            ),
            "inv_two_spread2": torch.ones(batch, device=self.device, dtype=self.dtype),
            "output_weights": torch.full(
                (batch, hidden), 1.0 / hidden, device=self.device, dtype=self.dtype
            ),
            "secondary_centers": torch.zeros(
                (batch, hidden, self.center_dimension + 1),
                device=self.device,
                dtype=self.dtype,
            ),
            "secondary_inv_two_spread2": torch.ones(
                batch, device=self.device, dtype=self.dtype
            ),
            "secondary_output_weights": torch.full(
                (batch, hidden), 1.0 / hidden, device=self.device, dtype=self.dtype
            ),
            "use_secondary": torch.zeros(
                batch, device=self.device, dtype=torch.bool
            ),
            "rows": torch.arange(batch, device=self.device, dtype=torch.long),
            "rule_enabled": torch.zeros(batch, device=self.device, dtype=torch.bool),
            "survival": torch.ones(
                (batch, self.inst.target_num), device=self.device, dtype=self.dtype
            ),
            "ammo": torch.as_tensor(
                self.inst.ammo, device=self.device, dtype=torch.int32
            ).repeat(batch, 1),
            "strike": torch.as_tensor(
                self.inst.strike, device=self.device, dtype=torch.int32
            ).repeat(batch, 1),
            "remaining": self.segment_capacity.repeat(batch, 1, 1),
            "shortage_prefix": torch.zeros(
                (
                    batch,
                    self.inst.radar_num,
                    self.max_demand,
                    self.candidates.max_segments_per_radar + 1,
                ),
                device=self.device,
                dtype=torch.int32,
            ),
            "active": torch.zeros(
                (batch, self.real_candidate_count + 1),
                device=self.device,
                dtype=torch.bool,
            ),
            "scores": torch.full(
                (batch, self.real_candidate_count + 1),
                -torch.inf,
                device=self.device,
                dtype=self.dtype,
            ),
            "selected_count": torch.zeros(
                batch, device=self.device, dtype=torch.int32
            ),
            "channel_rejections": torch.zeros(
                batch, device=self.device, dtype=torch.int32
            ),
            "selection_history": torch.full(
                (batch, max_accepts), -1, device=self.device, dtype=torch.int32
            ),
        }
        ordered_names = (
            "centers",
            "inv_two_spread2",
            "output_weights",
            "secondary_centers",
            "secondary_inv_two_spread2",
            "secondary_output_weights",
            "use_secondary",
            "rows",
            "rule_enabled",
            "survival",
            "ammo",
            "strike",
            "remaining",
            "shortage_prefix",
            "active",
            "scores",
            "selected_count",
            "channel_rejections",
            "selection_history",
        )
        step_args = tuple(buffers[name] for name in ordered_names)

        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._decode_step_static(*step_args)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(self.graph_unroll):
                self._decode_step_static(*step_args)

        torch.cuda.synchronize(self.device)
        self._graph_batch_size = batch
        self._graph_buffers = buffers
        self._step_graph = graph
        self.graph_capture_sec = time.perf_counter() - capture_start

    def _run_cuda_graph(
        self,
        rules: DecodedRules,
        rule_enabled: torch.Tensor,
        initial_active: torch.Tensor,
        initial_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch = initial_scores.shape[0]
        self._build_cuda_graph(batch)
        if self._step_graph is None:
            raise RuntimeError("CUDA Graph was not initialized")
        b = self._graph_buffers
        b["centers"].copy_(rules.centers)
        b["inv_two_spread2"].copy_(rules.inv_two_spread2)
        b["output_weights"].copy_(rules.output_weights)
        if rules.secondary_centers is None:
            b["secondary_centers"].zero_()
            b["secondary_inv_two_spread2"].fill_(1.0)
            b["secondary_output_weights"].fill_(1.0 / self.hidden_neurons)
            b["use_secondary"].zero_()
        else:
            b["secondary_centers"].copy_(rules.secondary_centers)
            b["secondary_inv_two_spread2"].copy_(
                rules.secondary_inv_two_spread2
            )
            b["secondary_output_weights"].copy_(rules.secondary_output_weights)
            b["use_secondary"].copy_(rules.use_secondary)
        b["rule_enabled"].copy_(rule_enabled)
        b["survival"].fill_(1.0)
        b["ammo"].copy_(
            torch.as_tensor(self.inst.ammo, device=self.device, dtype=torch.int32)
        )
        b["strike"].copy_(
            torch.as_tensor(self.inst.strike, device=self.device, dtype=torch.int32)
        )
        b["remaining"].copy_(self.segment_capacity)
        b["shortage_prefix"].zero_()
        b["active"].copy_(initial_active)
        b["scores"].copy_(initial_scores)
        b["selected_count"].zero_()
        b["channel_rejections"].zero_()
        b["selection_history"].fill_(-1)

        max_accepts = int(min(self.inst.ammo.sum(), self.inst.strike.sum()))
        for _ in range(max_accepts // self.graph_unroll):
            self._step_graph.replay()
        return (
            b["survival"],
            b["ammo"],
            b["strike"],
            b["remaining"],
            b["shortage_prefix"],
            b["active"],
            b["scores"],
            b["selected_count"],
            b["channel_rejections"],
            b["selection_history"],
        )

    def _initial_scores_static(
        self, rules: DecodedRules, survival: torch.Tensor
    ) -> torch.Tensor:
        # Dispatch to score-model-specific broadcast implementations when
        # available (the tiny MLP avoids materializing repeated rule indices).
        real_scores = self._initial_scores(rules, survival)
        dummy = torch.full(
            (real_scores.shape[0], 1),
            -torch.inf,
            dtype=self.dtype,
            device=self.device,
        )
        return torch.cat([real_scores, dummy], dim=1)

    def _deactivate_dense(
        self,
        rows: torch.Tensor,
        group_ids: torch.Tensor,
        table: torch.Tensor,
        apply: torch.Tensor,
        active: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        ids = table[group_ids]
        row_grid = rows[:, None].expand_as(ids)
        deactivate = apply[:, None] & (ids != self.dummy_candidate)
        old_active = active[row_grid, ids]
        old_scores = scores[row_grid, ids]
        active[row_grid, ids] = old_active & ~deactivate
        scores[row_grid, ids] = torch.where(deactivate, -torch.inf, old_scores)

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
        ids = self.target_table[target_ids]
        row_grid = rows[:, None].expand_as(ids)
        valid = ids != self.dummy_candidate
        values = self._score_pairs(
            rules,
            row_grid.reshape(-1),
            ids.reshape(-1),
            survival,
        ).reshape_as(ids)
        old_scores = scores[row_grid, ids]
        refresh = apply[:, None] & valid & active[row_grid, ids]
        scores[row_grid, ids] = torch.where(refresh, values, old_scores)

    def _prune_dense(
        self,
        rows: torch.Tensor,
        radar_ids: torch.Tensor,
        apply: torch.Tensor,
        shortage_prefix: torch.Tensor,
        active: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        ids = self.radar_table[radar_ids]
        row_grid = rows[:, None].expand_as(ids)
        valid = ids != self.dummy_candidate
        candidate_radar = self.radar[ids]
        demand_index = self.demand[ids] - 1
        blocked = (
            shortage_prefix[
                row_grid, candidate_radar, demand_index, self.segment_stop[ids]
            ]
            - shortage_prefix[
                row_grid, candidate_radar, demand_index, self.segment_start[ids]
            ]
        ) > 0
        deactivate = apply[:, None] & valid & blocked
        old_active = active[row_grid, ids]
        old_scores = scores[row_grid, ids]
        active[row_grid, ids] = old_active & ~deactivate
        scores[row_grid, ids] = torch.where(deactivate, -torch.inf, old_scores)

    def _decode_step_static(
        self,
        centers: torch.Tensor,
        inv_two_spread2: torch.Tensor,
        output_weights: torch.Tensor,
        secondary_centers: torch.Tensor,
        secondary_inv_two_spread2: torch.Tensor,
        secondary_output_weights: torch.Tensor,
        use_secondary: torch.Tensor,
        rows: torch.Tensor,
        rule_enabled: torch.Tensor,
        survival: torch.Tensor,
        ammo: torch.Tensor,
        strike: torch.Tensor,
        remaining: torch.Tensor,
        shortage_prefix: torch.Tensor,
        active: torch.Tensor,
        scores: torch.Tensor,
        selected_count: torch.Tensor,
        channel_rejections: torch.Tensor,
        selection_history: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        rules = DecodedRules(
            centers=centers,
            inv_two_spread2=inv_two_spread2,
            output_weights=output_weights,
            secondary_centers=secondary_centers,
            secondary_inv_two_spread2=secondary_inv_two_spread2,
            secondary_output_weights=secondary_output_weights,
            use_secondary=use_secondary,
        )
        best_score, chosen = self._select_best(scores)
        live = rule_enabled & torch.isfinite(best_score) & (best_score > 0)
        radar = self.radar[chosen]
        weapon = self.weapon[chosen]
        target = self.target[chosen]
        demand = self.demand[chosen]
        starts = self.segment_start[chosen]
        stops = self.segment_stop[chosen]

        blocked = (
            shortage_prefix[rows, radar, demand - 1, stops]
            - shortage_prefix[rows, radar, demand - 1, starts]
        ) > 0
        rejected = live & blocked
        old_chosen_active = active[rows, chosen]
        old_chosen_score = scores[rows, chosen]
        active[rows, chosen] = old_chosen_active & ~rejected
        scores[rows, chosen] = torch.where(rejected, -torch.inf, old_chosen_score)
        channel_rejections.add_(rejected.to(torch.int32))
        accepted = live & ~blocked

        interval_mask = (
            (self.segment_position[None, :] >= starts[:, None])
            & (self.segment_position[None, :] < stops[:, None])
            & accepted[:, None]
        )
        selected_remaining = remaining[rows, radar]
        updated_remaining = selected_remaining - (
            interval_mask.to(torch.int16) * demand[:, None].to(torch.int16)
        )
        remaining[rows, radar] = updated_remaining
        old_prefix = shortage_prefix[rows, radar, :, 1:]
        demand_levels = torch.arange(
            1, self.max_demand + 1, device=self.device, dtype=torch.int16
        )
        updated_prefix = torch.cumsum(
            updated_remaining[:, None, :] < demand_levels[None, :, None],
            dim=2,
            dtype=torch.int32,
        )
        shortage_prefix[rows, radar, :, 1:] = torch.where(
            accepted[:, None, None], updated_prefix, old_prefix
        )

        history_index = selected_count.to(torch.long).clamp_max(selection_history.shape[1] - 1)
        old_history = selection_history[rows, history_index]
        selection_history[rows, history_index] = torch.where(
            accepted, chosen.to(torch.int32), old_history
        )
        selected_count.add_(accepted.to(torch.int32))

        old_ammo = ammo[rows, weapon]
        ammo[rows, weapon] = old_ammo - accepted.to(torch.int32)
        old_strike = strike[rows, target]
        strike[rows, target] = old_strike - accepted.to(torch.int32)
        old_survival = survival[rows, target]
        survival[rows, target] = old_survival * torch.where(
            accepted, 1.0 - self.probability[chosen], torch.ones_like(old_survival)
        )

        identity_ids = self.identity[chosen]
        self._deactivate_dense(
            rows, identity_ids, self.identity_table, accepted, active, scores
        )
        for offset in range(self.max_launch_gap):
            transfer_apply = accepted & (self.launch_gap[chosen] > offset)
            transfer_group = (
                weapon * self.launch_slots + self.launch[chosen] + offset
            )
            self._deactivate_dense(
                rows,
                transfer_group,
                self.launch_resource_table,
                transfer_apply,
                active,
                scores,
            )
        ammo_empty = accepted & (ammo[rows, weapon] <= 0)
        self._deactivate_dense(
            rows, weapon, self.weapon_table, ammo_empty, active, scores
        )
        target_empty = accepted & (strike[rows, target] <= 0)
        self._deactivate_dense(
            rows, target, self.target_table, target_empty, active, scores
        )

        if self.has_unit_probability:
            unit = self.identity_unit_probability[identity_ids]
            for radar_id in range(self.inst.radar_num):
                unit_apply = accepted & unit[:, radar_id]
                radar_target_group = radar_id * self.inst.target_num + target
                self._deactivate_dense(
                    rows,
                    radar_target_group,
                    self.radar_target_table,
                    unit_apply,
                    active,
                    scores,
                )

        refresh = accepted & (strike[rows, target] > 0)
        self._refresh_target_dense(
            rules, rows, target, refresh, survival, active, scores
        )
        self._prune_dense(
            rows, radar, accepted, shortage_prefix, active, scores
        )
        return (
            survival,
            ammo,
            strike,
            remaining,
            shortage_prefix,
            active,
            scores,
            selected_count,
            channel_rejections,
            selection_history,
        )

    @torch.inference_mode()
    def evaluate(
        self,
        weights: torch.Tensor,
        record_selection: bool = False,
        rule_enabled: torch.Tensor | None = None,
    ) -> EvaluationResult:
        weights = weights.to(device=self.device, dtype=self.dtype)
        rules = self.decode_rules(weights)
        batch = weights.shape[0]
        rows = torch.arange(batch, device=self.device, dtype=torch.long)
        if rule_enabled is None:
            rule_enabled = torch.ones(batch, device=self.device, dtype=torch.bool)
        else:
            rule_enabled = rule_enabled.to(device=self.device, dtype=torch.bool)
        survival = torch.ones(
            (batch, self.inst.target_num), dtype=self.dtype, device=self.device
        )
        ammo = torch.as_tensor(
            self.inst.ammo, dtype=torch.int32, device=self.device
        ).repeat(batch, 1)
        strike = torch.as_tensor(
            self.inst.strike, dtype=torch.int32, device=self.device
        ).repeat(batch, 1)
        remaining = self.segment_capacity.repeat(batch, 1, 1)
        shortage_prefix = torch.zeros(
            (
                batch,
                self.inst.radar_num,
                self.max_demand,
                self.candidates.max_segments_per_radar + 1,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        active = self.interval_valid.repeat(batch, 1)
        ammo_available = torch.as_tensor(
            self.inst.ammo > 0, device=self.device, dtype=torch.bool
        )[self.weapon]
        target_available = torch.as_tensor(
            self.inst.strike > 0, device=self.device, dtype=torch.bool
        )[self.target]
        active &= rule_enabled[:, None] & ammo_available[None, :] & target_available[None, :]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        scoring_start = time.perf_counter()
        scores = self._initial_scores_static(rules, survival)
        scores.masked_fill_(~active, -torch.inf)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        initial_scoring_sec = time.perf_counter() - scoring_start

        max_accepts = int(min(self.inst.ammo.sum(), self.inst.strike.sum()))
        selected_count = torch.zeros(batch, dtype=torch.int32, device=self.device)
        channel_rejections = torch.zeros(batch, dtype=torch.int32, device=self.device)
        selection_history = torch.full(
            (batch, max_accepts), -1, dtype=torch.int32, device=self.device
        )

        decode_start = time.perf_counter()
        if max_accepts == 0:
            pass
        elif self.use_cuda_graph:
            (
                survival,
                ammo,
                strike,
                remaining,
                shortage_prefix,
                active,
                scores,
                selected_count,
                channel_rejections,
                selection_history,
            ) = self._run_cuda_graph(
                rules,
                rule_enabled,
                active,
                scores,
            )
        else:
            if rules.secondary_centers is None:
                secondary_centers = torch.zeros(
                    (
                        batch,
                        self.hidden_neurons,
                        self.center_dimension + 1,
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )
                secondary_inv_two_spread2 = torch.ones(
                    batch, device=self.device, dtype=self.dtype
                )
                secondary_output_weights = torch.full(
                    (batch, self.hidden_neurons),
                    1.0 / self.hidden_neurons,
                    device=self.device,
                    dtype=self.dtype,
                )
                use_secondary = torch.zeros(
                    batch, device=self.device, dtype=torch.bool
                )
            else:
                secondary_centers = rules.secondary_centers
                secondary_inv_two_spread2 = rules.secondary_inv_two_spread2
                secondary_output_weights = rules.secondary_output_weights
                use_secondary = rules.use_secondary
            for _ in range(max_accepts):
                (
                    survival,
                    ammo,
                    strike,
                    remaining,
                    shortage_prefix,
                    active,
                    scores,
                    selected_count,
                    channel_rejections,
                    selection_history,
                ) = self._step_runner(
                    rules.centers,
                    rules.inv_two_spread2,
                    rules.output_weights,
                    secondary_centers,
                    secondary_inv_two_spread2,
                    secondary_output_weights,
                    use_secondary,
                    rows,
                    rule_enabled,
                    survival,
                    ammo,
                    strike,
                    remaining,
                    shortage_prefix,
                    active,
                    scores,
                    selected_count,
                    channel_rejections,
                    selection_history,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_sec = time.perf_counter() - decode_start
        enabled_rejections = channel_rejections[rule_enabled]
        if (
            bool(torch.any(enabled_rejections != 0))
            and not bool(getattr(self, "allow_lazy_channel_rejections", False))
        ):
            raise RuntimeError(
                "Eager continuous-channel pruning left a stale blocked candidate"
            )
        if bool(torch.any(remaining[rule_enabled] < 0)):
            raise RuntimeError("Continuous channel capacity became negative")
        fitness = ((1.0 - survival) * self.target_value[None, :]).sum(dim=1)
        return EvaluationResult(
            fitness=fitness.clone(),
            selected_count=selected_count.clone(),
            channel_rejections=channel_rejections.clone(),
            initial_scoring_sec=initial_scoring_sec,
            decode_sec=decode_sec,
            iterations=max_accepts,
            selection_history=selection_history.clone() if record_selection else None,
        )

    def metadata(self) -> dict[str, object]:
        result = super().metadata()
        result.update(
            {
                "fixed_construction_steps": int(
                    min(self.inst.ammo.sum(), self.inst.strike.sum())
                ),
                "cuda_graph": self.use_cuda_graph,
                "graph_capture_sec": self.graph_capture_sec,
                "candidate_padding": 1,
                "cuda_graph_unroll": self.graph_unroll,
                "cuda_graph_replays_per_evaluation": int(
                    min(self.inst.ammo.sum(), self.inst.strike.sum())
                ) // self.graph_unroll,
            }
        )
        return result
