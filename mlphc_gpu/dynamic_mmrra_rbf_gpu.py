"""GPU-batched MMRRA-RBF decoder with exact continuous-time radar capacity.

This module is deliberately isolated from the legacy front/back decoder. Each
radar timeline is partitioned by all candidate endpoints. The residual channel
capacity on those elementary half-open segments is an exact representation of
the continuous-time load; no time-grid approximation is introduced.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import h5py


from .instance_io import InstanceData, count_feasible_quads, load_instance


@dataclass
class DynamicCandidateData:
    radar: np.ndarray
    weapon: np.ndarray
    target: np.ndarray
    launch: np.ndarray
    identity: np.ndarray
    begin: np.ndarray
    end: np.ndarray
    probability: np.ndarray
    demand: np.ndarray
    time_norm: np.ndarray
    interval_valid: np.ndarray
    segment_start: np.ndarray
    segment_stop: np.ndarray
    segment_capacity: np.ndarray
    segment_count_by_radar: np.ndarray
    target_table: np.ndarray
    weapon_table: np.ndarray
    radar_table: np.ndarray
    radar_target_table: np.ndarray
    identity_table: np.ndarray
    launch_resource_table: np.ndarray
    identity_unit_probability: np.ndarray
    launch_slots: int

    @property
    def count(self) -> int:
        return int(self.radar.size)

    @property
    def segment_count(self) -> int:
        return int(self.segment_count_by_radar.sum())

    @property
    def max_segments_per_radar(self) -> int:
        return int(self.segment_capacity.shape[1])


@dataclass
class DecodedRules:
    centers: torch.Tensor
    inv_two_spread2: torch.Tensor
    output_weights: torch.Tensor
    secondary_centers: torch.Tensor | None = None
    secondary_inv_two_spread2: torch.Tensor | None = None
    secondary_output_weights: torch.Tensor | None = None
    use_secondary: torch.Tensor | None = None


@dataclass
class EvaluationResult:
    fitness: torch.Tensor
    selected_count: torch.Tensor
    channel_rejections: torch.Tensor
    initial_scoring_sec: float
    decode_sec: float
    iterations: int
    selection_history: torch.Tensor | None = None


def _group_table(keys: np.ndarray, group_count: int, sentinel: int = -1) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    counts = np.bincount(keys, minlength=group_count)
    width = int(counts.max(initial=0))
    if width == 0:
        return np.full((group_count, 1), sentinel, dtype=np.int64)
    table = np.full((group_count, width), sentinel, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.cumsum(np.r_[0, counts[:-1]])
    offsets = np.arange(keys.size, dtype=np.int64) - starts[sorted_keys]
    table[sorted_keys, offsets] = order
    return table


def _matlab_array(dataset: h5py.Dataset) -> np.ndarray:
    value = np.asarray(dataset)
    if value.ndim > 1:
        value = value.transpose(tuple(range(value.ndim - 1, -1, -1)))
    return value


def _v73_cells(handle: h5py.File, dataset: h5py.Dataset) -> list[np.ndarray]:
    refs = np.asarray(dataset).reshape(-1, order="F")
    return [np.asarray(_matlab_array(handle[ref]), dtype=np.float64) for ref in refs]


def _v73_weapon_launch_gap(group: h5py.Group, weapon_num: int) -> np.ndarray:
    if "weapon_launch_min_gap" not in group:
        return np.zeros(weapon_num, dtype=np.int64)
    value = np.asarray(
        _matlab_array(group["weapon_launch_min_gap"]), dtype=np.float64
    ).reshape(-1)
    return np.rint(value).astype(np.int64)


def load_dynamic_instance(path: Path) -> InstanceData:
    """Load either legacy MAT files or strict official v7.3 snapshots."""
    try:
        inst = load_instance(path)
        setattr(inst, "target_channel_demand", np.ones(inst.target_num, dtype=np.int64))
        setattr(inst, "weapon_launch_gap", np.zeros(inst.weapon_num, dtype=np.int64))
        return inst
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as handle:
        group = handle["inst"]
        weapon_pr = np.asarray(_matlab_array(group["weapon_pr_matrix"]), dtype=np.float64)
        if weapon_pr.ndim == 2:
            weapon_pr = weapon_pr[:, :, None]
        if "weapon_vp_matrix" in group:
            weapon_vp = np.asarray(
                _matlab_array(group["weapon_vp_matrix"]), dtype=np.float64
            )
        else:
            # The compact official MOP1--MOP5 branch-price snapshots omit
            # this redundant matrix.  The original dynamic converters define
            # it as weapon_pr_matrix multiplied by the target value.  Rebuild
            # the same activity/value tensor without changing the instance.
            target_value_for_vp = np.asarray(
                _matlab_array(group["V_matrix"]), dtype=np.float64
            ).reshape(-1)
            weapon_vp = weapon_pr * target_value_for_vp[None, :, None]
        if weapon_vp.ndim == 2:
            weapon_vp = weapon_vp[:, :, None]
        weapon_num, target_num, time_num = weapon_pr.shape
        radar_num = int(np.asarray(group["radar_num"]).reshape(-1)[0])

        def vector(name: str) -> np.ndarray:
            return np.asarray(_matlab_array(group[name]), dtype=np.float64).reshape(-1)

        radar_capacity = np.rint(vector("radar_constraint")).astype(np.int64)
        if radar_capacity.size == 1 and radar_num > 1:
            radar_capacity = np.repeat(radar_capacity, radar_num)
        inst = InstanceData(
            radar_num=radar_num,
            weapon_num=weapon_num,
            target_num=target_num,
            time_num=time_num,
            weapon_pr=weapon_pr,
            weapon_vp=weapon_vp,
            ammo=np.rint(vector("danyao_constraint")).astype(np.int64),
            strike=np.rint(vector("strike_constraint")).astype(np.int64),
            radar_capacity=radar_capacity,
            target_value=vector("V_matrix"),
            radar_begin=_v73_cells(handle, group["radar_tbegin_cell"]),
            radar_end=_v73_cells(handle, group["radar_tend_cell"]),
            radar_pr=_v73_cells(handle, group["radar_pr_cell"]),
        )
        demand = (
            np.rint(vector("target_channel_demand")).astype(np.int64)
            if "target_channel_demand" in group
            else np.ones(target_num, dtype=np.int64)
        )
        launch_gap = _v73_weapon_launch_gap(group, weapon_num)
    if demand.size != target_num or np.any(demand < 1):
        raise ValueError("target_channel_demand must contain one positive value per target")
    setattr(inst, "target_channel_demand", demand)
    if launch_gap.size == 1 and weapon_num > 1:
        launch_gap = np.repeat(launch_gap, weapon_num)
    if launch_gap.size != weapon_num or np.any(launch_gap < 0):
        raise ValueError("weapon launch gaps must contain one nonnegative value per interceptor")
    setattr(inst, "weapon_launch_gap", launch_gap)
    return inst


def _membership_table(
    group_keys: np.ndarray,
    candidate_ids: np.ndarray,
    group_count: int,
    sentinel: int = -1,
) -> np.ndarray:
    if group_keys.size == 0:
        return np.full((max(1, group_count), 1), sentinel, dtype=np.int64)
    counts = np.bincount(group_keys, minlength=group_count)
    width = int(counts.max(initial=0))
    table = np.full((group_count, max(1, width)), sentinel, dtype=np.int64)
    order = np.argsort(group_keys, kind="stable")
    sorted_keys = group_keys[order]
    starts = np.cumsum(np.r_[0, counts[:-1]])
    offsets = np.arange(group_keys.size, dtype=np.int64) - starts[sorted_keys]
    table[sorted_keys, offsets] = candidate_ids[order]
    return table


def build_dynamic_candidates(inst: InstanceData) -> DynamicCandidateData:
    wp_flat = inst.weapon_pr.reshape(-1, order="F")
    wvp_flat = inst.weapon_vp.reshape(-1, order="F")
    radar_parts = []
    weapon_parts = []
    target_parts = []
    launch_parts = []
    identity_parts = []
    begin_parts = []
    end_parts = []
    probability_parts = []

    for radar_id in range(inst.radar_num):
        radar_probability = inst.radar_pr[radar_id].reshape(-1, order="F")
        active_identity = np.flatnonzero((radar_probability * wvp_flat) != 0.0)
        weapon = active_identity % inst.weapon_num
        tmp = active_identity // inst.weapon_num
        target = tmp % inst.target_num
        launch = tmp // inst.target_num
        begin = inst.radar_begin[radar_id].reshape(-1, order="F")[active_identity]
        end = inst.radar_end[radar_id].reshape(-1, order="F")[active_identity]

        radar_parts.append(np.full(active_identity.size, radar_id, dtype=np.int64))
        weapon_parts.append(weapon.astype(np.int64))
        target_parts.append(target.astype(np.int64))
        launch_parts.append(launch.astype(np.int64))
        identity_parts.append(active_identity.astype(np.int64))
        begin_parts.append(begin.astype(np.float64))
        end_parts.append(end.astype(np.float64))
        probability_parts.append((radar_probability[active_identity] * wp_flat[active_identity]).astype(np.float64))

    radar = np.concatenate(radar_parts)
    weapon = np.concatenate(weapon_parts)
    target = np.concatenate(target_parts)
    launch = np.concatenate(launch_parts)
    identity = np.concatenate(identity_parts)
    begin = np.concatenate(begin_parts)
    end = np.concatenate(end_parts)
    probability = np.concatenate(probability_parts)
    candidate_count = radar.size
    if candidate_count == 0:
        raise ValueError("Instance has no active MMRRA-RBF candidates")
    duration = end - begin
    duration_min = float(np.min(duration))
    duration_max = float(np.max(duration))
    duration_epsilon = 1e-12
    time_norm = np.clip(
        1.0
        - (duration - duration_min)
        / (duration_max - duration_min + duration_epsilon),
        0.0,
        1.0,
    )
    target_channel_demand = np.asarray(
        getattr(inst, "target_channel_demand", np.ones(inst.target_num)), dtype=np.int64
    ).reshape(-1)
    demand = target_channel_demand[target]

    interval_valid = np.isfinite(begin) & np.isfinite(end) & (end > begin)
    interval_valid &= (demand >= 1) & (demand <= inst.radar_capacity[radar])
    segment_start = np.zeros(candidate_count, dtype=np.int64)
    segment_stop = np.zeros(candidate_count, dtype=np.int64)
    segment_counts = np.zeros(inst.radar_num, dtype=np.int64)

    for radar_id in range(inst.radar_num):
        candidate_ids = np.flatnonzero((radar == radar_id) & interval_valid)
        if candidate_ids.size == 0:
            continue
        endpoints = np.unique(np.concatenate([begin[candidate_ids], end[candidate_ids]]))
        if endpoints.size < 2:
            interval_valid[candidate_ids] = False
            continue
        local_start = np.searchsorted(endpoints, begin[candidate_ids], side="left")
        local_stop = np.searchsorted(endpoints, end[candidate_ids], side="left")
        segment_start[candidate_ids] = local_start
        segment_stop[candidate_ids] = local_stop
        segment_count = endpoints.size - 1
        segment_counts[radar_id] = segment_count

    max_segments = int(segment_counts.max(initial=0))
    if max_segments <= 0:
        raise ValueError("Instance has no valid continuous-time channel intervals")
    segment_capacity = np.zeros((inst.radar_num, max_segments), dtype=np.int16)
    for radar_id, segment_count in enumerate(segment_counts):
        if segment_count > 0:
            segment_capacity[radar_id, :segment_count] = inst.radar_capacity[radar_id]

    launch_gap = np.asarray(
        getattr(inst, "weapon_launch_gap", np.zeros(inst.weapon_num)), dtype=np.int64
    ).reshape(-1)
    max_gap = int(launch_gap.max(initial=0))
    launch_slots = inst.time_num + max(0, max_gap - 1)
    if max_gap > 0:
        offsets = np.arange(max_gap, dtype=np.int64)
        candidate_grid = np.repeat(np.arange(candidate_count), max_gap)
        offset_grid = np.tile(offsets, candidate_count)
        membership_valid = (
            interval_valid[candidate_grid]
            & (offset_grid < launch_gap[weapon[candidate_grid]])
        )
        resource_candidates = candidate_grid[membership_valid]
        resource_keys = (
            weapon[resource_candidates] * launch_slots
            + launch[resource_candidates]
            + offset_grid[membership_valid]
        )
        launch_resource_table = _membership_table(
            resource_keys,
            resource_candidates,
            inst.weapon_num * launch_slots,
        )
    else:
        launch_resource_table = np.full((1, 1), -1, dtype=np.int64)

    identity_count = inst.weapon_num * inst.target_num * inst.time_num
    identity_table = np.full((identity_count, inst.radar_num), -1, dtype=np.int64)
    identity_table[identity, radar] = np.arange(candidate_count, dtype=np.int64)
    identity_unit_probability = np.zeros((identity_count, inst.radar_num), dtype=np.bool_)
    for radar_id in range(inst.radar_num):
        joint_probability = inst.radar_pr[radar_id].reshape(-1, order="F") * wp_flat
        identity_unit_probability[:, radar_id] = joint_probability == 1.0

    radar_target = radar * inst.target_num + target
    return DynamicCandidateData(
        radar=radar,
        weapon=weapon,
        target=target,
        launch=launch,
        identity=identity,
        begin=begin,
        end=end,
        probability=probability,
        demand=demand,
        time_norm=time_norm,
        interval_valid=interval_valid,
        segment_start=segment_start,
        segment_stop=segment_stop,
        segment_capacity=segment_capacity,
        segment_count_by_radar=segment_counts,
        target_table=_group_table(target, inst.target_num),
        weapon_table=_group_table(weapon, inst.weapon_num),
        radar_table=_group_table(radar, inst.radar_num),
        radar_target_table=_group_table(radar_target, inst.radar_num * inst.target_num),
        identity_table=identity_table,
        launch_resource_table=launch_resource_table,
        identity_unit_probability=identity_unit_probability,
        launch_slots=launch_slots,
    )


class DynamicMMRRBFGPUEvaluator:
    """Evaluate a population of RBF rules on one fixed instance."""

    def __init__(
        self,
        instance_path: Path,
        device: str = "cuda",
        hidden_neurons: int = 3,
        score_chunk: int = 2048,
        use_time_dimension: bool = True,
        external_third_feature: Path | None = None,
        external_fourth_feature: Path | None = None,
        rbf_input_mode: str = "standard",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        preprocess_start = time.perf_counter()
        self.instance_path = Path(instance_path).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected Python environment")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.dtype = dtype
        self.hidden_neurons = int(hidden_neurons)
        self.use_time_dimension = bool(use_time_dimension)
        self.rbf_input_mode = str(rbf_input_mode).lower()
        if self.rbf_input_mode not in {
            "standard",
            "hgnn_only",
            "duration_hgnn_4d",
        }:
            raise ValueError(f"Unknown RBF input mode: {rbf_input_mode}")
        self.center_dimension = (
            1
            if self.rbf_input_mode == "hgnn_only"
            else (
                4
                if self.rbf_input_mode == "duration_hgnn_4d"
                else (3 if self.use_time_dimension else 2)
            )
        )
        self.rule_dimension = (
            self.hidden_neurons * self.center_dimension + 1 + self.hidden_neurons
        )
        self.score_chunk = max(1, int(score_chunk))

        self.inst = load_dynamic_instance(self.instance_path)
        self.candidates = build_dynamic_candidates(self.inst)
        cand = self.candidates

        self.external_third_feature = (
            Path(external_third_feature).resolve()
            if external_third_feature is not None
            else None
        )
        if self.external_third_feature is not None:
            if not self.use_time_dimension and self.rbf_input_mode != "hgnn_only":
                raise ValueError("An external third feature requires a three-dimensional RBF")
            third_feature = np.asarray(
                np.load(self.external_third_feature), dtype=np.float64
            ).reshape(-1)
            if third_feature.size != cand.count:
                raise ValueError(
                    "External third feature has "
                    f"{third_feature.size} values; expected {cand.count}"
                )
            if not np.all(np.isfinite(third_feature)):
                raise ValueError("External third feature contains non-finite values")
            if np.any(third_feature < -1e-12) or np.any(third_feature > 1.0 + 1e-12):
                raise ValueError("External third feature must be normalized to [0, 1]")
            third_feature = np.clip(third_feature, 0.0, 1.0)
            self.third_feature_name = "external_hgnn_score"
        else:
            if self.rbf_input_mode == "hgnn_only":
                raise ValueError("hgnn_only mode requires --external-third-feature")
            third_feature = cand.time_norm
            self.third_feature_name = "normalized_short_channel_occupancy"

        self.external_fourth_feature = (
            Path(external_fourth_feature).resolve()
            if external_fourth_feature is not None
            else None
        )
        if self.rbf_input_mode == "duration_hgnn_4d":
            if not self.use_time_dimension:
                raise ValueError("duration_hgnn_4d requires the time dimension")
            if self.external_third_feature is not None:
                raise ValueError(
                    "duration_hgnn_4d keeps duration as the third input; "
                    "use --external-fourth-feature for the HGNN score"
                )
            if self.external_fourth_feature is None:
                raise ValueError(
                    "duration_hgnn_4d requires --external-fourth-feature"
                )
            fourth_feature = np.asarray(
                np.load(self.external_fourth_feature), dtype=np.float64
            ).reshape(-1)
            if fourth_feature.size != cand.count:
                raise ValueError(
                    "External fourth feature has "
                    f"{fourth_feature.size} values; expected {cand.count}"
                )
            if not np.all(np.isfinite(fourth_feature)):
                raise ValueError("External fourth feature contains non-finite values")
            if np.any(fourth_feature < -1e-12) or np.any(fourth_feature > 1.0 + 1e-12):
                raise ValueError("External fourth feature must be normalized to [0, 1]")
            fourth_feature = np.clip(fourth_feature, 0.0, 1.0)
            self.fourth_feature_name = "external_hgnn_score"
        else:
            if self.external_fourth_feature is not None:
                raise ValueError(
                    "An external fourth feature requires duration_hgnn_4d mode"
                )
            fourth_feature = np.zeros(cand.count, dtype=np.float64)
            self.fourth_feature_name = None

        def tensor(value: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
            return torch.as_tensor(value, device=self.device, dtype=dtype)

        self.radar = tensor(cand.radar, torch.long)
        self.weapon = tensor(cand.weapon, torch.long)
        self.target = tensor(cand.target, torch.long)
        self.identity = tensor(cand.identity, torch.long)
        self.probability = tensor(cand.probability, self.dtype)
        self.demand = tensor(cand.demand, torch.long)
        self.max_demand = int(cand.demand.max(initial=1))
        self.time_norm = tensor(cand.time_norm, self.dtype)
        self.third_feature = tensor(third_feature, self.dtype)
        self.fourth_feature = tensor(fourth_feature, self.dtype)
        self.launch = tensor(cand.launch, torch.long)
        launch_gap = np.asarray(self.inst.weapon_launch_gap, dtype=np.int64)
        self.launch_gap = tensor(launch_gap[cand.weapon], torch.long)
        self.max_launch_gap = int(launch_gap.max(initial=0))
        self.launch_slots = int(cand.launch_slots)
        self.interval_valid = tensor(cand.interval_valid, torch.bool)
        self.segment_start = tensor(cand.segment_start, torch.long)
        self.segment_stop = tensor(cand.segment_stop, torch.long)
        self.segment_capacity = tensor(cand.segment_capacity, torch.int16)
        self.segment_count_by_radar = tensor(cand.segment_count_by_radar, torch.long)
        self.segment_position = torch.arange(
            cand.max_segments_per_radar, device=self.device, dtype=torch.long
        )
        self.target_table = tensor(cand.target_table, torch.long)
        self.weapon_table = tensor(cand.weapon_table, torch.long)
        self.radar_table = tensor(cand.radar_table, torch.long)
        self.radar_target_table = tensor(cand.radar_target_table, torch.long)
        self.identity_table = tensor(cand.identity_table, torch.long)
        self.launch_resource_table = tensor(cand.launch_resource_table, torch.long)
        self.identity_unit_probability = tensor(cand.identity_unit_probability, torch.bool)
        self.target_value_max = max(float(np.max(self.inst.target_value)), 1e-12)
        self.target_base = tensor(
            self.inst.target_value / self.target_value_max, self.dtype
        )
        self.target_value = tensor(self.inst.target_value, self.dtype)
        self.has_unit_probability = bool(np.any(cand.identity_unit_probability))
        self.feasible_quad_count = count_feasible_quads(self.inst)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.preprocess_sec = time.perf_counter() - preprocess_start

    def decode_rules(self, weights: torch.Tensor) -> DecodedRules:
        if weights.ndim != 2 or weights.shape[1] != self.rule_dimension:
            raise ValueError(
                f"Expected a population N x {self.rule_dimension}, found {tuple(weights.shape)}"
            )
        weights = weights.to(device=self.device, dtype=self.dtype)
        hidden = self.hidden_neurons
        center_stop = hidden * self.center_dimension
        spread_index = center_stop
        centers = (
            weights[:, :center_stop]
            .reshape(-1, self.center_dimension, hidden)
            .transpose(1, 2)
            .contiguous()
        )
        spreads = weights[:, spread_index].clamp_min(1e-8)
        output_weights = weights[:, spread_index + 1 : spread_index + 1 + hidden]
        output_sum = output_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        output_weights = output_weights / output_sum
        return DecodedRules(
            centers=centers,
            inv_two_spread2=1.0 / (2.0 * spreads.square()),
            output_weights=output_weights,
        )

    def _score_pairs(
        self,
        rules: DecodedRules,
        rule_rows: torch.Tensor,
        candidate_ids: torch.Tensor,
        survival: torch.Tensor,
    ) -> torch.Tensor:
        target = self.target[candidate_ids]
        threat = self.target_base[target] * survival[rule_rows, target]
        probability = self.probability[candidate_ids]
        value = torch.zeros_like(probability)
        for center_id in range(self.hidden_neurons):
            center = rules.centers[rule_rows, center_id]
            if self.rbf_input_mode == "hgnn_only":
                distance2 = (
                    self.third_feature[candidate_ids] - center[:, 0]
                ).square()
            else:
                distance2 = (
                    (threat - center[:, 0]).square()
                    + (probability - center[:, 1]).square()
                )
            if self.rbf_input_mode == "standard" and self.use_time_dimension:
                distance2.add_(
                    (self.third_feature[candidate_ids] - center[:, 2]).square()
                )
            elif self.rbf_input_mode == "duration_hgnn_4d":
                distance2.add_(
                    (self.third_feature[candidate_ids] - center[:, 2]).square()
                )
                distance2.add_(
                    (self.fourth_feature[candidate_ids] - center[:, 3]).square()
                )
            value.add_(
                torch.exp(-distance2 * rules.inv_two_spread2[rule_rows])
                * rules.output_weights[rule_rows, center_id]
            )
        return value.abs_()

    def _initial_scores(self, rules: DecodedRules, survival: torch.Tensor) -> torch.Tensor:
        batch = rules.centers.shape[0]
        candidate_count = self.candidates.count
        scores = torch.empty((batch, candidate_count), dtype=self.dtype, device=self.device)
        rows = torch.arange(batch, device=self.device, dtype=torch.long)
        for start in range(0, candidate_count, self.score_chunk):
            stop = min(candidate_count, start + self.score_chunk)
            candidate_ids = torch.arange(start, stop, device=self.device, dtype=torch.long)
            rule_rows = rows[:, None].expand(-1, stop - start).reshape(-1)
            pair_candidates = candidate_ids[None, :].expand(batch, -1).reshape(-1)
            scores[:, start:stop] = self._score_pairs(
                rules, rule_rows, pair_candidates, survival
            ).reshape(batch, stop - start)
        return scores

    @staticmethod
    def _valid_pairs(rows: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row_grid = rows[:, None].expand_as(ids)
        valid = ids >= 0
        return row_grid[valid], ids[valid]

    def _deactivate_table(
        self,
        rows: torch.Tensor,
        group_ids: torch.Tensor,
        table: torch.Tensor,
        active: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        if rows.numel() == 0:
            return
        ids = table[group_ids]
        rule_rows, candidate_ids = self._valid_pairs(rows, ids)
        active[rule_rows, candidate_ids] = False
        scores[rule_rows, candidate_ids] = -torch.inf

    def _refresh_target_scores(
        self,
        rules: DecodedRules,
        rows: torch.Tensor,
        target_ids: torch.Tensor,
        survival: torch.Tensor,
        active: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        if rows.numel() == 0:
            return
        ids = self.target_table[target_ids]
        rule_rows, candidate_ids = self._valid_pairs(rows, ids)
        values = self._score_pairs(rules, rule_rows, candidate_ids, survival)
        scores[rule_rows, candidate_ids] = torch.where(
            active[rule_rows, candidate_ids], values, -torch.inf
        )

    def _prune_infeasible_segments(
        self,
        rows: torch.Tensor,
        radar_ids: torch.Tensor,
        shortage_prefix: torch.Tensor,
        active: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        """Remove intervals whose target-specific demand exceeds residual capacity."""
        if rows.numel() == 0:
            return
        ids = self.radar_table[radar_ids]
        rule_rows, candidate_ids = self._valid_pairs(rows, ids)
        demand_index = self.demand[candidate_ids] - 1
        blocked = (
            shortage_prefix[
                rule_rows,
                self.radar[candidate_ids],
                demand_index,
                self.segment_stop[candidate_ids],
            ]
            - shortage_prefix[
                rule_rows,
                self.radar[candidate_ids],
                demand_index,
                self.segment_start[candidate_ids],
            ]
        ) > 0
        blocked_rows = rule_rows[blocked]
        blocked_candidates = candidate_ids[blocked]
        active[blocked_rows, blocked_candidates] = False
        scores[blocked_rows, blocked_candidates] = -torch.inf

    @torch.inference_mode()
    def evaluate(self, weights: torch.Tensor, record_selection: bool = False) -> EvaluationResult:
        weights = weights.to(device=self.device, dtype=self.dtype)
        rules = self.decode_rules(weights)
        batch = weights.shape[0]
        candidate_count = self.candidates.count
        rows_all = torch.arange(batch, device=self.device, dtype=torch.long)
        survival = torch.ones((batch, self.inst.target_num), dtype=self.dtype, device=self.device)
        ammo = torch.as_tensor(self.inst.ammo, dtype=torch.int32, device=self.device).repeat(batch, 1)
        strike = torch.as_tensor(self.inst.strike, dtype=torch.int32, device=self.device).repeat(batch, 1)
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
        if np.any(self.inst.ammo <= 0):
            invalid_weapon = torch.as_tensor(
                np.isin(self.candidates.weapon, np.flatnonzero(self.inst.ammo <= 0)),
                device=self.device,
            )
            active[:, invalid_weapon] = False
        if np.any(self.inst.strike <= 0):
            invalid_target = torch.as_tensor(
                np.isin(self.candidates.target, np.flatnonzero(self.inst.strike <= 0)),
                device=self.device,
            )
            active[:, invalid_target] = False

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        scoring_start = time.perf_counter()
        scores = self._initial_scores(rules, survival)
        scores.masked_fill_(~active, -torch.inf)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        initial_scoring_sec = time.perf_counter() - scoring_start

        selected_count = torch.zeros(batch, dtype=torch.int32, device=self.device)
        channel_rejections = torch.zeros(batch, dtype=torch.int32, device=self.device)
        max_accepts = int(min(self.inst.ammo.sum(), self.inst.strike.sum()))
        selection_history = None
        if record_selection:
            selection_history = torch.full(
                (batch, max_accepts), -1, dtype=torch.int32, device=self.device
            )

        decode_start = time.perf_counter()
        iterations = 0
        for iteration in range(candidate_count + 1):
            iterations = iteration + 1
            best_score, chosen = scores.max(dim=1)
            live = torch.isfinite(best_score) & (best_score > 0)
            if not bool(torch.any(live)):
                break

            rows = rows_all[live]
            chosen = chosen[live]
            starts = self.segment_start[chosen]
            stops = self.segment_stop[chosen]
            chosen_radar = self.radar[chosen]
            chosen_demand = self.demand[chosen]
            blocked = (
                shortage_prefix[rows, chosen_radar, chosen_demand - 1, stops]
                - shortage_prefix[rows, chosen_radar, chosen_demand - 1, starts]
            ) > 0
            if bool(torch.any(blocked)):
                blocked_rows = rows[blocked]
                blocked_candidates = chosen[blocked]
                active[blocked_rows, blocked_candidates] = False
                scores[blocked_rows, blocked_candidates] = -torch.inf
                channel_rejections[blocked_rows] += 1

            feasible = ~blocked
            if not bool(torch.any(feasible)):
                continue
            rows = rows[feasible]
            chosen = chosen[feasible]
            starts = starts[feasible]
            stops = stops[feasible]
            radar = chosen_radar[feasible]
            demand = chosen_demand[feasible]
            weapon = self.weapon[chosen]
            target = self.target[chosen]

            interval_mask = (
                (self.segment_position[None, :] >= starts[:, None])
                & (self.segment_position[None, :] < stops[:, None])
            )
            updated_remaining = remaining[rows, radar] - (
                interval_mask.to(torch.int16) * demand[:, None].to(torch.int16)
            )
            if bool(torch.any(updated_remaining < 0)):
                raise RuntimeError("Continuous channel capacity became negative")
            remaining[rows, radar] = updated_remaining
            shortage_prefix[rows, radar, :, 0] = 0
            demand_levels = torch.arange(
                1, self.max_demand + 1, device=self.device, dtype=torch.int16
            )
            shortage_prefix[rows, radar, :, 1:] = torch.cumsum(
                updated_remaining[:, None, :] < demand_levels[None, :, None],
                dim=2,
                dtype=torch.int32,
            )

            old_count = selected_count[rows].to(torch.long)
            if selection_history is not None:
                selection_history[rows, old_count] = chosen.to(torch.int32)
            selected_count[rows] += 1
            ammo[rows, weapon] -= 1
            strike[rows, target] -= 1
            survival[rows, target] *= 1.0 - self.probability[chosen]

            identity_ids = self.identity[chosen]
            self._deactivate_table(rows, identity_ids, self.identity_table, active, scores)

            for offset in range(self.max_launch_gap):
                transfer_rows = rows[self.launch_gap[chosen] > offset]
                transfer_chosen = chosen[self.launch_gap[chosen] > offset]
                transfer_groups = (
                    self.weapon[transfer_chosen] * self.launch_slots
                    + self.launch[transfer_chosen]
                    + offset
                )
                self._deactivate_table(
                    transfer_rows,
                    transfer_groups,
                    self.launch_resource_table,
                    active,
                    scores,
                )

            ammo_empty = ammo[rows, weapon] <= 0
            self._deactivate_table(
                rows[ammo_empty], weapon[ammo_empty], self.weapon_table, active, scores
            )
            target_empty = strike[rows, target] <= 0
            self._deactivate_table(
                rows[target_empty], target[target_empty], self.target_table, active, scores
            )

            if self.has_unit_probability:
                unit = self.identity_unit_probability[identity_ids]
                unit_rows, unit_radar = torch.nonzero(unit, as_tuple=True)
                rt_rows = rows[unit_rows]
                rt_groups = unit_radar * self.inst.target_num + target[unit_rows]
                self._deactivate_table(rt_rows, rt_groups, self.radar_target_table, active, scores)

            refresh = strike[rows, target] > 0
            self._refresh_target_scores(
                rules,
                rows[refresh],
                target[refresh],
                survival,
                active,
                scores,
            )
            self._prune_infeasible_segments(
                rows, radar, shortage_prefix, active, scores
            )
        else:
            raise RuntimeError("Dynamic decoder exceeded the candidate elimination bound")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_sec = time.perf_counter() - decode_start
        fitness = ((1.0 - survival) * self.target_value[None, :]).sum(dim=1)
        return EvaluationResult(
            fitness=fitness,
            selected_count=selected_count,
            channel_rejections=channel_rejections,
            initial_scoring_sec=initial_scoring_sec,
            decode_sec=decode_sec,
            iterations=iterations,
            selection_history=selection_history,
        )

    def metadata(self) -> Dict[str, object]:
        if self.rbf_input_mode == "hgnn_only":
            score_features = [self.third_feature_name]
        elif self.rbf_input_mode == "duration_hgnn_4d":
            score_features = [
                "remaining_threat",
                "joint_probability",
                self.third_feature_name,
                self.fourth_feature_name,
            ]
        else:
            score_features = ["remaining_threat", "joint_probability"] + (
                [self.third_feature_name] if self.use_time_dimension else []
            )
        return {
            "instance": str(self.instance_path),
            "device": str(self.device),
            "dtype": "float64",
            "hidden_neurons": self.hidden_neurons,
            "score_features": score_features,
            "rbf_input_mode": self.rbf_input_mode,
            "use_time_dimension": self.use_time_dimension,
            "center_dimension": self.center_dimension,
            "rule_dimension": self.rule_dimension,
            "transfer_constraint": self.max_launch_gap > 0,
            "max_interceptor_launch_gap": self.max_launch_gap,
            "third_feature_contract": (
                "precomputed external score in [0,1]"
                if self.external_third_feature is not None
                else (
                    "clip(1-(duration-duration_min)/"
                    "(duration_max-duration_min+1e-12),0,1)"
                )
            ),
            "time_feature_active": (
                self.use_time_dimension
                and self.external_third_feature is None
                and self.rbf_input_mode in {"standard", "duration_hgnn_4d"}
            ),
            "time_feature_contract": (
                "clip(1-(duration-duration_min)/"
                "(duration_max-duration_min+1e-12),0,1)"
            ),
            "shorter_channel_occupancy_has_larger_feature": (
                self.external_third_feature is None
            ),
            "external_third_feature": (
                str(self.external_third_feature)
                if self.external_third_feature is not None
                else None
            ),
            "external_fourth_feature": (
                str(self.external_fourth_feature)
                if self.external_fourth_feature is not None
                else None
            ),
            "target_threat_contract": (
                "not_used"
                if self.rbf_input_mode == "hgnn_only"
                else "target_value/instance_target_value_max"
            ),
            "target_value_max": self.target_value_max,
            "feasible_quad_count": self.feasible_quad_count,
            "active_candidate_count": self.candidates.count,
            "continuous_segment_count": self.candidates.segment_count,
            "max_segments_per_radar": self.candidates.max_segments_per_radar,
            "preprocess_sec": self.preprocess_sec,
        }


def matlab_round_positive(value: torch.Tensor) -> torch.Tensor:
    """MATLAB round for nonnegative values (ties away from zero)."""
    return torch.floor(value + 0.5)
