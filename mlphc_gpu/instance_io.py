"""Instance loading utilities for the SITA-TW benchmark format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from scipy.io import loadmat


@dataclass
class InstanceData:
    radar_num: int
    weapon_num: int
    target_num: int
    time_num: int
    weapon_pr: np.ndarray
    weapon_vp: np.ndarray
    ammo: np.ndarray
    strike: np.ndarray
    radar_capacity: np.ndarray
    target_value: np.ndarray
    radar_begin: List[np.ndarray]
    radar_end: List[np.ndarray]
    radar_pr: List[np.ndarray]


def _numeric_cells(value: np.ndarray, radar_num: int) -> List[np.ndarray]:
    arr = np.asarray(value)
    if arr.dtype == object:
        cells = [np.asarray(x, dtype=np.float64) for x in arr.ravel(order="F")]
    elif radar_num == 1:
        cells = [np.asarray(arr, dtype=np.float64)]
    else:
        raise ValueError("Expected a MATLAB cell array for radar data")
    if len(cells) != radar_num:
        raise ValueError(f"Expected {radar_num} radar cells, found {len(cells)}")
    return cells


def load_instance(path: Path) -> InstanceData:
    raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    radar_num = int(np.asarray(raw["radar_num"]).item())
    weapon_pr = np.asarray(raw["weapon_pr_matrix"], dtype=np.float64)
    weapon_vp = np.asarray(raw["weapon_vp_matrix"], dtype=np.float64)
    weapon_num, target_num, time_num = weapon_pr.shape
    radar_capacity = np.asarray(
        raw["radar_constraint"], dtype=np.float64
    ).reshape(-1, order="F")
    if radar_capacity.size == 1 and radar_num > 1:
        radar_capacity = np.repeat(radar_capacity, radar_num)
    return InstanceData(
        radar_num=radar_num,
        weapon_num=weapon_num,
        target_num=target_num,
        time_num=time_num,
        weapon_pr=weapon_pr,
        weapon_vp=weapon_vp,
        ammo=np.rint(
            np.asarray(raw["danyao_constraint"], dtype=np.float64).reshape(
                -1, order="F"
            )
        ).astype(np.int64),
        strike=np.rint(
            np.asarray(raw["strike_constraint"], dtype=np.float64).reshape(
                -1, order="F"
            )
        ).astype(np.int64),
        radar_capacity=np.rint(radar_capacity).astype(np.int64),
        target_value=np.asarray(raw["V_matrix"], dtype=np.float64).reshape(
            -1, order="F"
        ),
        radar_begin=_numeric_cells(raw["radar_tbegin_cell"], radar_num),
        radar_end=_numeric_cells(raw["radar_tend_cell"], radar_num),
        radar_pr=_numeric_cells(raw["radar_pr_cell"], radar_num),
    )


def count_feasible_quads(inst: InstanceData) -> int:
    """Count quads with positive radar and interceptor probabilities."""
    return int(
        sum(
            np.count_nonzero((radar_pr > 0.0) & (inst.weapon_pr > 0.0))
            for radar_pr in inst.radar_pr
        )
    )
