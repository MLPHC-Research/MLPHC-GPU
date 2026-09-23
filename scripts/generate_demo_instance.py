#!/usr/bin/env python
"""Generate a small deterministic SITA-TW instance for a smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "demo_instance.mat",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(2026091101)
    radar_num, weapon_num, target_num, time_num = 2, 3, 8, 2
    shape = (weapon_num, target_num, time_num)

    weapon_pr = rng.uniform(0.55, 0.9, size=shape)
    target_value = np.linspace(70.0, 140.0, target_num)
    weapon_vp = weapon_pr * target_value[None, :, None]
    radar_pr_cell = np.empty((radar_num, 1), dtype=object)
    radar_begin_cell = np.empty((radar_num, 1), dtype=object)
    radar_end_cell = np.empty((radar_num, 1), dtype=object)
    for radar in range(radar_num):
        radar_pr_cell[radar, 0] = rng.uniform(0.6, 0.95, size=shape)
        begin = rng.integers(0, 18, size=shape).astype(np.float64)
        duration = rng.integers(2, 7, size=shape).astype(np.float64)
        radar_begin_cell[radar, 0] = begin
        radar_end_cell[radar, 0] = begin + duration

    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        args.output,
        {
            "radar_num": np.array([[radar_num]], dtype=np.float64),
            "weapon_pr_matrix": weapon_pr,
            "weapon_vp_matrix": weapon_vp,
            "danyao_constraint": np.array([[3], [3], [2]], dtype=np.float64),
            "strike_constraint": np.full((target_num, 1), 2.0),
            "radar_constraint": np.array([[2], [2]], dtype=np.float64),
            "V_matrix": target_value[:, None],
            "radar_tbegin_cell": radar_begin_cell,
            "radar_tend_cell": radar_end_cell,
            "radar_pr_cell": radar_pr_cell,
        },
        do_compression=True,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
