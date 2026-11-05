#!/usr/bin/env python3
"""Exploratory Phase-B mechanisms: step-size tails and within-path geometry.

This module sits alongside ``phase_B_mechanisms.py`` and intentionally keeps
these candidate mechanisms separate from the main paper pipeline. It asks
whether residualised uniqueness is better explained by:

    1. Step-size heavy-tailness: are consecutive paths mostly local moves, or
       a mix of local moves plus occasional large jumps?
    2. Within-path geometric complexity: are individual paths themselves
       geometrically rich, independent of where they sit in the 2-D canvas?

Outputs land in ``analysis_outputs/phaseB_exploration_dynamics/`` by default.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from scipy.stats import kurtosis, skew  # noqa: E402

from analyze_ogt_groups import (  # noqa: E402
    ParticipantRoutes,
    compute_uniqueness_metrics,
    load_controls,
    load_patients,
    moving_average,
    to_float,
    write_csv,
)
from motor_purification import (  # noqa: E402
    DEFAULT_MOTOR_RESIDUALISER_COLUMNS,
    compute_motor_fingerprint,
    compute_position_only_uniqueness_metrics,
    residualise_column,
)
from phase_B_mechanisms import (  # noqa: E402
    DEFAULT_GRID_X_CELLS,
    DEFAULT_GRID_Y_CELLS,
    DEFAULT_MECHANISM_COLUMNS,
    _mirror_aware_distances,
    build_feature_cache,
    compare_group_rows,
    compute_deliberation,
    compute_perseveration,
    compute_roaming_entropy,
    compute_strategy_flexibility,
    compute_submovements,
    compute_temporal_uniqueness_slope,
    correlate_columns,
    shapley_r_squared,
)


TAIL_METRIC_COLUMNS = [
    "step_distance_mean",
    "step_distance_median",
    "step_distance_cv",
    "step_distance_skew",
    "step_distance_excess_kurtosis",
    "step_distance_hill_alpha_top20",
    "step_distance_big_jump_fraction_q75",
    "step_distance_big_jump_fraction_q90",
]

GEOMETRY_METRIC_COLUMNS = [
    "path_tortuosity_mean",
    "path_tortuosity_median",
    "path_heading_change_entropy_mean",
    "path_heading_change_entropy_median",
    "path_abs_turn_mean_rad",
    "path_curvature_variance_mean",
]

NEW_MECHANISM_COLUMNS = [
    "step_distance_excess_kurtosis",
    "step_distance_hill_alpha_top20",
    "step_distance_big_jump_fraction_q90",
    "path_tortuosity_mean",
    "path_heading_change_entropy_mean",
    "path_curvature_variance_mean",
]


def finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def consecutive_step_distances(
    feature_cache: dict[str, object],
) -> dict[str, np.ndarray]:
    """Mirror-aware consecutive path distances per participant."""
    distance_full = np.asarray(feature_cache["distance_full"], dtype=float)
    distance_flip = np.asarray(feature_cache["distance_flip"], dtype=float)
    participant_index = feature_cache["participant_index"]
    out: dict[str, np.ndarray] = {}
    for pid, indices in participant_index.items():
        if len(indices) < 2:
            out[pid] = np.empty(0, dtype=float)
            continue
        subj_dist = _mirror_aware_distances(distance_full, distance_flip, list(indices))
        out[pid] = np.asarray(
            [subj_dist[i, i + 1] for i in range(subj_dist.shape[0] - 1)],
            dtype=float,
        )
    return out


def hill_alpha(values: np.ndarray, top_fraction: float = 0.20) -> float:
    """Hill estimator for Pareto tail alpha on the largest observations."""
    vals = np.sort(finite_array(values))
    if vals.size < 8:
        return math.nan
    k = max(3, int(math.ceil(vals.size * top_fraction)))
    if vals.size <= k:
        return math.nan
    tail = vals[-k:]
    threshold = vals[-k - 1]
    if threshold <= 0 or np.any(tail <= 0):
        return math.nan
    denom = float(np.sum(np.log(tail / threshold)))
    return float(k / denom) if denom > 0 else math.nan


def compute_step_tailness(
    feature_cache: dict[str, object],
    q75_threshold: float | None = None,
    q90_threshold: float | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    step_by_pid = consecutive_step_distances(feature_cache)
    pooled = finite_array([v for vals in step_by_pid.values() for v in vals])
    thresholds = {
        "step_distance_big_jump_threshold_q75": float(np.quantile(pooled, 0.75))
        if pooled.size
        else math.nan,
        "step_distance_big_jump_threshold_q90": float(np.quantile(pooled, 0.90))
        if pooled.size
        else math.nan,
    }
    if q75_threshold is not None:
        thresholds["step_distance_big_jump_threshold_q75"] = q75_threshold
    if q90_threshold is not None:
        thresholds["step_distance_big_jump_threshold_q90"] = q90_threshold

    out: dict[str, dict[str, float]] = {}
    for pid, vals_raw in step_by_pid.items():
        vals = finite_array(vals_raw)
        if vals.size < 3:
            out[pid] = {metric: math.nan for metric in TAIL_METRIC_COLUMNS}
            out[pid]["step_distance_n"] = int(vals.size)
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else math.nan
        q75 = thresholds["step_distance_big_jump_threshold_q75"]
        q90 = thresholds["step_distance_big_jump_threshold_q90"]
        out[pid] = {
            "step_distance_mean": mean,
            "step_distance_median": float(np.median(vals)),
            "step_distance_cv": sd / mean if mean > 0 else math.nan,
            "step_distance_skew": float(skew(vals, bias=False)) if vals.size >= 3 else math.nan,
            "step_distance_excess_kurtosis": float(kurtosis(vals, fisher=True, bias=False))
            if vals.size >= 4
            else math.nan,
            "step_distance_hill_alpha_top20": hill_alpha(vals, top_fraction=0.20),
            "step_distance_big_jump_fraction_q75": float(np.mean(vals > q75))
            if np.isfinite(q75)
            else math.nan,
            "step_distance_big_jump_fraction_q90": float(np.mean(vals > q90))
            if np.isfinite(q90)
            else math.nan,
            "step_distance_n": int(vals.size),
        }
    return out, thresholds


def wrapped_angle_diff(angles: np.ndarray) -> np.ndarray:
    return (np.diff(angles) + math.pi) % (2 * math.pi) - math.pi


def path_geometry(route: np.ndarray, n_turn_bins: int, smooth_window: int) -> dict[str, float] | None:
    if route.shape[0] < 5:
        return None
    x = route[:, 1]
    y = route[:, 2]
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    valid = seg > 0
    if np.count_nonzero(valid) < 4:
        return None

    arc = float(np.sum(seg[valid]))
    endpoint = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    tortuosity = arc / endpoint if endpoint > 0 else math.nan

    angles = np.arctan2(dy[valid], dx[valid])
    turns = wrapped_angle_diff(angles)
    if smooth_window > 1 and turns.size >= smooth_window:
        turns_for_variance = moving_average(turns, smooth_window)
    else:
        turns_for_variance = turns

    abs_turns = np.abs(turns)
    if abs_turns.size:
        counts, _ = np.histogram(abs_turns, bins=np.linspace(0, math.pi, n_turn_bins + 1))
        probs = counts.astype(float) / counts.sum() if counts.sum() else np.array([])
        entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        entropy_norm = entropy / math.log(n_turn_bins) if n_turn_bins > 1 else math.nan
    else:
        entropy_norm = math.nan

    return {
        "path_tortuosity": tortuosity,
        "path_heading_change_entropy": entropy_norm,
        "path_abs_turn_mean_rad": float(np.mean(abs_turns)) if abs_turns.size else math.nan,
        "path_curvature_variance": float(np.var(turns_for_variance, ddof=1))
        if turns_for_variance.size > 1
        else math.nan,
    }


def compute_within_path_geometry(
    participants: Sequence[ParticipantRoutes],
    n_turn_bins: int = 12,
    smooth_window: int = 5,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for participant in participants:
        per_path = [
            geom
            for route in participant.test_routes
            if (geom := path_geometry(route, n_turn_bins=n_turn_bins, smooth_window=smooth_window))
            is not None
        ]
        if not per_path:
            out[participant.participant_id] = {metric: math.nan for metric in GEOMETRY_METRIC_COLUMNS}
            out[participant.participant_id]["path_geometry_n_paths"] = 0
            continue

        def vals(name: str) -> np.ndarray:
            return finite_array([row[name] for row in per_path])

        tort = vals("path_tortuosity")
        ent = vals("path_heading_change_entropy")
        turn = vals("path_abs_turn_mean_rad")
        curv = vals("path_curvature_variance")
        out[participant.participant_id] = {
            "path_tortuosity_mean": float(np.mean(tort)) if tort.size else math.nan,
            "path_tortuosity_median": float(np.median(tort)) if tort.size else math.nan,
            "path_heading_change_entropy_mean": float(np.mean(ent)) if ent.size else math.nan,
            "path_heading_change_entropy_median": float(np.median(ent)) if ent.size else math.nan,
            "path_abs_turn_mean_rad": float(np.mean(turn)) if turn.size else math.nan,
            "path_curvature_variance_mean": float(np.mean(curv)) if curv.size else math.nan,
            "path_geometry_n_paths": len(per_path),
        }
    return out


def build_rows(
    participants: Sequence[ParticipantRoutes],
    base_rows: list[dict[str, object]],
    tailness: dict[str, dict[str, float]],
    geometry: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    base_by_pid = {str(row["participant_id"]): row for row in base_rows}
    rows: list[dict[str, object]] = []
    for participant in participants:
        pid = participant.participant_id
        row: dict[str, object] = {}
        row.update(base_by_pid.get(pid, {}))
        row.update(tailness.get(pid, {}))
        row.update(geometry.get(pid, {}))
        rows.append(row)
    return rows


def full_drop_one_r2(
    rows: list[dict[str, object]],
    target: str,
    predictors: Sequence[str],
    group_filter: str | None = "patient",
) -> list[dict[str, object]]:
    y_vals: list[float] = []
    x_vals: list[list[float]] = []
    preds = list(predictors)
    for row in rows:
        if group_filter is not None and row.get("group") != group_filter:
            continue
        y = to_float(row.get(target, math.nan))
        x = [to_float(row.get(p, math.nan)) for p in preds]
        if np.isfinite(y) and all(np.isfinite(v) for v in x):
            y_vals.append(y)
            x_vals.append(x)
    y_arr = np.asarray(y_vals, dtype=float)
    x_arr = np.asarray(x_vals, dtype=float)
    n = int(y_arr.size)
    if n < len(preds) + 3:
        return [
            {
                "predictor": predictor,
                "full_model_r_squared": math.nan,
                "drop_one_r_squared": math.nan,
                "unique_drop_one_r_squared": math.nan,
                "n": n,
            }
            for predictor in preds
        ]

    def r2(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        x_model = np.column_stack([np.ones(x.shape[0]), x])
        beta, *_ = np.linalg.lstsq(x_model, y_arr, rcond=None)
        fitted = x_model @ beta
        ss_res = float(np.sum((y_arr - fitted) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan

    total = r2(x_arr)
    out: list[dict[str, object]] = []
    for idx, predictor in enumerate(preds):
        keep = [j for j in range(len(preds)) if j != idx]
        reduced_r2 = r2(x_arr[:, keep]) if keep else 0.0
        out.append(
            {
                "predictor": predictor,
                "full_model_r_squared": total,
                "drop_one_r_squared": reduced_r2,
                "unique_drop_one_r_squared": total - reduced_r2
                if np.isfinite(total) and np.isfinite(reduced_r2)
                else math.nan,
                "n": n,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-csv", type=Path, default=Path("control_demographics_sorted.csv"))
    parser.add_argument("--control-dir", type=Path, default=Path("data_controls"))
    parser.add_argument("--patient-octal-dir", type=Path, default=Path("data_patients/OCTAL"))
    parser.add_argument(
        "--patient-jatos-dirs",
        nargs="+",
        type=Path,
        default=[
            Path("data_patients/OGT_English"),
            Path("data_patients/OGT_Mandarin"),
            Path("data_patients/OGT_Spanish"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/phaseB_exploration_dynamics"))
    parser.add_argument("--np-points", type=int, default=200)
    parser.add_argument("--smooth-window", type=int, default=40)
    parser.add_argument("--velocity-downsample", type=int, default=10)
    parser.add_argument("--distance-mode", choices=["full", "position"], default="full")
    parser.add_argument("--grid-x-cells", type=int, default=DEFAULT_GRID_X_CELLS)
    parser.add_argument("--grid-y-cells", type=int, default=DEFAULT_GRID_Y_CELLS)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--n-turn-bins", type=int, default=12)
    parser.add_argument("--turn-smooth-window", type=int, default=5)
    parser.add_argument("--shapley-target", default="test_uniqueness_residualised")
    parser.add_argument("--residualiser-columns", nargs="+", default=DEFAULT_MOTOR_RESIDUALISER_COLUMNS)
    parser.add_argument("--residualiser-group", choices=["patient", "control", "pooled"], default="patient")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controls = load_controls(args.control_csv, args.control_dir)
    patients = load_patients(args.patient_octal_dir, args.patient_jatos_dirs)
    participants: list[ParticipantRoutes] = list(controls) + list(patients)
    print(f"Loaded {len(controls)} controls and {len(patients)} patients.")

    feature_cache = build_feature_cache(
        participants,
        np_points=args.np_points,
        smooth_window=args.smooth_window,
        velocity_downsample=args.velocity_downsample,
        distance_mode=args.distance_mode,
    )

    motor_fp = {
        p.participant_id: compute_motor_fingerprint(p, smooth_window=args.smooth_window)
        for p in participants
    }
    uniqueness = compute_uniqueness_metrics(
        participants,
        np_points=args.np_points,
        smooth_window=args.smooth_window,
        velocity_downsample=args.velocity_downsample,
    )
    position_uniqueness = compute_position_only_uniqueness_metrics(participants, np_points=args.np_points)
    perseveration = compute_perseveration(feature_cache)
    temporal = compute_temporal_uniqueness_slope(feature_cache)
    deliberation = compute_deliberation(participants, feature_cache)
    roaming = compute_roaming_entropy(
        participants,
        grid_x_cells=args.grid_x_cells,
        grid_y_cells=args.grid_y_cells,
    )
    flexibility, _ = compute_strategy_flexibility(feature_cache, n_clusters=args.n_clusters)
    submovements = compute_submovements(participants, smooth_window=args.smooth_window)

    base_rows = []
    for participant in participants:
        pid = participant.participant_id
        row: dict[str, object] = {
            "group": participant.group,
            "participant_id": pid,
            "source_id": participant.source_id,
            "source_platform": participant.source_platform,
            "raw_source": participant.metadata.get("raw_source", ""),
        }
        row.update(motor_fp.get(pid, {}))
        row.update(uniqueness.get(pid, {}))
        row.update(position_uniqueness.get(pid, {}))
        row.update(perseveration.get(pid, {}))
        row.update(temporal.get(pid, {}))
        row.update(deliberation.get(pid, {}))
        row.update(roaming.get(pid, {}))
        row.update(flexibility.get(pid, {}))
        row.update(submovements.get(pid, {}))
        base_rows.append(row)

    tailness, thresholds = compute_step_tailness(feature_cache)
    geometry = compute_within_path_geometry(
        participants,
        n_turn_bins=args.n_turn_bins,
        smooth_window=args.turn_smooth_window,
    )
    participant_rows = build_rows(participants, base_rows, tailness, geometry)

    group_filter = None if args.residualiser_group == "pooled" else args.residualiser_group
    residuals, _, _ = residualise_column(
        participant_rows,
        target="test_uniqueness_mean",
        predictors=args.residualiser_columns,
        group_filter=group_filter,
    )
    for row, residual in zip(participant_rows, residuals):
        row["test_uniqueness_residualised"] = residual

    metric_columns = TAIL_METRIC_COLUMNS + GEOMETRY_METRIC_COLUMNS
    group_rows = compare_group_rows(participant_rows, metric_columns)
    patient_rows = [row for row in participant_rows if row.get("group") == "patient"]
    uniqueness_rows = correlate_columns(
        patient_rows,
        columns_x=metric_columns,
        columns_y=["test_uniqueness_mean", "test_uniqueness_residualised"],
    )

    new_only_shapley = shapley_r_squared(
        participant_rows,
        target=args.shapley_target,
        predictors=NEW_MECHANISM_COLUMNS,
        group_filter="patient",
    )
    core_plus_new = list(DEFAULT_MECHANISM_COLUMNS) + NEW_MECHANISM_COLUMNS
    core_plus_new_shapley = shapley_r_squared(
        participant_rows,
        target=args.shapley_target,
        predictors=core_plus_new,
        group_filter="patient",
    )
    core_plus_new_unique = full_drop_one_r2(
        participant_rows,
        target=args.shapley_target,
        predictors=core_plus_new,
        group_filter="patient",
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "phaseB_exploration_dynamics_participant_metrics.csv",
        participant_rows,
        list(participant_rows[0].keys()) if participant_rows else [],
    )
    write_csv(
        output_dir / "phaseB_exploration_dynamics_group_comparison.csv",
        group_rows,
        list(group_rows[0].keys()) if group_rows else [],
    )
    write_csv(
        output_dir / "phaseB_exploration_dynamics_uniqueness_correlations.csv",
        uniqueness_rows,
        list(uniqueness_rows[0].keys()) if uniqueness_rows else [],
    )
    write_csv(
        output_dir / "phaseB_exploration_dynamics_shapley_new_only.csv",
        new_only_shapley,
        list(new_only_shapley[0].keys()) if new_only_shapley else [],
    )
    write_csv(
        output_dir / "phaseB_exploration_dynamics_shapley_core_plus_new.csv",
        core_plus_new_shapley,
        list(core_plus_new_shapley[0].keys()) if core_plus_new_shapley else [],
    )
    write_csv(
        output_dir / "phaseB_exploration_dynamics_unique_drop_one_core_plus_new.csv",
        core_plus_new_unique,
        list(core_plus_new_unique[0].keys()) if core_plus_new_unique else [],
    )
    write_csv(
        output_dir / "phaseB_exploration_dynamics_thresholds.csv",
        [thresholds],
        list(thresholds.keys()),
    )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
