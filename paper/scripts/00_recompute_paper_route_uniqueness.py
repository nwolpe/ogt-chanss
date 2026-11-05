#!/usr/bin/env python3
"""Recompute paper participant metrics with the 100-control sample.

This script reuses the existing raw-data loaders from ``analyze_ogt_groups.py``
but overrides the control-session selection so a participant with only a ``_2``
session is retained. It then recomputes route-level Ang-style uniqueness against
the full 100-control + 150-patient route set using a blockwise nearest-neighbour
calculation, avoiding a full in-memory pairwise distance matrix.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(".")
OUTPUT_DIR = ROOT / "paper" / "outputs" / "shared"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYZE_PATH = ROOT / "analyze_ogt_groups.py"
CONTROL_CSV = ROOT / "control_demographics_sorted.csv"
CONTROL_DIR = ROOT / "data_controls"
PATIENT_OCTAL_DIR = ROOT / "data_patients" / "OCTAL"
PATIENT_JATOS_DIRS = [
    ROOT / "data_patients" / "OGT_English",
    ROOT / "data_patients" / "OGT_Mandarin",
    ROOT / "data_patients" / "OGT_Spanish",
]

NP_POINTS = 200
SMOOTH_WINDOW = 40
VELOCITY_DOWNSAMPLE = 10
BLOCK_SIZE = 128
X_CENTER = 400.0
Y_CENTER = 600.0


def load_analyze_namespace() -> dict:
    """Exec loader/helper definitions from analyze_ogt_groups without SciPy."""
    source = ANALYZE_PATH.read_text(encoding="utf-8")
    filtered_lines: list[str] = []
    skip_block = False
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import matplotlib") or stripped.startswith("matplotlib.use"):
            continue
        if stripped.startswith("from scipy."):
            if stripped.endswith("("):
                skip_block = True
            continue
        if skip_block:
            if stripped == ")":
                skip_block = False
            continue
        if stripped.startswith("import matplotlib.pyplot"):
            continue
        filtered_lines.append(line)

    ns: dict = {
        "__name__": "paper_loader_namespace",
        "np": np,
        "pd": pd,
    }
    module = types.ModuleType("paper_loader_namespace")
    module.__dict__.update(ns)
    sys.modules["paper_loader_namespace"] = module
    ns = module.__dict__
    exec("\n".join(filtered_lines), ns)
    return ns


SESSION_SUFFIX_RE = re.compile(r"^(.*?)(?:_(0*1|0*2))$")


def strip_session_suffix(participant_id: str) -> tuple[str, int | None]:
    match = SESSION_SUFFIX_RE.match(participant_id.strip())
    if not match:
        return participant_id.strip(), None
    base_id, session_str = match.groups()
    return base_id, int(session_str)


def select_control_rows_include_second_only(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unsuffixed_rows: list[dict[str, str]] = []
    suffixed_rows: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        participant_id = row.get("subID", "").strip()
        base_id, session = strip_session_suffix(participant_id)
        if session is None:
            unsuffixed_rows.append(row)
        else:
            suffixed_rows[base_id][session] = row

    selected = list(unsuffixed_rows)
    second_only: list[dict[str, str]] = []
    for base_id, session_rows in sorted(suffixed_rows.items()):
        if 1 in session_rows:
            selected.append(session_rows[1])
        elif 2 in session_rows:
            selected.append(session_rows[2])
            second_only.append(session_rows[2])

    if second_only:
        pd.DataFrame(second_only).to_csv(
            OUTPUT_DIR / "paper_second_session_only_controls.csv",
            index=False,
        )
    return sorted(selected, key=lambda row: row["study_result_id"])


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def safe_nanmean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def route_summary(routes: list[np.ndarray]) -> dict[str, float]:
    if not routes:
        return {
            "route_count": 0,
            "valid_route_count": 0,
            "session_duration_ms": math.nan,
            "total_draw_time_ms": math.nan,
            "mean_path_duration_ms": math.nan,
            "mean_interpath_interval_ms": math.nan,
            "total_distance_px": math.nan,
            "mean_path_length_px": math.nan,
            "mean_speed_px_per_s": math.nan,
        }
    starts = np.array([route[0, 0] for route in routes], dtype=float)
    ends = np.array([route[-1, 0] for route in routes], dtype=float)
    durations = ends - starts
    interpath = starts[1:] - ends[:-1] if len(routes) > 1 else np.array([], dtype=float)
    path_lengths = []
    for route in routes:
        if route.shape[0] < 2:
            path_lengths.append(0.0)
            continue
        xy = route[:, 1:3]
        path_lengths.append(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    total_distance = float(np.sum(path_lengths))
    total_draw_time = float(np.sum(durations))
    return {
        "route_count": int(len(routes)),
        "valid_route_count": int(sum(route.shape[0] > 1 for route in routes)),
        "session_duration_ms": float(np.max(ends) - np.min(starts)),
        "total_draw_time_ms": total_draw_time,
        "mean_path_duration_ms": safe_nanmean(durations),
        "mean_interpath_interval_ms": safe_nanmean(interpath),
        "total_distance_px": total_distance,
        "mean_path_length_px": safe_nanmean(path_lengths),
        "mean_speed_px_per_s": total_distance / (total_draw_time / 1000.0) if total_draw_time > 0 else math.nan,
    }


def empty_motor_fingerprint() -> dict[str, float]:
    return {
        "control_path_count": 0,
        "control_mean_speed_px_per_s": math.nan,
        "control_peak_speed_px_per_s": math.nan,
        "control_mean_straightness": math.nan,
        "control_tremor_rms_px_per_s": math.nan,
        "control_jerk_rms": math.nan,
        "control_endpoint_scatter_px": math.nan,
        "control_path_duration_cv": math.nan,
    }


def compute_motor_fingerprint(participant, smooth_window: int = SMOOTH_WINDOW) -> dict[str, float]:
    routes = participant.practice_routes or []
    if not routes:
        return empty_motor_fingerprint()

    mean_speeds: list[float] = []
    peak_speeds: list[float] = []
    straightness_values: list[float] = []
    tremor_values: list[float] = []
    jerk_values: list[float] = []
    durations_ms: list[float] = []
    start_points: list[tuple[float, float]] = []
    end_points: list[tuple[float, float]] = []

    for route in routes:
        if route.shape[0] < 4:
            continue
        time_ms = route[:, 0]
        x = route[:, 1]
        y = route[:, 2]
        dt_s = np.diff(time_ms) / 1000.0
        dx = np.diff(x)
        dy = np.diff(y)
        step_len = np.hypot(dx, dy)
        valid = dt_s > 0
        if not np.any(valid):
            continue
        speed = np.zeros_like(step_len)
        speed[valid] = step_len[valid] / dt_s[valid]
        path_length = float(np.sum(step_len))
        duration_s = (time_ms[-1] - time_ms[0]) / 1000.0
        if path_length <= 0 or duration_s <= 0:
            continue
        chord = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        smoothed_speed = moving_average(speed, smooth_window)
        residual = speed - smoothed_speed
        vx = np.zeros_like(dx)
        vy = np.zeros_like(dy)
        vx[valid] = dx[valid] / dt_s[valid]
        vy[valid] = dy[valid] / dt_s[valid]
        jerk_rms = math.nan
        if vx.size >= 3:
            ax = np.diff(vx)
            ay = np.diff(vy)
            if ax.size >= 2:
                jx = np.diff(ax)
                jy = np.diff(ay)
                jerk_rms = float(np.sqrt(np.mean(jx**2 + jy**2)))
        mean_speeds.append(path_length / duration_s)
        peak_speeds.append(float(np.max(speed)))
        straightness_values.append(chord / path_length)
        tremor_values.append(float(np.sqrt(np.mean(residual**2))))
        jerk_values.append(jerk_rms)
        durations_ms.append(duration_s * 1000.0)
        start_points.append((float(x[0]), float(y[0])))
        end_points.append((float(x[-1]), float(y[-1])))

    if not mean_speeds:
        return empty_motor_fingerprint()

    start_arr = np.asarray(start_points, dtype=float)
    end_arr = np.asarray(end_points, dtype=float)
    start_scatter = float(np.hypot(np.std(start_arr[:, 0]), np.std(start_arr[:, 1])))
    end_scatter = float(np.hypot(np.std(end_arr[:, 0]), np.std(end_arr[:, 1])))
    duration_mean = float(np.mean(durations_ms))
    return {
        "control_path_count": int(len(mean_speeds)),
        "control_mean_speed_px_per_s": float(np.mean(mean_speeds)),
        "control_peak_speed_px_per_s": float(np.max(peak_speeds)),
        "control_mean_straightness": float(np.mean(straightness_values)),
        "control_tremor_rms_px_per_s": float(np.mean(tremor_values)),
        "control_jerk_rms": float(safe_nanmean(jerk_values)),
        "control_endpoint_scatter_px": 0.5 * (start_scatter + end_scatter),
        "control_path_duration_cv": float(np.std(durations_ms) / duration_mean) if duration_mean > 0 else math.nan,
    }


def resample_route(route: np.ndarray, np_points: int = NP_POINTS) -> np.ndarray | None:
    z = route[:, 1] + 1j * route[:, 2]
    if z.size < 2:
        return None
    s = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(z)))])
    unique_mask = np.concatenate([[True], np.diff(s) > 0])
    s = s[unique_mask]
    z = z[unique_mask]
    if s.size < 2 or s[-1] <= 0:
        return None
    grid = np.linspace(0.0, 1.0, np_points)
    normalized_s = s / s[-1]
    real_interp = np.interp(grid, normalized_s, z.real)
    imag_interp = np.interp(grid, normalized_s, z.imag)
    return real_interp + 1j * imag_interp


def route_feature_pair(route: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    z = resample_route(route)
    if z is None:
        return None
    real_part = np.real(z)
    imag_part = np.imag(z)
    vel_real = moving_average(np.diff(real_part), SMOOTH_WINDOW)
    vel_imag = moving_average(np.diff(imag_part), SMOOTH_WINDOW)
    acc_real = moving_average(np.diff(vel_real), SMOOTH_WINDOW)
    acc_imag = moving_average(np.diff(vel_imag), SMOOTH_WINDOW)
    vel_real = vel_real[::VELOCITY_DOWNSAMPLE]
    vel_imag = vel_imag[::VELOCITY_DOWNSAMPLE]
    acc_real = acc_real[::VELOCITY_DOWNSAMPLE]
    acc_imag = acc_imag[::VELOCITY_DOWNSAMPLE]
    feature = np.concatenate([
        real_part - X_CENTER,
        imag_part - Y_CENTER,
        vel_real,
        vel_imag,
        acc_real,
        acc_imag,
    ]).astype(np.float32)
    flippable_mask = np.concatenate([
        np.ones(real_part.size, dtype=bool),
        np.zeros(imag_part.size, dtype=bool),
        np.ones(vel_real.size, dtype=bool),
        np.zeros(vel_imag.size, dtype=bool),
        np.ones(acc_real.size, dtype=bool),
        np.zeros(acc_imag.size, dtype=bool),
    ])
    flipped = feature.copy()
    flipped[flippable_mask] *= -1
    return feature, flipped.astype(np.float32)


def participant_key(participant) -> str:
    return f"{participant.group}||{participant.participant_id}||{participant.source_id}"


def compute_route_uniqueness(participants: list) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    records: list[dict] = []
    features: list[np.ndarray] = []
    flipped_features: list[np.ndarray] = []
    for p_index, participant in enumerate(participants):
        p_key = participant_key(participant)
        for route_index, route in enumerate(participant.test_routes):
            pair = route_feature_pair(route)
            if pair is None:
                continue
            records.append({
                "route_global_index": len(records),
                "participant_key": p_key,
                "group": participant.group,
                "participant_id": participant.participant_id,
                "source_id": participant.source_id,
                "route_index": route_index,
            })
            features.append(pair[0])
            flipped_features.append(pair[1])

    if not features:
        return {}, pd.DataFrame()

    X = np.vstack(features).astype(np.float32)
    XF = np.vstack(flipped_features).astype(np.float32)
    x_norm = np.einsum("ij,ij->i", X, X).astype(np.float32)
    n_routes = X.shape[0]
    nearest = np.full(n_routes, np.inf, dtype=np.float32)

    print(f"Computing nearest-route uniqueness for {n_routes} routes...")
    for start in range(0, n_routes, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n_routes)
        for query in (X[start:end], XF[start:end]):
            q_norm = np.einsum("ij,ij->i", query, query).astype(np.float32)
            dist2 = q_norm[:, None] + x_norm[None, :] - 2.0 * (query @ X.T)
            np.maximum(dist2, 0, out=dist2)
            for local_idx, global_idx in enumerate(range(start, end)):
                dist2[local_idx, global_idx] = np.inf
            block_min = np.sqrt(np.min(dist2, axis=1))
            nearest[start:end] = np.minimum(nearest[start:end], block_min)
        if start == 0 or end == n_routes or (start // BLOCK_SIZE) % 20 == 0:
            print(f"  processed routes {end}/{n_routes}")

    route_df = pd.DataFrame(records)
    route_df["route_uniqueness"] = nearest.astype(float)

    summary: dict[str, dict[str, float]] = {}
    for p_key, group in route_df.groupby("participant_key"):
        vals = group["route_uniqueness"].to_numpy(dtype=float)
        positive = vals[vals > 0]
        summary[p_key] = {
            "test_uniqueness_mean": float(np.nanmean(vals)) if vals.size else math.nan,
            "test_log_uniqueness_mean": float(np.nanmean(np.log(positive))) if positive.size else math.nan,
            "test_uniqueness_route_count": int(vals.size),
        }
    return summary, route_df


def main() -> None:
    ns = load_analyze_namespace()
    ns["select_control_rows"] = select_control_rows_include_second_only

    controls = ns["load_controls"](CONTROL_CSV, CONTROL_DIR)
    patients = ns["load_patients"](PATIENT_OCTAL_DIR, PATIENT_JATOS_DIRS)
    participants = list(controls) + list(patients)
    print(f"Loaded {len(controls)} controls and {len(patients)} patients.")
    attempt_log = ns.get("EXTRA_CONTROL_ATTEMPT_LOG", [])
    if attempt_log:
        pd.DataFrame(attempt_log).to_csv(OUTPUT_DIR / "extra_control_attempts.csv", index=False)

    uniqueness, route_df = compute_route_uniqueness(participants)
    route_df.to_csv(OUTPUT_DIR / "paper_route_uniqueness.csv", index=False)

    rows: list[dict] = []
    calibration_rows: list[dict] = []
    for participant in participants:
        p_key = participant_key(participant)
        row = {
            "group": participant.group,
            "participant_id": participant.participant_id,
            "source_id": participant.source_id,
            "source_platform": participant.source_platform,
            "raw_source": participant.metadata.get("raw_source", ""),
        }
        row.update(compute_motor_fingerprint(participant))
        for condition_name, routes in (("practice", participant.practice_routes), ("test", participant.test_routes)):
            for key, value in route_summary(routes).items():
                row[f"{condition_name}_{key}"] = value
        row.update(uniqueness.get(p_key, {
            "test_uniqueness_mean": math.nan,
            "test_log_uniqueness_mean": math.nan,
            "test_uniqueness_route_count": 0,
        }))
        if participant.group == "control":
            for metadata_key in ("study_result_id", "component_result_id", "age", "sex", "hand", "timestamp_utc", "timestamp_local"):
                row[metadata_key] = participant.metadata.get(metadata_key, "")
        rows.append(row)

        calibration_rows.append({
            "group": participant.group,
            "participant_id": participant.participant_id,
            "source_id": participant.source_id,
            "source_platform": participant.source_platform,
            "coordinate_transform": participant.metadata.get("coordinate_transform", ""),
            "coordinate_calibration_method": participant.metadata.get("coordinate_calibration_method", ""),
            "calibration_practice_routes": participant.metadata.get("calibration_practice_routes", ""),
            "calibration_x_center_raw": participant.metadata.get("calibration_x_center_raw", ""),
            "calibration_y_top_raw": participant.metadata.get("calibration_y_top_raw", ""),
            "calibration_y_bottom_raw": participant.metadata.get("calibration_y_bottom_raw", ""),
            "calibration_x_scale": participant.metadata.get("calibration_x_scale", ""),
            "calibration_y_scale": participant.metadata.get("calibration_y_scale", ""),
        })

    participant_df = pd.DataFrame(rows)
    participant_df.to_csv(OUTPUT_DIR / "paper_phaseA_participant_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUTPUT_DIR / "paper_coordinate_calibration.csv", index=False)

    audit = pd.DataFrame([
        {"item": "controls", "n": int((participant_df["group"] == "control").sum())},
        {"item": "patients", "n": int((participant_df["group"] == "patient").sum())},
        {"item": "controls_finite_uniqueness", "n": int(((participant_df["group"] == "control") & np.isfinite(participant_df["test_uniqueness_mean"])).sum())},
        {"item": "patients_finite_uniqueness", "n": int(((participant_df["group"] == "patient") & np.isfinite(participant_df["test_uniqueness_mean"])).sum())},
        {"item": "route_count", "n": int(len(route_df))},
    ])
    audit.to_csv(OUTPUT_DIR / "paper_route_uniqueness_audit.csv", index=False)
    print(audit.to_string(index=False))
    print(f"Wrote shared paper metrics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
