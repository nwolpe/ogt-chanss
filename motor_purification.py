#!/usr/bin/env python3
"""Motor-purified uniqueness and motor/cognitive dissociation analysis.

This is Phase A of the two-part OGT paper plan. It lives alongside
``analyze_ogt_groups.py`` and reuses its loaders/helpers so that raw data
does not have to be re-parsed or re-wrangled here.

What this module adds on top of the existing pipeline:

1. A richer control-task *motor fingerprint* per participant (mean and peak
   speed, path straightness, tremor RMS, jerk RMS, endpoint scatter, duration
   variability).

2. A *feature-level* purified uniqueness: Ang-style uniqueness computed from
   position-only feature vectors (drops the velocity and acceleration blocks
   that are used in the original distance metric), so the distance between
   paths reflects only their shape/geometry.

3. A *subject-level* purified uniqueness: OLS residuals of the raw Ang
   uniqueness after regressing out motor-fingerprint covariates. Fitted
   within the patient group to avoid group-mean confounding.

4. A *dissociation table*: for each pairing of a motor fingerprint metric, a
   purified uniqueness metric, and a clinical scale, we compute Spearman
   correlations on both sides, the shared correlation between motor and
   cognitive, and the Hotelling-Williams test for whether the two dependent
   correlations differ significantly.

5. A compact forest-plot figure visualising the dissociation for a flagship
   motor metric and the subject-level residualised uniqueness.

Outputs land in ``analysis_outputs/phaseA/`` by default.

Run with ``python motor_purification.py --help`` for options.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist, pdist, squareform  # noqa: E402
from scipy.stats import spearmanr, t as t_dist  # noqa: E402

# Reuse everything already built in the existing pipeline.
from analyze_ogt_groups import (  # noqa: E402
    CLINICAL_CORRELATION_FIELDS,
    ParticipantRoutes,
    X_CENTER,
    Y_CENTER,
    build_clinical_match_rows,
    compute_uniqueness_metrics,
    load_clinical_id_corrections,
    load_clinical_rows,
    load_controls,
    load_patients,
    moving_average,
    resample_route,
    safe_nanmean,
    to_float,
    write_csv,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MOTOR_COLUMNS: list[str] = [
    "control_mean_speed_px_per_s",
    "control_peak_speed_px_per_s",
    "control_mean_straightness",
    "control_tremor_rms_px_per_s",
    "control_jerk_rms",
    "control_endpoint_scatter_px",
    "control_path_duration_cv",
]

DEFAULT_MOTOR_RESIDUALISER_COLUMNS: list[str] = [
    "control_mean_speed_px_per_s",
    "control_tremor_rms_px_per_s",
    "control_mean_straightness",
]

DEFAULT_COGNITIVE_COLUMNS: list[str] = [
    "test_uniqueness_mean",
    "test_position_uniqueness_mean",
    "test_uniqueness_residualised",
]

DEFAULT_DISSOCIATION_MOTOR_SUBSET: list[str] = [
    "control_mean_speed_px_per_s",
    "control_peak_speed_px_per_s",
    "control_tremor_rms_px_per_s",
    "control_jerk_rms",
]


# ---------------------------------------------------------------------------
# Motor fingerprint (from control / practice routes)
# ---------------------------------------------------------------------------


def _empty_motor_fingerprint() -> dict[str, float]:
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


def compute_motor_fingerprint(
    participant: ParticipantRoutes,
    smooth_window: int = 40,
) -> dict[str, float]:
    """Summarise motor behaviour from the 30 s control / practice block.

    Returns a dict of per-participant motor descriptors. All values are
    NaN if no usable practice routes are present.
    """
    routes = participant.practice_routes or []
    if not routes:
        return _empty_motor_fingerprint()

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
        straightness = chord / path_length

        mean_speed = path_length / duration_s

        # Tremor: high-frequency residual between raw speed and its smoothed
        # counterpart. The smoothing window mirrors the one used in the Ang
        # feature vector so the motor fingerprint captures what the original
        # metric is already implicitly using.
        smoothed_speed = moving_average(speed, smooth_window)
        residual = speed - smoothed_speed
        tremor_rms = float(np.sqrt(np.mean(residual ** 2)))

        # Jerk: RMS magnitude of the third derivative of position,
        # approximated with successive differences of velocity components.
        vx = np.zeros_like(dx)
        vy = np.zeros_like(dy)
        vx[valid] = dx[valid] / dt_s[valid]
        vy[valid] = dy[valid] / dt_s[valid]
        if vx.size >= 3:
            ax = np.diff(vx)
            ay = np.diff(vy)
            if ax.size >= 2:
                jx = np.diff(ax)
                jy = np.diff(ay)
                jerk_rms = float(np.sqrt(np.mean(jx ** 2 + jy ** 2)))
            else:
                jerk_rms = math.nan
        else:
            jerk_rms = math.nan

        mean_speeds.append(mean_speed)
        peak_speeds.append(float(np.max(speed)))
        straightness_values.append(straightness)
        tremor_values.append(tremor_rms)
        jerk_values.append(jerk_rms)
        durations_ms.append(duration_s * 1000.0)
        start_points.append((float(x[0]), float(y[0])))
        end_points.append((float(x[-1]), float(y[-1])))

    if not mean_speeds:
        return _empty_motor_fingerprint()

    start_arr = np.asarray(start_points, dtype=float)
    end_arr = np.asarray(end_points, dtype=float)
    start_scatter = float(np.hypot(np.std(start_arr[:, 0]), np.std(start_arr[:, 1])))
    end_scatter = float(np.hypot(np.std(end_arr[:, 0]), np.std(end_arr[:, 1])))
    endpoint_scatter = 0.5 * (start_scatter + end_scatter)

    duration_mean = float(np.mean(durations_ms))
    duration_cv = (
        float(np.std(durations_ms) / duration_mean) if duration_mean > 0 else math.nan
    )

    return {
        "control_path_count": int(len(mean_speeds)),
        "control_mean_speed_px_per_s": float(np.mean(mean_speeds)),
        "control_peak_speed_px_per_s": float(np.max(peak_speeds)),
        "control_mean_straightness": float(np.mean(straightness_values)),
        "control_tremor_rms_px_per_s": float(np.mean(tremor_values)),
        "control_jerk_rms": float(safe_nanmean(jerk_values)),
        "control_endpoint_scatter_px": endpoint_scatter,
        "control_path_duration_cv": duration_cv,
    }


# ---------------------------------------------------------------------------
# Feature-level purification: position-only uniqueness
# ---------------------------------------------------------------------------


def _position_feature_pair(route: np.ndarray, np_points: int) -> np.ndarray | None:
    """Return a 2xF matrix: original feature vector and its mirror-flipped twin."""
    z = resample_route(route, np_points=np_points)
    if z is None:
        return None
    real_part = np.real(z) - X_CENTER
    imag_part = np.imag(z) - Y_CENTER
    feat = np.concatenate([real_part, imag_part])
    # Mirror about the vertical midline: negate x only.
    feat_flipped = np.concatenate([-real_part, imag_part])
    return np.vstack([feat, feat_flipped])


def compute_position_only_uniqueness_metrics(
    participants: Sequence[ParticipantRoutes],
    np_points: int = 200,
) -> dict[str, dict[str, float]]:
    """Ang-style uniqueness using *only* position features (drops velocity/accel)."""

    route_records: list[dict[str, object]] = []
    for participant in participants:
        for route_index, route in enumerate(participant.test_routes):
            pair = _position_feature_pair(route, np_points=np_points)
            if pair is None:
                continue
            route_records.append(
                {
                    "participant_id": participant.participant_id,
                    "route_index": route_index,
                    "feature": pair[0],
                    "feature_flipped": pair[1],
                }
            )

    if not route_records:
        return {}

    features = np.vstack([record["feature"] for record in route_records])
    features_flipped = np.vstack([record["feature_flipped"] for record in route_records])
    distance_matrix = squareform(pdist(features, metric="euclidean"))
    flip_distance_matrix = cdist(features_flipped, features, metric="euclidean")

    by_participant: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(route_records):
        by_participant[str(record["participant_id"])].append(idx)

    results: dict[str, dict[str, float]] = {}
    for participant in participants:
        indices = by_participant.get(participant.participant_id, [])
        if not indices:
            results[participant.participant_id] = {
                "test_position_uniqueness_mean": math.nan,
                "test_log_position_uniqueness_mean": math.nan,
                "test_position_uniqueness_route_count": 0,
            }
            continue

        uniqueness_values: list[float] = []
        for idx in indices:
            d_orig = distance_matrix[:, idx].astype(float).copy()
            d_flip = flip_distance_matrix[:, idx].astype(float).copy()
            d_orig[idx] = np.nan
            d_flip[idx] = np.nan
            combined = np.concatenate([d_orig, d_flip])
            uniqueness_values.append(float(np.nanmin(combined)))

        positive = [v for v in uniqueness_values if v > 0]
        results[participant.participant_id] = {
            "test_position_uniqueness_mean": safe_nanmean(uniqueness_values),
            "test_log_position_uniqueness_mean": (
                float(np.mean(np.log(positive))) if positive else math.nan
            ),
            "test_position_uniqueness_route_count": int(len(uniqueness_values)),
        }

    return results


# ---------------------------------------------------------------------------
# Subject-level purification: OLS residualisation
# ---------------------------------------------------------------------------


def residualise_column(
    rows: list[dict[str, object]],
    target: str,
    predictors: Sequence[str],
    group_filter: str | None = "patient",
) -> tuple[list[float], np.ndarray | None, int]:
    """Regress ``target`` on ``predictors`` within the specified group.

    Residuals are returned for every input row in the same order. Rows that
    are not in ``group_filter`` (or that have missing values) receive NaN.

    Also returns the fitted coefficient vector (``None`` if the fit failed)
    and the number of observations used in the fit, so the caller can log
    the model.
    """

    y_fit: list[float] = []
    x_fit: list[list[float]] = []
    idx_used: list[int] = []

    for i, row in enumerate(rows):
        if group_filter is not None and row.get("group") != group_filter:
            continue
        y_val = to_float(row.get(target, math.nan))
        if math.isnan(y_val):
            continue
        x_vals = [to_float(row.get(p, math.nan)) for p in predictors]
        if any(math.isnan(x) for x in x_vals):
            continue
        y_fit.append(y_val)
        x_fit.append(x_vals)
        idx_used.append(i)

    residuals: list[float] = [math.nan] * len(rows)
    n = len(idx_used)

    # Need at least a few more observations than predictors for a stable fit.
    if n < len(predictors) + 3:
        return residuals, None, n

    X_arr = np.asarray(x_fit, dtype=float)
    y_arr = np.asarray(y_fit, dtype=float)
    X_with_const = np.column_stack([np.ones(n), X_arr])
    beta, *_ = np.linalg.lstsq(X_with_const, y_arr, rcond=None)
    y_hat = X_with_const @ beta
    resid = y_arr - y_hat
    for k, i in enumerate(idx_used):
        residuals[i] = float(resid[k])

    return residuals, beta, n


# ---------------------------------------------------------------------------
# Hotelling-Williams test for dependent correlations
# ---------------------------------------------------------------------------


def williams_test(r12: float, r13: float, r23: float, n: int) -> tuple[float, float, float]:
    """Hotelling-Williams t-test for H0: r12 == r13.

    ``r12`` and ``r13`` are two dependent correlations with a *shared*
    variable (variable 1); ``r23`` is the correlation between the two
    non-shared variables. Returns (t, df, p) with a two-tailed p-value.

    Reference: Steiger, J.H. (1980), Psychological Bulletin 87(2), 245-251.
    """
    if any(math.isnan(r) for r in (r12, r13, r23)) or n < 5:
        return math.nan, math.nan, math.nan

    R_det = 1.0 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2.0 * r12 * r13 * r23
    if R_det <= 0:
        return math.nan, math.nan, math.nan

    avg_r = 0.5 * (r12 + r13)
    numerator = (r12 - r13) * math.sqrt((n - 1) * (1.0 + r23))
    denom_inner = 2.0 * ((n - 1) / (n - 3)) * R_det + (avg_r ** 2) * (1.0 - r23) ** 3
    if denom_inner <= 0:
        return math.nan, math.nan, math.nan

    t_stat = numerator / math.sqrt(denom_inner)
    df = n - 3
    p_val = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df))
    return float(t_stat), float(df), float(p_val)


# ---------------------------------------------------------------------------
# Dissociation table
# ---------------------------------------------------------------------------


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    n = int(np.count_nonzero(mask))
    if n < 4:
        return math.nan, math.nan, n
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def build_dissociation_rows(
    matched_rows: list[dict[str, object]],
    motor_columns: Sequence[str],
    cognitive_columns: Sequence[str],
    clinical_columns: Sequence[str],
) -> list[dict[str, object]]:
    """Long-format dissociation table.

    One row per (motor_metric, cognitive_metric, clinical_metric) combination.
    Each row reports the two 'side' correlations (motor↔clinical and
    cognitive↔clinical), the shared motor↔cognitive correlation, and the
    Williams test for the difference of the first two.
    """

    def _col(name: str) -> np.ndarray:
        return np.asarray(
            [to_float(row.get(name, math.nan)) for row in matched_rows],
            dtype=float,
        )

    rows: list[dict[str, object]] = []
    for clinical in clinical_columns:
        clinical_vals = _col(clinical)
        for motor in motor_columns:
            motor_vals = _col(motor)
            for cognitive in cognitive_columns:
                cog_vals = _col(cognitive)

                mask = (
                    (~np.isnan(clinical_vals))
                    & (~np.isnan(motor_vals))
                    & (~np.isnan(cog_vals))
                )
                n = int(np.count_nonzero(mask))

                if n < 4:
                    rho_mc = p_mc = rho_uc = p_uc = rho_mu = p_mu = math.nan
                    t_w = df_w = p_w = math.nan
                else:
                    rho_mc, p_mc = spearmanr(motor_vals[mask], clinical_vals[mask])
                    rho_uc, p_uc = spearmanr(cog_vals[mask], clinical_vals[mask])
                    rho_mu, p_mu = spearmanr(motor_vals[mask], cog_vals[mask])
                    t_w, df_w, p_w = williams_test(
                        float(rho_mc), float(rho_uc), float(rho_mu), n
                    )

                rows.append(
                    {
                        "clinical_metric": clinical,
                        "motor_metric": motor,
                        "cognitive_metric": cognitive,
                        "n": n,
                        "motor_vs_clinical_rho": float(rho_mc)
                        if not (isinstance(rho_mc, float) and math.isnan(rho_mc))
                        else math.nan,
                        "motor_vs_clinical_p": float(p_mc)
                        if not (isinstance(p_mc, float) and math.isnan(p_mc))
                        else math.nan,
                        "cognitive_vs_clinical_rho": float(rho_uc)
                        if not (isinstance(rho_uc, float) and math.isnan(rho_uc))
                        else math.nan,
                        "cognitive_vs_clinical_p": float(p_uc)
                        if not (isinstance(p_uc, float) and math.isnan(p_uc))
                        else math.nan,
                        "motor_vs_cognitive_rho": float(rho_mu)
                        if not (isinstance(rho_mu, float) and math.isnan(rho_mu))
                        else math.nan,
                        "motor_vs_cognitive_p": float(p_mu)
                        if not (isinstance(p_mu, float) and math.isnan(p_mu))
                        else math.nan,
                        "williams_t": t_w,
                        "williams_df": df_w,
                        "williams_p": p_w,
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Forest-plot visualisation
# ---------------------------------------------------------------------------


def save_dissociation_forest_plot(
    path: Path,
    dissociation_rows: list[dict[str, object]],
    motor_metric: str,
    cognitive_metric: str,
    clinical_columns: Sequence[str],
) -> None:
    """Plot motor vs cognitive correlations with a shared clinical-measure axis."""

    relevant = [
        row
        for row in dissociation_rows
        if row["motor_metric"] == motor_metric and row["cognitive_metric"] == cognitive_metric
    ]
    if not relevant:
        return

    by_clin = {row["clinical_metric"]: row for row in relevant}
    ordered = [m for m in clinical_columns if m in by_clin]
    if not ordered:
        return

    y_positions = np.arange(len(ordered))
    motor_rhos = np.asarray(
        [to_float(by_clin[m].get("motor_vs_clinical_rho", math.nan)) for m in ordered],
        dtype=float,
    )
    cog_rhos = np.asarray(
        [to_float(by_clin[m].get("cognitive_vs_clinical_rho", math.nan)) for m in ordered],
        dtype=float,
    )
    n_values = [int(by_clin[m].get("n", 0)) for m in ordered]
    williams_p = [to_float(by_clin[m].get("williams_p", math.nan)) for m in ordered]

    # 95% CI for Spearman rho via Fisher z approximation.
    def _ci(rho: float, n: int) -> tuple[float, float]:
        if math.isnan(rho) or n < 5:
            return math.nan, math.nan
        z = np.arctanh(rho)
        se = 1.0 / math.sqrt(n - 3)
        return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))

    motor_ci = [_ci(float(r), n) for r, n in zip(motor_rhos, n_values)]
    cog_ci = [_ci(float(r), n) for r, n in zip(cog_rhos, n_values)]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(ordered) + 2.5))

    offset = 0.18
    motor_color = "#5A7D9A"
    cognitive_color = "#C26D52"

    for i, (rho, (lo, hi)) in enumerate(zip(motor_rhos, motor_ci)):
        if not math.isnan(rho):
            ax.plot([lo, hi], [i - offset, i - offset], color=motor_color, lw=1.5)
            ax.plot(rho, i - offset, marker="o", color=motor_color, markersize=7)
    for i, (rho, (lo, hi)) in enumerate(zip(cog_rhos, cog_ci)):
        if not math.isnan(rho):
            ax.plot([lo, hi], [i + offset, i + offset], color=cognitive_color, lw=1.5)
            ax.plot(rho, i + offset, marker="s", color=cognitive_color, markersize=7)

    for i, p in enumerate(williams_p):
        if math.isnan(p):
            continue
        marker = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        if marker:
            ax.text(
                1.02,
                i,
                marker,
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=12,
            )

    ax.axvline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered)
    ax.set_xlabel("Spearman rho with clinical measure")
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    ax.set_title(
        f"Motor ({motor_metric}) vs cognitive ({cognitive_metric})\n"
        "Williams test: * p<0.05, ** p<0.01"
    )

    motor_patch = plt.Line2D([0], [0], color=motor_color, marker="o", lw=1.5, label="Motor")
    cog_patch = plt.Line2D([0], [0], color=cognitive_color, marker="s", lw=1.5, label="Cognitive")
    ax.legend(handles=[motor_patch, cog_patch], loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Participant row assembly
# ---------------------------------------------------------------------------


def build_phaseA_participant_rows(
    participants: Sequence[ParticipantRoutes],
    motor_fingerprints: dict[str, dict[str, float]],
    uniqueness_metrics: dict[str, dict[str, float]],
    position_uniqueness_metrics: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant in participants:
        row: dict[str, object] = {
            "group": participant.group,
            "participant_id": participant.participant_id,
            "source_id": participant.source_id,
            "source_platform": participant.source_platform,
            "raw_source": participant.metadata.get("raw_source", ""),
        }
        row.update(motor_fingerprints.get(participant.participant_id, _empty_motor_fingerprint()))
        row.update(
            uniqueness_metrics.get(
                participant.participant_id,
                {
                    "test_uniqueness_mean": math.nan,
                    "test_log_uniqueness_mean": math.nan,
                    "test_mds_hull_area": math.nan,
                    "test_uniqueness_route_count": 0,
                },
            )
        )
        row.update(
            position_uniqueness_metrics.get(
                participant.participant_id,
                {
                    "test_position_uniqueness_mean": math.nan,
                    "test_log_position_uniqueness_mean": math.nan,
                    "test_position_uniqueness_route_count": 0,
                },
            )
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-csv",
        type=Path,
        default=Path("control_demographics_sorted.csv"),
    )
    parser.add_argument("--control-dir", type=Path, default=Path("data_controls"))
    parser.add_argument(
        "--patient-octal-dir",
        type=Path,
        default=Path("data_patients/OCTAL"),
    )
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
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=Path("CHANSS_clinical_demog.csv"),
    )
    parser.add_argument(
        "--clinical-id-corrections-csv",
        type=Path,
        default=Path("analysis_outputs/ogt_patients_missing_clinical_data_corrected.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/phaseA"),
    )
    parser.add_argument("--np-points", type=int, default=200)
    parser.add_argument("--smooth-window", type=int, default=40)
    parser.add_argument("--velocity-downsample", type=int, default=10)
    parser.add_argument(
        "--residualiser-columns",
        nargs="+",
        default=DEFAULT_MOTOR_RESIDUALISER_COLUMNS,
        help=(
            "Motor fingerprint columns used as predictors when producing the "
            "subject-level residualised uniqueness."
        ),
    )
    parser.add_argument(
        "--residualiser-group",
        default="patient",
        choices=["patient", "control", "pooled"],
        help="Group within which the OLS fit is performed.",
    )
    parser.add_argument(
        "--forest-motor-metric",
        default="control_mean_speed_px_per_s",
        help="Motor metric to highlight in the forest-plot figure.",
    )
    parser.add_argument(
        "--forest-cognitive-metric",
        default="test_uniqueness_residualised",
        help="Cognitive metric to highlight in the forest-plot figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load participants using the existing loaders.
    controls = load_controls(args.control_csv, args.control_dir)
    patients = load_patients(args.patient_octal_dir, args.patient_jatos_dirs)
    participants: list[ParticipantRoutes] = list(controls) + list(patients)
    print(f"Loaded {len(controls)} controls and {len(patients)} patients.")

    # 2. Motor fingerprint from control/practice routes.
    motor_fingerprints: dict[str, dict[str, float]] = {}
    for participant in participants:
        motor_fingerprints[participant.participant_id] = compute_motor_fingerprint(
            participant,
            smooth_window=args.smooth_window,
        )

    # 3. Raw Ang-style uniqueness (reuses existing implementation).
    uniqueness_metrics = compute_uniqueness_metrics(
        participants,
        np_points=args.np_points,
        smooth_window=args.smooth_window,
        velocity_downsample=args.velocity_downsample,
    )

    # 4. Feature-level purified uniqueness (position only).
    position_uniqueness_metrics = compute_position_only_uniqueness_metrics(
        participants,
        np_points=args.np_points,
    )

    # 5. Assemble participant rows with raw + position-only uniqueness + motor.
    participant_rows = build_phaseA_participant_rows(
        participants,
        motor_fingerprints,
        uniqueness_metrics,
        position_uniqueness_metrics,
    )

    # 6. Subject-level residualisation of raw uniqueness on motor predictors.
    group_filter = None if args.residualiser_group == "pooled" else args.residualiser_group
    residuals, beta, n_fit = residualise_column(
        participant_rows,
        target="test_uniqueness_mean",
        predictors=args.residualiser_columns,
        group_filter=group_filter,
    )
    for row, r in zip(participant_rows, residuals):
        row["test_uniqueness_residualised"] = r

    # Record the fitted model for transparency.
    model_rows: list[dict[str, object]] = []
    if beta is not None:
        coef_names = ["intercept", *args.residualiser_columns]
        for name, value in zip(coef_names, beta):
            model_rows.append({"term": name, "coefficient": float(value)})
        model_rows.append({"term": "_n_used", "coefficient": float(n_fit)})
        model_rows.append({"term": "_group", "coefficient": args.residualiser_group})
    else:
        model_rows.append({"term": "_status", "coefficient": "insufficient_data"})
        model_rows.append({"term": "_n_used", "coefficient": float(n_fit)})

    # 7. Match to clinical data and build the dissociation table.
    clinical_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    if args.clinical_csv.exists():
        clinical_rows = load_clinical_rows(args.clinical_csv)
        id_corrections = load_clinical_id_corrections(args.clinical_id_corrections_csv)
        matched_rows, missing_rows = build_clinical_match_rows(
            participant_rows,
            clinical_rows,
            id_corrections,
        )

    dissociation_rows: list[dict[str, object]] = []
    if matched_rows:
        dissociation_rows = build_dissociation_rows(
            matched_rows,
            motor_columns=DEFAULT_DISSOCIATION_MOTOR_SUBSET,
            cognitive_columns=DEFAULT_COGNITIVE_COLUMNS,
            clinical_columns=CLINICAL_CORRELATION_FIELDS,
        )

    # 8. Write outputs.
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_fieldnames = list(participant_rows[0].keys()) if participant_rows else []
    write_csv(
        output_dir / "phaseA_participant_metrics.csv",
        participant_rows,
        participant_fieldnames,
    )

    motor_rows = [
        {"participant_id": pid, **fp} for pid, fp in motor_fingerprints.items()
    ]
    motor_fieldnames = ["participant_id"] + list(_empty_motor_fingerprint().keys())
    write_csv(output_dir / "phaseA_motor_fingerprint.csv", motor_rows, motor_fieldnames)

    write_csv(
        output_dir / "phaseA_residualiser_model.csv",
        model_rows,
        ["term", "coefficient"],
    )

    if dissociation_rows:
        write_csv(
            output_dir / "phaseA_motor_cognitive_dissociation.csv",
            dissociation_rows,
            list(dissociation_rows[0].keys()),
        )
        save_dissociation_forest_plot(
            output_dir / "figures" / "phaseA_dissociation_forest.png",
            dissociation_rows,
            motor_metric=args.forest_motor_metric,
            cognitive_metric=args.forest_cognitive_metric,
            clinical_columns=CLINICAL_CORRELATION_FIELDS,
        )

    if missing_rows:
        write_csv(
            output_dir / "phaseA_patients_missing_clinical_data.csv",
            missing_rows,
            list(missing_rows[0].keys()),
        )

    print(f"Phase-A outputs written to {output_dir}")
    if dissociation_rows:
        print(
            f"Dissociation table has {len(dissociation_rows)} rows "
            f"({len(DEFAULT_DISSOCIATION_MOTOR_SUBSET)} motor x "
            f"{len(DEFAULT_COGNITIVE_COLUMNS)} cognitive x "
            f"{len(CLINICAL_CORRELATION_FIELDS)} clinical)."
        )
    else:
        print("No clinical matches found; dissociation table not produced.")


if __name__ == "__main__":
    main()
