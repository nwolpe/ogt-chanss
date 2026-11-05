#!/usr/bin/env python3
"""Phase-B mechanism analysis for the OGT / CHANSS paper.

This module sits alongside ``analyze_ogt_groups.py`` (Part-1 pipeline) and
``motor_purification.py`` (Phase A). It implements the cognitive mechanism
measures that are hypothesised to drive reduced uniqueness in patients:

    1. Perseveration index - similarity between consecutive paths, normalised
       by the participant's overall pairwise-distance baseline.
    2. Temporal uniqueness slope - is the participant producing more novel
       paths as the trial progresses (exploration) or fewer (convergence)?
    3. Pre-path deliberation - mean/median/CV of the pause between paths, a
       2-component Gaussian mixture of short vs long pauses, and whether
       long pauses predict higher within-subject prefix uniqueness.
    4. Spatial roaming entropy - Shannon entropy of the cell-visit
       distribution over a discretised canvas, plus binary coverage.
    5. Strategy flexibility - K-means clustering of all test paths across the
       whole cohort, then per-participant count of distinct archetypes and
       the Shannon entropy of their archetype-visit distribution.
    6. (Secondary) within-path submovement count - velocity minima per path,
       as an online re-planning proxy.

It then runs:

    - group comparisons (t-test and Mann-Whitney) on every mechanism
    - Spearman correlations between each mechanism and (a) raw Ang uniqueness,
      (b) Phase-A residualised uniqueness, (c) every clinical field already
      defined in the existing pipeline
    - a Phase-C preview: Shapley / dominance-analysis decomposition of the
      variance in residualised uniqueness attributable to each mechanism
    - a within-subject path-level fit: for each participant, an OLS
      regression of prefix uniqueness on the three trial-level mechanisms
      (distance to previous path, log pause before path, submovements inside
      path), yielding a per-participant R^2 and standardised betas
    - a leave-one-out cross-validated predictive R^2 of the subject-level
      six-mechanism model, with per-participant predicted, observed and
      residual columns so "how well is THIS person explained by the group
      model?" becomes an inspectable quantity

Outputs land in ``analysis_outputs/phaseB/`` by default: six CSVs and three
figures. The CSVs are the participant metrics, group comparison, clinical
correlations, uniqueness correlations, Shapley variance partitioning, within-
subject path-level fit, and leave-one-out predictive fit.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.vq import kmeans2  # noqa: E402
from scipy.spatial.distance import cdist, pdist, squareform  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind  # noqa: E402

from analyze_ogt_groups import (  # noqa: E402
    CLINICAL_CORRELATION_FIELDS,
    ParticipantRoutes,
    X_CENTER,
    Y_CENTER,
    build_clinical_match_rows,
    build_route_feature_vector,
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
from motor_purification import (  # noqa: E402
    DEFAULT_MOTOR_RESIDUALISER_COLUMNS,
    _position_feature_pair,
    compute_motor_fingerprint,
    compute_position_only_uniqueness_metrics,
    residualise_column,
)


# ---------------------------------------------------------------------------
# Constants / canvas defaults
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 2 * int(X_CENTER)
CANVAS_HEIGHT = 2 * int(Y_CENTER)

# Default discretisation for roaming entropy (~50 px cells on an 800x1200 active task canvas).
DEFAULT_GRID_X_CELLS = 16
DEFAULT_GRID_Y_CELLS = 24

# Default number of strategy archetypes for K-means.
DEFAULT_N_CLUSTERS = 8

# Default set of mechanisms to include in the variance-partitioning preview.
DEFAULT_MECHANISM_COLUMNS: list[str] = [
    "perseveration_ratio",
    "temporal_prefix_uniqueness_log_slope",
    "deliberation_mean_ms",
    "roaming_entropy_normalised",
    "flexibility_cluster_entropy_normalised",
]


# ---------------------------------------------------------------------------
# Shared feature-matrix cache
# ---------------------------------------------------------------------------


def build_feature_cache(
    participants: Sequence[ParticipantRoutes],
    np_points: int,
    smooth_window: int,
    velocity_downsample: int,
    distance_mode: str = "full",
) -> dict[str, object]:
    """Build feature matrices once, share across the mechanism functions.

    Returns a dict with:
        features:          (n_total_routes, D) np.ndarray of per-path features
        features_flipped:  same shape, mirror-flipped
        distance_full:     squareform pairwise distance matrix
        distance_flip:     (n_total_routes, n_total_routes) flipped vs original
        participant_index: dict mapping participant_id -> list of global indices
        route_order:       list of (participant_id, route_index_within_subject)
                           aligned with rows of 'features'.
    """
    route_records: list[tuple[str, int, np.ndarray, np.ndarray]] = []

    for participant in participants:
        for route_index, route in enumerate(participant.test_routes):
            if distance_mode == "full":
                pair, _ = build_route_feature_vector(
                    route,
                    np_points=np_points,
                    smooth_window=smooth_window,
                    velocity_downsample=velocity_downsample,
                )
                if pair is None:
                    continue
                feat, feat_flip = pair[0], pair[1]
            else:  # position-only
                pair = _position_feature_pair(route, np_points=np_points)
                if pair is None:
                    continue
                feat, feat_flip = pair[0], pair[1]

            route_records.append(
                (participant.participant_id, route_index, feat, feat_flip)
            )

    if not route_records:
        return {
            "features": np.empty((0, 0)),
            "features_flipped": np.empty((0, 0)),
            "distance_full": np.empty((0, 0)),
            "distance_flip": np.empty((0, 0)),
            "participant_index": {},
            "route_order": [],
        }

    features = np.vstack([rec[2] for rec in route_records])
    features_flipped = np.vstack([rec[3] for rec in route_records])
    distance_full = squareform(pdist(features, metric="euclidean"))
    distance_flip = cdist(features_flipped, features, metric="euclidean")

    participant_index: dict[str, list[int]] = defaultdict(list)
    route_order: list[tuple[str, int]] = []
    for global_idx, (pid, r_idx, _, _) in enumerate(route_records):
        participant_index[pid].append(global_idx)
        route_order.append((pid, r_idx))

    return {
        "features": features,
        "features_flipped": features_flipped,
        "distance_full": distance_full,
        "distance_flip": distance_flip,
        "participant_index": dict(participant_index),
        "route_order": route_order,
    }


def _mirror_aware_distances(
    distance_full: np.ndarray,
    distance_flip: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """For each pair (i, j) with i, j in `indices`, take the minimum of the
    original-vs-original distance and the flipped-vs-original distance.

    Returns a 2D array of shape (len(indices), len(indices)).
    """
    arr = np.asarray(indices, dtype=int)
    d_orig = distance_full[np.ix_(arr, arr)]
    d_flip_rows = distance_flip[np.ix_(arr, arr)]
    d_flip_rows_T = distance_flip[np.ix_(arr, arr)].T
    return np.minimum(d_orig, np.minimum(d_flip_rows, d_flip_rows_T))


# ---------------------------------------------------------------------------
# Mechanism 1: Perseveration
# ---------------------------------------------------------------------------


def compute_perseveration(
    feature_cache: dict[str, object],
    rng_seed: int = 0,
    n_shuffle: int = 500,
) -> dict[str, dict[str, float]]:
    """Consecutive-path similarity, normalised by overall pairwise baseline.

    Low ratios indicate serial perseveration: each path resembles its
    immediate predecessor more than typical paths do.

    A permutation-based z-score is also computed: we shuffle the path order
    many times and recompute the mean consecutive distance to get a null
    distribution.
    """
    results: dict[str, dict[str, float]] = {}
    distance_full = feature_cache["distance_full"]
    distance_flip = feature_cache["distance_flip"]
    participant_index = feature_cache["participant_index"]
    rng = np.random.default_rng(rng_seed)

    for participant_id, indices in participant_index.items():
        indices_arr = list(indices)
        if len(indices_arr) < 3:
            results[participant_id] = {
                "perseveration_consecutive_distance_mean": math.nan,
                "perseveration_all_pairs_distance_mean": math.nan,
                "perseveration_ratio": math.nan,
                "perseveration_shuffle_z": math.nan,
            }
            continue

        subj_dist = _mirror_aware_distances(distance_full, distance_flip, indices_arr)
        n = subj_dist.shape[0]

        consec = np.array([subj_dist[i, i + 1] for i in range(n - 1)], dtype=float)
        triu = subj_dist[np.triu_indices(n, k=1)]
        mean_consec = float(np.mean(consec))
        mean_all = float(np.mean(triu))
        ratio = mean_consec / mean_all if mean_all > 0 else math.nan

        # Permutation null: fix the distance matrix, shuffle the path order.
        null_means = np.empty(n_shuffle, dtype=float)
        for s in range(n_shuffle):
            perm = rng.permutation(n)
            d_perm = subj_dist[np.ix_(perm, perm)]
            null_means[s] = float(
                np.mean([d_perm[i, i + 1] for i in range(n - 1)])
            )
        null_mean = float(np.mean(null_means))
        null_sd = float(np.std(null_means, ddof=1)) if null_means.size > 1 else math.nan
        z = (mean_consec - null_mean) / null_sd if null_sd and null_sd > 0 else math.nan

        results[participant_id] = {
            "perseveration_consecutive_distance_mean": mean_consec,
            "perseveration_all_pairs_distance_mean": mean_all,
            "perseveration_ratio": ratio,
            "perseveration_shuffle_z": z,
        }

    return results


# ---------------------------------------------------------------------------
# Mechanism 2: Temporal uniqueness slope
# ---------------------------------------------------------------------------


def _prefix_uniqueness(
    subj_dist: np.ndarray,
) -> list[float]:
    """Prefix uniqueness at each trial: min distance to any preceding trial.

    Position 0 has no predecessors, so returns NaN at index 0.
    """
    n = subj_dist.shape[0]
    values: list[float] = [math.nan]
    for k in range(1, n):
        prior = subj_dist[k, :k]
        values.append(float(np.min(prior)))
    return values


def compute_temporal_uniqueness_slope(
    feature_cache: dict[str, object],
) -> dict[str, dict[str, float]]:
    """Regress within-subject prefix uniqueness on trial index.

    A flat or positive slope indicates productive exploration / maintained
    novelty. A negative slope means the participant is converging on similar
    paths as the trial goes on. Reported metrics:

        temporal_prefix_uniqueness_spearman_rho/p - rank-trend
        temporal_prefix_uniqueness_log_slope      - OLS slope of log(prefix_u)
                                                    on trial index
        temporal_first_half_prefix_u_mean/second_half_mean  - halves contrast
        temporal_prefix_u_halves_ratio            - second-half / first-half
    """
    results: dict[str, dict[str, float]] = {}
    distance_full = feature_cache["distance_full"]
    distance_flip = feature_cache["distance_flip"]
    participant_index = feature_cache["participant_index"]

    for participant_id, indices in participant_index.items():
        if len(indices) < 4:
            results[participant_id] = {
                "temporal_prefix_uniqueness_spearman_rho": math.nan,
                "temporal_prefix_uniqueness_spearman_p": math.nan,
                "temporal_prefix_uniqueness_log_slope": math.nan,
                "temporal_first_half_prefix_u_mean": math.nan,
                "temporal_second_half_prefix_u_mean": math.nan,
                "temporal_prefix_u_halves_ratio": math.nan,
            }
            continue

        subj_dist = _mirror_aware_distances(distance_full, distance_flip, list(indices))
        prefix_u = np.asarray(_prefix_uniqueness(subj_dist), dtype=float)
        trial_idx = np.arange(prefix_u.size, dtype=float)

        valid = ~np.isnan(prefix_u)
        if np.count_nonzero(valid) < 4:
            results[participant_id] = {
                "temporal_prefix_uniqueness_spearman_rho": math.nan,
                "temporal_prefix_uniqueness_spearman_p": math.nan,
                "temporal_prefix_uniqueness_log_slope": math.nan,
                "temporal_first_half_prefix_u_mean": math.nan,
                "temporal_second_half_prefix_u_mean": math.nan,
                "temporal_prefix_u_halves_ratio": math.nan,
            }
            continue

        rho, p_rho = spearmanr(trial_idx[valid], prefix_u[valid])

        positive = prefix_u[valid] > 0
        if np.count_nonzero(positive) >= 4:
            y = np.log(prefix_u[valid][positive])
            x = trial_idx[valid][positive]
            # OLS slope by hand to keep dependencies minimal.
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            denom = float(np.sum((x - x_mean) ** 2))
            log_slope = (
                float(np.sum((x - x_mean) * (y - y_mean)) / denom)
                if denom > 0
                else math.nan
            )
        else:
            log_slope = math.nan

        half = max(1, int(np.count_nonzero(valid) // 2))
        valid_values = prefix_u[valid]
        first_half = float(np.mean(valid_values[:half]))
        second_half = float(np.mean(valid_values[half:]))
        halves_ratio = second_half / first_half if first_half > 0 else math.nan

        results[participant_id] = {
            "temporal_prefix_uniqueness_spearman_rho": float(rho)
            if not math.isnan(rho)
            else math.nan,
            "temporal_prefix_uniqueness_spearman_p": float(p_rho)
            if not math.isnan(p_rho)
            else math.nan,
            "temporal_prefix_uniqueness_log_slope": log_slope,
            "temporal_first_half_prefix_u_mean": first_half,
            "temporal_second_half_prefix_u_mean": second_half,
            "temporal_prefix_u_halves_ratio": halves_ratio,
        }

    return results


# ---------------------------------------------------------------------------
# Mechanism 3: Pre-path deliberation
# ---------------------------------------------------------------------------


def _fit_1d_two_gmm(
    samples: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a 2-component 1-D Gaussian mixture by EM.

    Returns (pi, mu, sigma) sorted by mu ascending, or None if the sample is
    too small. Inputs containing NaNs are dropped.
    """
    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 6:
        return None

    med = float(np.median(x))
    low = x[x <= med]
    high = x[x > med]
    if low.size < 2 or high.size < 2:
        return None

    pi = np.array([0.5, 0.5])
    mu = np.array([float(np.mean(low)), float(np.mean(high))])
    sigma = np.array([float(np.std(low)) + 1e-6, float(np.std(high)) + 1e-6])

    prev_ll = -np.inf
    for _ in range(max_iter):
        diff = x[:, None] - mu[None, :]
        expo = -0.5 * (diff / sigma[None, :]) ** 2
        pdf = np.exp(expo) / (sigma[None, :] * math.sqrt(2 * math.pi))
        resp = pi[None, :] * pdf
        total = resp.sum(axis=1, keepdims=True) + 1e-12
        resp = resp / total
        ll = float(np.sum(np.log(total)))

        nk = resp.sum(axis=0)
        nk_safe = np.where(nk > 0, nk, 1e-12)
        pi = nk / x.size
        mu = (resp * x[:, None]).sum(axis=0) / nk_safe
        sigma = np.sqrt((resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk_safe) + 1e-6

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    order = np.argsort(mu)
    return pi[order], mu[order], sigma[order]


def compute_deliberation(
    participants: Sequence[ParticipantRoutes],
    feature_cache: dict[str, object],
) -> dict[str, dict[str, float]]:
    """Deliberation / inter-path pause dynamics.

    For each participant we compute basic descriptors of the inter-path
    interval distribution, a 2-component Gaussian mixture on log-pauses,
    and - crucially - the within-subject Spearman correlation between log
    pause before path N and the prefix uniqueness of path N. The latter
    asks: does 'thinking longer' actually buy more novelty?
    """
    results: dict[str, dict[str, float]] = {}
    distance_full = feature_cache["distance_full"]
    distance_flip = feature_cache["distance_flip"]
    participant_index = feature_cache["participant_index"]

    for participant in participants:
        pid = participant.participant_id
        routes = participant.test_routes

        if len(routes) < 3:
            results[pid] = {
                "deliberation_n_intervals": 0,
                "deliberation_mean_ms": math.nan,
                "deliberation_median_ms": math.nan,
                "deliberation_cv": math.nan,
                "deliberation_skew": math.nan,
                "deliberation_short_mean_ms": math.nan,
                "deliberation_long_mean_ms": math.nan,
                "deliberation_long_proportion": math.nan,
                "deliberation_pause_predicts_uniqueness_rho": math.nan,
                "deliberation_pause_predicts_uniqueness_p": math.nan,
            }
            continue

        starts = np.array([route[0, 0] for route in routes], dtype=float)
        ends = np.array([route[-1, 0] for route in routes], dtype=float)
        pauses = starts[1:] - ends[:-1]
        pauses = pauses[np.isfinite(pauses) & (pauses > 0)]

        if pauses.size < 2:
            descriptive = {
                "deliberation_n_intervals": int(pauses.size),
                "deliberation_mean_ms": math.nan,
                "deliberation_median_ms": math.nan,
                "deliberation_cv": math.nan,
                "deliberation_skew": math.nan,
                "deliberation_short_mean_ms": math.nan,
                "deliberation_long_mean_ms": math.nan,
                "deliberation_long_proportion": math.nan,
                "deliberation_pause_predicts_uniqueness_rho": math.nan,
                "deliberation_pause_predicts_uniqueness_p": math.nan,
            }
            results[pid] = descriptive
            continue

        mean_p = float(np.mean(pauses))
        median_p = float(np.median(pauses))
        std_p = float(np.std(pauses, ddof=1)) if pauses.size > 1 else math.nan
        cv = std_p / mean_p if mean_p > 0 else math.nan
        m3 = float(np.mean((pauses - mean_p) ** 3))
        skew = m3 / (std_p ** 3) if std_p and std_p > 0 else math.nan

        gmm = _fit_1d_two_gmm(np.log(pauses))
        if gmm is None:
            short_mean = long_mean = long_prop = math.nan
        else:
            pi, mu, _ = gmm
            short_mean = float(np.exp(mu[0]))
            long_mean = float(np.exp(mu[1]))
            long_prop = float(pi[1])

        # Does pause_N predict prefix_uniqueness_N?
        indices = participant_index.get(pid, [])
        rho = p_rho = math.nan
        if len(indices) >= 4:
            subj_dist = _mirror_aware_distances(
                distance_full, distance_flip, list(indices)
            )
            prefix_u = np.asarray(_prefix_uniqueness(subj_dist), dtype=float)
            # pauses[i] is the pause BEFORE route i+1 (i = 0..len-2).
            # So pause_for_path[k] exists for k = 1..len-1.
            pause_for_path = np.concatenate([[math.nan], pauses])
            k = min(pause_for_path.size, prefix_u.size)
            logp = np.log(pause_for_path[:k])
            pu = prefix_u[:k]
            valid = (~np.isnan(logp)) & (~np.isnan(pu)) & (pu > 0)
            if np.count_nonzero(valid) >= 4:
                rho, p_rho = spearmanr(logp[valid], pu[valid])
                rho = float(rho)
                p_rho = float(p_rho)

        results[pid] = {
            "deliberation_n_intervals": int(pauses.size),
            "deliberation_mean_ms": mean_p,
            "deliberation_median_ms": median_p,
            "deliberation_cv": float(cv) if not math.isnan(cv) else math.nan,
            "deliberation_skew": float(skew) if not math.isnan(skew) else math.nan,
            "deliberation_short_mean_ms": short_mean,
            "deliberation_long_mean_ms": long_mean,
            "deliberation_long_proportion": long_prop,
            "deliberation_pause_predicts_uniqueness_rho": rho,
            "deliberation_pause_predicts_uniqueness_p": p_rho,
        }

    return results


# ---------------------------------------------------------------------------
# Mechanism 4: Spatial roaming entropy
# ---------------------------------------------------------------------------


def compute_roaming_entropy(
    participants: Sequence[ParticipantRoutes],
    grid_x_cells: int = DEFAULT_GRID_X_CELLS,
    grid_y_cells: int = DEFAULT_GRID_Y_CELLS,
    canvas_width: int = CANVAS_WIDTH,
    canvas_height: int = CANVAS_HEIGHT,
) -> dict[str, dict[str, float]]:
    """Shannon entropy of spatial visit distribution over a discretised canvas.

    Two related measures:
        roaming_entropy_normalised - Shannon entropy divided by log(n_cells),
            so in [0, 1]. Close to 1 = very evenly spread across space;
            close to 0 = concentrated in a few cells.
        spatial_coverage_fraction  - fraction of grid cells visited at least
            once. Ignores how much time was spent per cell.
    """
    results: dict[str, dict[str, float]] = {}
    n_cells = grid_x_cells * grid_y_cells
    log_n_cells = math.log(n_cells) if n_cells > 1 else math.nan

    for participant in participants:
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for route in participant.test_routes:
            if route.shape[0] < 2:
                continue
            all_x.append(route[:, 1])
            all_y.append(route[:, 2])

        if not all_x:
            results[participant.participant_id] = {
                "roaming_entropy_raw": math.nan,
                "roaming_entropy_normalised": math.nan,
                "spatial_coverage_fraction": math.nan,
                "roaming_grid_n_cells": n_cells,
            }
            continue

        x = np.concatenate(all_x)
        y = np.concatenate(all_y)

        # Bin into the grid.
        bx = np.floor(x / canvas_width * grid_x_cells).astype(int)
        by = np.floor(y / canvas_height * grid_y_cells).astype(int)
        bx = np.clip(bx, 0, grid_x_cells - 1)
        by = np.clip(by, 0, grid_y_cells - 1)

        cell_id = by * grid_x_cells + bx
        counts = np.bincount(cell_id, minlength=n_cells).astype(float)
        total = counts.sum()
        if total <= 0:
            results[participant.participant_id] = {
                "roaming_entropy_raw": math.nan,
                "roaming_entropy_normalised": math.nan,
                "spatial_coverage_fraction": math.nan,
                "roaming_grid_n_cells": n_cells,
            }
            continue

        probs = counts / total
        nonzero = probs[probs > 0]
        entropy_raw = float(-np.sum(nonzero * np.log(nonzero)))
        entropy_norm = entropy_raw / log_n_cells if log_n_cells > 0 else math.nan
        coverage = float(np.count_nonzero(counts)) / n_cells

        results[participant.participant_id] = {
            "roaming_entropy_raw": entropy_raw,
            "roaming_entropy_normalised": float(entropy_norm),
            "spatial_coverage_fraction": coverage,
            "roaming_grid_n_cells": n_cells,
        }

    return results


# ---------------------------------------------------------------------------
# Mechanism 5: Strategy flexibility via cohort-wide K-means clustering
# ---------------------------------------------------------------------------


def compute_strategy_flexibility(
    feature_cache: dict[str, object],
    n_clusters: int = DEFAULT_N_CLUSTERS,
    seed: int = 0,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    """K-means clustering of all test paths across the cohort.

    Returns a per-participant dict with (flexibility_clusters_used,
    flexibility_cluster_entropy_normalised) plus a description of the
    archetypes (centroids in feature space and the cluster-assignment array
    aligned with feature_cache['route_order']).
    """
    features = feature_cache["features"]
    features_flipped = feature_cache["features_flipped"]
    participant_index = feature_cache["participant_index"]
    route_order = feature_cache["route_order"]

    if features.shape[0] < n_clusters:
        empty_results: dict[str, dict[str, float]] = {
            pid: {
                "flexibility_clusters_used": 0,
                "flexibility_cluster_entropy_normalised": math.nan,
                "flexibility_n_clusters_global": int(n_clusters),
            }
            for pid in participant_index
        }
        return empty_results, {
            "centroids": np.empty((0, 0)),
            "assignments": np.empty((0,), dtype=int),
            "route_order": route_order,
            "n_clusters": n_clusters,
        }

    rng = np.random.default_rng(seed)
    # kmeans2 accepts a seeding mode; we use 'random' then run multiple inits.
    best_centroids = None
    best_labels = None
    best_inertia = np.inf
    for trial in range(5):
        trial_seed = int(rng.integers(0, 2 ** 31 - 1))
        np.random.seed(trial_seed)
        centroids, labels = kmeans2(
            features, n_clusters, minit="++", missing="raise"
        )
        d = cdist(features, centroids)
        inertia = float(np.sum(np.min(d, axis=1) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    # Mirror-aware assignment: pick min(d(orig, c), d(flipped, c)).
    d_orig = cdist(features, best_centroids)
    d_flip = cdist(features_flipped, best_centroids)
    assignments = np.argmin(np.minimum(d_orig, d_flip), axis=1)

    log_k = math.log(n_clusters)
    results: dict[str, dict[str, float]] = {}
    for pid, indices in participant_index.items():
        if not indices:
            results[pid] = {
                "flexibility_clusters_used": 0,
                "flexibility_cluster_entropy_normalised": math.nan,
                "flexibility_n_clusters_global": int(n_clusters),
            }
            continue
        my_assign = assignments[np.asarray(indices, dtype=int)]
        counts = np.bincount(my_assign, minlength=n_clusters).astype(float)
        total = counts.sum()
        if total <= 0:
            results[pid] = {
                "flexibility_clusters_used": 0,
                "flexibility_cluster_entropy_normalised": math.nan,
                "flexibility_n_clusters_global": int(n_clusters),
            }
            continue
        probs = counts / total
        nonzero = probs[probs > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)))
        entropy_norm = entropy / log_k if log_k > 0 else math.nan
        results[pid] = {
            "flexibility_clusters_used": int(np.count_nonzero(counts)),
            "flexibility_cluster_entropy_normalised": float(entropy_norm),
            "flexibility_n_clusters_global": int(n_clusters),
        }

    cluster_info = {
        "centroids": best_centroids,
        "assignments": assignments,
        "route_order": route_order,
        "n_clusters": n_clusters,
    }
    return results, cluster_info


# ---------------------------------------------------------------------------
# Mechanism 6: Within-path submovements (secondary / optional)
# ---------------------------------------------------------------------------


def _count_submovements(route: np.ndarray, smooth_window: int) -> int:
    """Count velocity minima (local minima of smoothed speed) inside one path.

    Uses the first and last samples as boundaries; internal local minima
    indicate intra-path slowdowns that plausibly reflect online re-planning.
    """
    if route.shape[0] < 5:
        return 0
    time_ms = route[:, 0]
    dx = np.diff(route[:, 1])
    dy = np.diff(route[:, 2])
    dt = np.diff(time_ms) / 1000.0
    valid = dt > 0
    speed = np.zeros_like(dx)
    speed[valid] = np.hypot(dx[valid], dy[valid]) / dt[valid]
    speed = moving_average(speed, smooth_window)
    if speed.size < 3:
        return 0
    # Strict interior local minima.
    mins = np.flatnonzero(
        (speed[1:-1] < speed[:-2]) & (speed[1:-1] < speed[2:])
    )
    return int(mins.size)


def compute_submovements(
    participants: Sequence[ParticipantRoutes],
    smooth_window: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for participant in participants:
        counts = [
            _count_submovements(route, smooth_window)
            for route in participant.test_routes
            if route.shape[0] >= 5
        ]
        if not counts:
            results[participant.participant_id] = {
                "submovements_mean_per_path": math.nan,
                "submovements_total": 0,
                "submovements_n_paths": 0,
            }
            continue
        results[participant.participant_id] = {
            "submovements_mean_per_path": float(np.mean(counts)),
            "submovements_total": int(np.sum(counts)),
            "submovements_n_paths": int(len(counts)),
        }
    return results


# ---------------------------------------------------------------------------
# Group comparisons and correlations
# ---------------------------------------------------------------------------


def compare_group_rows(
    participant_rows: list[dict[str, object]],
    metric_columns: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    controls = [r for r in participant_rows if r.get("group") == "control"]
    patients = [r for r in participant_rows if r.get("group") == "patient"]

    for metric in metric_columns:
        ctrl_vals = np.asarray(
            [to_float(r.get(metric, math.nan)) for r in controls], dtype=float
        )
        pat_vals = np.asarray(
            [to_float(r.get(metric, math.nan)) for r in patients], dtype=float
        )
        ctrl_valid = ctrl_vals[~np.isnan(ctrl_vals)]
        pat_valid = pat_vals[~np.isnan(pat_vals)]

        ttest_p = math.nan
        mw_p = math.nan
        if ctrl_valid.size >= 2 and pat_valid.size >= 2:
            ttest_p = float(
                ttest_ind(ctrl_valid, pat_valid, equal_var=False).pvalue
            )
            mw_p = float(
                mannwhitneyu(ctrl_valid, pat_valid, alternative="two-sided").pvalue
            )

        rows.append(
            {
                "metric": metric,
                "n_control": int(ctrl_valid.size),
                "n_patient": int(pat_valid.size),
                "mean_control": float(np.mean(ctrl_valid))
                if ctrl_valid.size
                else math.nan,
                "mean_patient": float(np.mean(pat_valid))
                if pat_valid.size
                else math.nan,
                "median_control": float(np.median(ctrl_valid))
                if ctrl_valid.size
                else math.nan,
                "median_patient": float(np.median(pat_valid))
                if pat_valid.size
                else math.nan,
                "ttest_p": ttest_p,
                "mannwhitney_p": mw_p,
            }
        )
    return rows


def correlate_columns(
    rows: list[dict[str, object]],
    columns_x: Sequence[str],
    columns_y: Sequence[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for x_name in columns_x:
        x_vals = np.asarray(
            [to_float(r.get(x_name, math.nan)) for r in rows], dtype=float
        )
        for y_name in columns_y:
            y_vals = np.asarray(
                [to_float(r.get(y_name, math.nan)) for r in rows], dtype=float
            )
            mask = (~np.isnan(x_vals)) & (~np.isnan(y_vals))
            n = int(np.count_nonzero(mask))
            rho = p = math.nan
            if n >= 4:
                rho_s, p_s = spearmanr(x_vals[mask], y_vals[mask])
                rho = float(rho_s)
                p = float(p_s)
            out.append(
                {
                    "mechanism": x_name,
                    "target": y_name,
                    "n": n,
                    "spearman_rho": rho,
                    "spearman_p": p,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Phase-C preview: Shapley decomposition of R^2
# ---------------------------------------------------------------------------


def _r_squared(y: np.ndarray, X: np.ndarray) -> float:
    if X.size == 0:
        return 0.0
    Xc = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    y_hat = Xc @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def shapley_r_squared(
    rows: list[dict[str, object]],
    target: str,
    predictors: Sequence[str],
    group_filter: str | None = "patient",
) -> list[dict[str, object]]:
    """Dominance-analysis / Shapley decomposition of R^2.

    For K predictors, enumerates all 2^K subsets and computes each
    predictor's average marginal contribution to R^2. For K = 4 this is 16
    model fits - trivial.
    """
    preds = list(predictors)
    K = len(preds)
    if K == 0:
        return []

    y_list: list[float] = []
    x_list: list[list[float]] = []
    for row in rows:
        if group_filter is not None and row.get("group") != group_filter:
            continue
        yv = to_float(row.get(target, math.nan))
        xv = [to_float(row.get(p, math.nan)) for p in preds]
        if math.isnan(yv) or any(math.isnan(v) for v in xv):
            continue
        y_list.append(yv)
        x_list.append(xv)

    n = len(y_list)
    if n < K + 3:
        return [
            {
                "predictor": p,
                "shapley_r_squared": math.nan,
                "total_r_squared": math.nan,
                "n": n,
            }
            for p in preds
        ]

    y = np.asarray(y_list, dtype=float)
    X = np.asarray(x_list, dtype=float)

    subset_r2: dict[frozenset[int], float] = {}
    subset_r2[frozenset()] = 0.0
    for r in range(1, K + 1):
        for subset in combinations(range(K), r):
            key = frozenset(subset)
            cols = np.asarray(subset, dtype=int)
            subset_r2[key] = _r_squared(y, X[:, cols])

    # Shapley value for predictor i:
    #   sum over subsets S not containing i of
    #   |S|! * (K - |S| - 1)! / K! * (R2(S u {i}) - R2(S))
    total_r2 = subset_r2[frozenset(range(K))]
    factorial = [math.factorial(k) for k in range(K + 1)]
    results: list[dict[str, object]] = []
    for i, name in enumerate(preds):
        shap = 0.0
        for subset_key, r2 in subset_r2.items():
            if i in subset_key:
                continue
            added = subset_key | {i}
            gain = subset_r2[added] - r2
            s = len(subset_key)
            weight = factorial[s] * factorial[K - s - 1] / factorial[K]
            shap += weight * gain
        results.append(
            {
                "predictor": name,
                "shapley_r_squared": float(shap),
                "total_r_squared": float(total_r2),
                "n": n,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Model-fit diagnostics: how well do the mechanisms actually explain uniqueness?
# ---------------------------------------------------------------------------


def compute_within_subject_path_fit(
    participants: Sequence[ParticipantRoutes],
    feature_cache: dict[str, object],
    smooth_window: int,
    min_paths: int = 6,
) -> dict[str, dict[str, float]]:
    """Per-participant OLS fit of path-level prefix uniqueness on the three
    trial-level mechanism predictors.

    For each participant with at least ``min_paths`` usable paths, regresses
    the path-level outcome

        y_k  = prefix_uniqueness(path k)

    on three path-level predictors

        x1_k = distance(path k, path k-1)       # perseveration at trial level
        x2_k = log(pause preceding path k)      # deliberation at trial level
        x3_k = submovement count within path k  # re-planning at trial level

    for k = 1 .. N-1. Returns a dict keyed by participant_id with:

        within_fit_n_paths
        within_fit_r_squared                 unstandardised OLS R^2
        within_fit_adjusted_r_squared        Adjusted for K=3 predictors
        within_fit_beta_consec_distance      Standardised beta
        within_fit_beta_log_pause
        within_fit_beta_submovements
        within_fit_rho_consec_distance       Marginal Spearman rho
        within_fit_rho_log_pause
        within_fit_rho_submovements
    """
    distance_full = feature_cache["distance_full"]
    distance_flip = feature_cache["distance_flip"]
    participant_index = feature_cache["participant_index"]

    nan_template: dict[str, float] = {
        "within_fit_n_paths": 0,
        "within_fit_r_squared": math.nan,
        "within_fit_adjusted_r_squared": math.nan,
        "within_fit_beta_consec_distance": math.nan,
        "within_fit_beta_log_pause": math.nan,
        "within_fit_beta_submovements": math.nan,
        "within_fit_rho_consec_distance": math.nan,
        "within_fit_rho_log_pause": math.nan,
        "within_fit_rho_submovements": math.nan,
    }

    results: dict[str, dict[str, float]] = {}

    for participant in participants:
        pid = participant.participant_id
        routes = participant.test_routes
        indices = list(participant_index.get(pid, []))

        if len(indices) < min_paths or len(routes) < min_paths:
            results[pid] = dict(nan_template)
            continue

        subj_dist = _mirror_aware_distances(distance_full, distance_flip, indices)
        n = subj_dist.shape[0]
        prefix_u = np.asarray(_prefix_uniqueness(subj_dist), dtype=float)

        # x1: distance to immediately-preceding path.
        consec = np.full(n, math.nan, dtype=float)
        for k in range(1, n):
            consec[k] = subj_dist[k, k - 1]

        # x2: log of the pause immediately before each path.
        log_pause = np.full(n, math.nan, dtype=float)
        n_routes = min(n, len(routes))
        starts = np.array([routes[k][0, 0] for k in range(n_routes)], dtype=float)
        ends = np.array([routes[k][-1, 0] for k in range(n_routes)], dtype=float)
        for k in range(1, n_routes):
            pause = starts[k] - ends[k - 1]
            if np.isfinite(pause) and pause > 0:
                log_pause[k] = math.log(pause)

        # x3: submovements inside the current path.
        subs = np.full(n, math.nan, dtype=float)
        for k in range(n_routes):
            if routes[k].shape[0] >= 5:
                subs[k] = float(_count_submovements(routes[k], smooth_window))

        y_all = prefix_u[1:n]
        X_all = np.column_stack([consec[1:n], log_pause[1:n], subs[1:n]])

        mask = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
        n_eff = int(np.count_nonzero(mask))
        if n_eff < min_paths:
            results[pid] = dict(nan_template)
            continue

        y_v = y_all[mask]
        X_v = X_all[mask]

        r2 = _r_squared(y_v, X_v)
        K = X_v.shape[1]
        adj_r2 = (
            1.0 - (1.0 - r2) * (n_eff - 1) / (n_eff - K - 1)
            if (np.isfinite(r2) and n_eff - K - 1 > 0)
            else math.nan
        )

        # Standardised betas (so coefficients are comparable across people).
        x_means = X_v.mean(axis=0)
        x_stds = X_v.std(axis=0, ddof=1)
        y_mean = float(y_v.mean())
        y_std = float(y_v.std(ddof=1))

        if np.any(x_stds <= 0) or y_std <= 0:
            beta_z = np.full(K + 1, math.nan)
        else:
            X_z = (X_v - x_means) / x_stds
            y_z = (y_v - y_mean) / y_std
            Xc = np.column_stack([np.ones(n_eff), X_z])
            beta_z, *_ = np.linalg.lstsq(Xc, y_z, rcond=None)

        # Marginal Spearman rho per predictor (robust univariate check).
        def _rho(col: np.ndarray) -> float:
            if n_eff < 4:
                return math.nan
            if np.std(col, ddof=1) <= 0:
                return math.nan
            r, _ = spearmanr(col, y_v)
            return float(r) if np.isfinite(r) else math.nan

        results[pid] = {
            "within_fit_n_paths": n_eff,
            "within_fit_r_squared": float(r2) if np.isfinite(r2) else math.nan,
            "within_fit_adjusted_r_squared": float(adj_r2)
            if np.isfinite(adj_r2)
            else math.nan,
            "within_fit_beta_consec_distance": float(beta_z[1])
            if np.isfinite(beta_z[1])
            else math.nan,
            "within_fit_beta_log_pause": float(beta_z[2])
            if np.isfinite(beta_z[2])
            else math.nan,
            "within_fit_beta_submovements": float(beta_z[3])
            if np.isfinite(beta_z[3])
            else math.nan,
            "within_fit_rho_consec_distance": _rho(X_v[:, 0]),
            "within_fit_rho_log_pause": _rho(X_v[:, 1]),
            "within_fit_rho_submovements": _rho(X_v[:, 2]),
        }

    return results


def loo_predictive_r_squared(
    rows: list[dict[str, object]],
    target: str,
    predictors: Sequence[str],
    group_filter: str | None = "patient",
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Leave-one-out cross-validated predictive fit of ``target`` from
    ``predictors`` at the subject level.

    Fits an OLS model on all participants except one, predicts the held-out
    participant's ``target`` from their mechanism profile, and records the
    residual. Does this for every kept participant.

    Returns (per_participant_rows, summary_dict).

    per_participant_rows each contain:
        participant_id, group, target, observed, predicted,
        residual, abs_residual, squared_standardised_residual
    summary_dict contains the cohort-level LOO R^2
        loo_r_squared = 1 - SS_res_loo / SS_tot_observed
    which can be negative if the model predicts worse than the mean; plus the
    Spearman correlation between predicted and observed.
    """
    preds = list(predictors)
    K = len(preds)

    kept_rows: list[dict[str, object]] = []
    y_list: list[float] = []
    x_list: list[list[float]] = []
    for row in rows:
        if group_filter is not None and row.get("group") != group_filter:
            continue
        yv = to_float(row.get(target, math.nan))
        xv = [to_float(row.get(p, math.nan)) for p in preds]
        if math.isnan(yv) or any(math.isnan(v) for v in xv):
            continue
        kept_rows.append(row)
        y_list.append(yv)
        x_list.append(xv)

    n = len(y_list)
    summary: dict[str, float] = {
        "target": target,
        "n": n,
        "k_predictors": K,
        "loo_r_squared": math.nan,
        "loo_rmse": math.nan,
        "loo_mean_abs_error": math.nan,
        "predicted_observed_spearman_rho": math.nan,
        "predicted_observed_spearman_p": math.nan,
    }
    per_rows: list[dict[str, object]] = []

    if n < K + 3:
        return per_rows, summary

    y = np.asarray(y_list, dtype=float)
    X = np.asarray(x_list, dtype=float)
    X_full = np.column_stack([np.ones(n), X])

    predictions = np.full(n, math.nan, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta, *_ = np.linalg.lstsq(X_full[mask], y[mask], rcond=None)
        predictions[i] = float(X_full[i] @ beta)

    residuals = y - predictions
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    loo_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    rmse = float(np.sqrt(ss_res / n)) if n > 0 else math.nan
    mae = float(np.mean(np.abs(residuals))) if n > 0 else math.nan

    rho_val = p_val = math.nan
    if n >= 4 and np.std(predictions, ddof=1) > 0 and np.std(y, ddof=1) > 0:
        rho_s, p_s = spearmanr(predictions, y)
        rho_val = float(rho_s)
        p_val = float(p_s)

    summary.update(
        {
            "loo_r_squared": float(loo_r2) if np.isfinite(loo_r2) else math.nan,
            "loo_rmse": rmse,
            "loo_mean_abs_error": mae,
            "predicted_observed_spearman_rho": rho_val,
            "predicted_observed_spearman_p": p_val,
        }
    )

    sd_tot = math.sqrt(ss_tot / n) if ss_tot > 0 and n > 0 else math.nan
    for i, row in enumerate(kept_rows):
        std_res = (
            residuals[i] / sd_tot
            if (sd_tot and sd_tot > 0 and np.isfinite(sd_tot))
            else math.nan
        )
        per_rows.append(
            {
                "participant_id": row.get("participant_id", ""),
                "group": row.get("group", ""),
                "target": target,
                "observed": float(y[i]),
                "predicted": float(predictions[i]),
                "residual": float(residuals[i]),
                "abs_residual": float(abs(residuals[i])),
                "squared_standardised_residual": float(std_res ** 2)
                if (isinstance(std_res, float) and np.isfinite(std_res))
                else math.nan,
            }
        )

    return per_rows, summary


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def save_mechanism_rainclouds(
    path: Path,
    participant_rows: list[dict[str, object]],
    metrics: Sequence[str],
) -> None:
    groups = ["control", "patient"]
    colors = ["#5A7D9A", "#C26D52"]
    metrics = [m for m in metrics if m]
    if not metrics:
        return

    n = len(metrics)
    cols = min(3, n)
    rows_plot = int(math.ceil(n / cols))
    fig, axes = plt.subplots(
        rows_plot, cols, figsize=(4.2 * cols, 3.2 * rows_plot), squeeze=False
    )
    rng = np.random.default_rng(0)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        data = []
        for g in groups:
            values = np.asarray(
                [
                    to_float(r.get(metric, math.nan))
                    for r in participant_rows
                    if r.get("group") == g
                ],
                dtype=float,
            )
            data.append(values[~np.isnan(values)])

        if not any(v.size for v in data):
            ax.set_visible(False)
            continue

        positions = np.arange(len(groups), dtype=float)
        nonempty = [(i, values) for i, values in enumerate(data) if values.size > 0]
        if nonempty:
            violin_positions = [positions[i] - 0.12 for i, _ in nonempty]
            violin_data = [values for _, values in nonempty]
            parts = ax.violinplot(
                violin_data,
                positions=violin_positions,
                widths=0.5,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, (i, _) in zip(parts["bodies"], nonempty):
                body.set_facecolor(colors[i])
                body.set_edgecolor("none")
                body.set_alpha(0.28)

            ax.boxplot(
                violin_data,
                positions=[positions[i] + 0.05 for i, _ in nonempty],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
            )

        for i, values in enumerate(data):
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.06, 0.06, size=values.size)
            ax.scatter(
                np.full(values.size, positions[i] + 0.22) + jitter,
                values,
                s=14,
                alpha=0.55,
                color=colors[i],
                edgecolors="white",
                linewidths=0.3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(["Ctrl", "Pat"])
        ax.set_title(metric.replace("_", " "), fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, rows_plot * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_strategy_archetypes(
    path: Path,
    cluster_info: dict[str, object],
    feature_cache: dict[str, object],
    participants: Sequence[ParticipantRoutes],
    np_points: int,
) -> None:
    """Draw each cluster as the mean of its constituent (resampled) paths."""
    assignments = cluster_info.get("assignments", np.empty((0,), dtype=int))
    route_order = cluster_info.get("route_order", [])
    n_clusters = int(cluster_info.get("n_clusters", 0))
    if assignments.size == 0 or n_clusters == 0:
        return

    # Build resampled complex paths for each global index (used for plotting).
    path_lookup: dict[tuple[str, int], np.ndarray] = {}
    for participant in participants:
        for r_idx, route in enumerate(participant.test_routes):
            z = resample_route(route, np_points=np_points)
            if z is not None:
                path_lookup[(participant.participant_id, r_idx)] = z

    cluster_paths: dict[int, list[np.ndarray]] = defaultdict(list)
    for global_idx, (pid, r_idx) in enumerate(route_order):
        key = (pid, r_idx)
        if key not in path_lookup:
            continue
        cluster_paths[int(assignments[global_idx])].append(path_lookup[key])

    cols = min(4, n_clusters)
    rows_plot = int(math.ceil(n_clusters / cols))
    fig, axes = plt.subplots(
        rows_plot, cols, figsize=(3.0 * cols, 2.4 * rows_plot), squeeze=False
    )

    for k in range(n_clusters):
        ax = axes[k // cols][k % cols]
        paths = cluster_paths.get(k, [])
        if not paths:
            ax.set_visible(False)
            continue

        for z in paths[:40]:
            ax.plot(np.real(z), np.imag(z), color="#888", lw=0.4, alpha=0.4)

        stacked_real = np.stack([np.real(z) for z in paths], axis=0)
        stacked_imag = np.stack([np.imag(z) for z in paths], axis=0)
        ax.plot(
            np.mean(stacked_real, axis=0),
            np.mean(stacked_imag, axis=0),
            color="#C26D52",
            lw=1.8,
        )
        ax.set_title(f"Cluster {k} (n={len(paths)})", fontsize=9)
        ax.set_xlim(0, CANVAS_WIDTH)
        ax.set_ylim(CANVAS_HEIGHT, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_clusters, rows_plot * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mechanism_clinical_heatmap(
    path: Path,
    correlation_rows: list[dict[str, object]],
    mechanism_columns: Sequence[str],
    clinical_columns: Sequence[str],
) -> None:
    if not correlation_rows:
        return
    mech_list = list(mechanism_columns)
    clin_list = list(clinical_columns)
    mat = np.full((len(mech_list), len(clin_list)), np.nan, dtype=float)
    p_mat = np.full_like(mat, np.nan)
    for row in correlation_rows:
        if row["mechanism"] not in mech_list or row["target"] not in clin_list:
            continue
        i = mech_list.index(str(row["mechanism"]))
        j = clin_list.index(str(row["target"]))
        mat[i, j] = to_float(row.get("spearman_rho", math.nan))
        p_mat[i, j] = to_float(row.get("spearman_p", math.nan))

    fig, ax = plt.subplots(
        figsize=(max(8, len(clin_list) * 0.9), max(3, len(mech_list) * 0.55))
    )
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(clin_list)))
    ax.set_xticklabels(clin_list, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(mech_list)))
    ax.set_yticklabels(mech_list)
    for i in range(len(mech_list)):
        for j in range(len(clin_list)):
            rho = mat[i, j]
            p = p_mat[i, j]
            if math.isnan(rho):
                continue
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.text(j, i, f"{rho:.2f}{star}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman rho")
    ax.set_title("Mechanism vs clinical (Spearman)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def build_phaseB_participant_rows(
    participants: Sequence[ParticipantRoutes],
    uniqueness_metrics: dict[str, dict[str, float]],
    position_uniqueness_metrics: dict[str, dict[str, float]],
    motor_fingerprints: dict[str, dict[str, float]],
    perseveration: dict[str, dict[str, float]],
    temporal: dict[str, dict[str, float]],
    deliberation: dict[str, dict[str, float]],
    roaming: dict[str, dict[str, float]],
    flexibility: dict[str, dict[str, float]],
    submovements: dict[str, dict[str, float]],
    within_subject_fit: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant in participants:
        pid = participant.participant_id
        row: dict[str, object] = {
            "group": participant.group,
            "participant_id": pid,
            "source_id": participant.source_id,
            "source_platform": participant.source_platform,
            "raw_source": participant.metadata.get("raw_source", ""),
        }
        row.update(motor_fingerprints.get(pid, {}))
        row.update(uniqueness_metrics.get(pid, {}))
        row.update(position_uniqueness_metrics.get(pid, {}))
        row.update(perseveration.get(pid, {}))
        row.update(temporal.get(pid, {}))
        row.update(deliberation.get(pid, {}))
        row.update(roaming.get(pid, {}))
        row.update(flexibility.get(pid, {}))
        row.update(submovements.get(pid, {}))
        if within_subject_fit is not None:
            row.update(within_subject_fit.get(pid, {}))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    parser.add_argument("--clinical-csv", type=Path, default=Path("CHANSS_clinical_demog.csv"))
    parser.add_argument(
        "--clinical-id-corrections-csv",
        type=Path,
        default=Path("analysis_outputs/ogt_patients_missing_clinical_data_corrected.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/phaseB"))
    parser.add_argument("--np-points", type=int, default=200)
    parser.add_argument("--smooth-window", type=int, default=40)
    parser.add_argument("--velocity-downsample", type=int, default=10)
    parser.add_argument(
        "--distance-mode",
        choices=["full", "position"],
        default="full",
        help=(
            "Feature space used for perseveration, temporal slope, and "
            "strategy clustering. 'full' = position+velocity+acceleration "
            "(matches the headline Ang distance); 'position' = position-only."
        ),
    )
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--grid-x-cells", type=int, default=DEFAULT_GRID_X_CELLS)
    parser.add_argument("--grid-y-cells", type=int, default=DEFAULT_GRID_Y_CELLS)
    parser.add_argument(
        "--residualiser-columns",
        nargs="+",
        default=DEFAULT_MOTOR_RESIDUALISER_COLUMNS,
    )
    parser.add_argument(
        "--residualiser-group",
        choices=["patient", "control", "pooled"],
        default="patient",
    )
    parser.add_argument(
        "--mechanism-columns",
        nargs="+",
        default=DEFAULT_MECHANISM_COLUMNS,
        help="Which mechanism scores go into the variance-partitioning step.",
    )
    parser.add_argument(
        "--shapley-target",
        default="test_uniqueness_residualised",
        help="Target of the Shapley-R^2 decomposition and LOO predictive fit.",
    )
    parser.add_argument(
        "--within-fit-min-paths",
        type=int,
        default=6,
        help=(
            "Minimum usable paths required before a within-subject path-level "
            "regression is fit for that participant."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    controls = load_controls(args.control_csv, args.control_dir)
    patients = load_patients(args.patient_octal_dir, args.patient_jatos_dirs)
    participants: list[ParticipantRoutes] = list(controls) + list(patients)
    print(f"Loaded {len(controls)} controls and {len(patients)} patients.")

    # Shared feature cache.
    feature_cache = build_feature_cache(
        participants,
        np_points=args.np_points,
        smooth_window=args.smooth_window,
        velocity_downsample=args.velocity_downsample,
        distance_mode=args.distance_mode,
    )

    # Motor fingerprint and the two uniqueness variants from Phase A.
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
    position_uniqueness = compute_position_only_uniqueness_metrics(
        participants, np_points=args.np_points
    )

    # Mechanism computations.
    perseveration = compute_perseveration(feature_cache)
    temporal = compute_temporal_uniqueness_slope(feature_cache)
    deliberation = compute_deliberation(participants, feature_cache)
    roaming = compute_roaming_entropy(
        participants,
        grid_x_cells=args.grid_x_cells,
        grid_y_cells=args.grid_y_cells,
    )
    flexibility, cluster_info = compute_strategy_flexibility(
        feature_cache, n_clusters=args.n_clusters
    )
    submovements = compute_submovements(participants, smooth_window=args.smooth_window)

    # Within-subject path-level fit: per-participant R^2 for the three
    # trial-level mechanisms.
    within_fit = compute_within_subject_path_fit(
        participants,
        feature_cache,
        smooth_window=args.smooth_window,
        min_paths=args.within_fit_min_paths,
    )

    # Participant-level rows.
    participant_rows = build_phaseB_participant_rows(
        participants,
        uniqueness,
        position_uniqueness,
        motor_fp,
        perseveration,
        temporal,
        deliberation,
        roaming,
        flexibility,
        submovements,
        within_subject_fit=within_fit,
    )

    # Residualise raw uniqueness on motor fingerprint (same logic as Phase A).
    group_filter = None if args.residualiser_group == "pooled" else args.residualiser_group
    residuals, beta, n_fit = residualise_column(
        participant_rows,
        target="test_uniqueness_mean",
        predictors=args.residualiser_columns,
        group_filter=group_filter,
    )
    for row, r in zip(participant_rows, residuals):
        row["test_uniqueness_residualised"] = r

    # Group comparisons on every mechanism + supporting measure.
    mechanism_report_columns = [
        "perseveration_ratio",
        "perseveration_shuffle_z",
        "temporal_prefix_uniqueness_spearman_rho",
        "temporal_prefix_uniqueness_log_slope",
        "temporal_prefix_u_halves_ratio",
        "deliberation_mean_ms",
        "deliberation_cv",
        "deliberation_long_proportion",
        "deliberation_pause_predicts_uniqueness_rho",
        "roaming_entropy_normalised",
        "spatial_coverage_fraction",
        "flexibility_clusters_used",
        "flexibility_cluster_entropy_normalised",
        "submovements_mean_per_path",
        "within_fit_r_squared",
        "within_fit_adjusted_r_squared",
        "within_fit_beta_consec_distance",
        "within_fit_beta_log_pause",
        "within_fit_beta_submovements",
        "test_uniqueness_mean",
        "test_uniqueness_residualised",
    ]
    group_rows = compare_group_rows(participant_rows, mechanism_report_columns)

    # Clinical correlations.
    clinical_correlation_rows: list[dict[str, object]] = []
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
        clinical_correlation_rows = correlate_columns(
            matched_rows,
            columns_x=mechanism_report_columns,
            columns_y=CLINICAL_CORRELATION_FIELDS,
        )

    # Mechanism ↔ uniqueness correlations (patient-only by default).
    patient_rows = [r for r in participant_rows if r.get("group") == "patient"]
    uniqueness_correlation_rows = correlate_columns(
        patient_rows,
        columns_x=mechanism_report_columns,
        columns_y=["test_uniqueness_mean", "test_uniqueness_residualised"],
    )

    # Phase-C preview: Shapley decomposition of R^2.
    shapley_rows = shapley_r_squared(
        participant_rows,
        target=args.shapley_target,
        predictors=args.mechanism_columns,
        group_filter="patient",
    )

    # Leave-one-out predictive R^2 of the full mechanism model at subject level.
    loo_per_rows, loo_summary = loo_predictive_r_squared(
        participant_rows,
        target=args.shapley_target,
        predictors=args.mechanism_columns,
        group_filter="patient",
    )

    # Merge per-participant LOO outputs (predicted / residual) back into the
    # headline participant_rows so the per-person fit quality is right there
    # next to the mechanisms.
    loo_by_pid = {row["participant_id"]: row for row in loo_per_rows}
    for row in participant_rows:
        pid = row.get("participant_id", "")
        loo_row = loo_by_pid.get(pid)
        if loo_row is None:
            row["loo_predicted_uniqueness"] = math.nan
            row["loo_residual"] = math.nan
            row["loo_squared_standardised_residual"] = math.nan
        else:
            row["loo_predicted_uniqueness"] = loo_row["predicted"]
            row["loo_residual"] = loo_row["residual"]
            row["loo_squared_standardised_residual"] = loo_row[
                "squared_standardised_residual"
            ]

    # --- Write outputs ------------------------------------------------------
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    write_csv(
        output_dir / "phaseB_participant_metrics.csv",
        participant_rows,
        list(participant_rows[0].keys()) if participant_rows else [],
    )
    write_csv(
        output_dir / "phaseB_mechanism_group_comparison.csv",
        group_rows,
        list(group_rows[0].keys()) if group_rows else [],
    )
    if clinical_correlation_rows:
        write_csv(
            output_dir / "phaseB_mechanism_clinical_correlations.csv",
            clinical_correlation_rows,
            list(clinical_correlation_rows[0].keys()),
        )
    write_csv(
        output_dir / "phaseB_mechanism_uniqueness_correlations.csv",
        uniqueness_correlation_rows,
        list(uniqueness_correlation_rows[0].keys())
        if uniqueness_correlation_rows
        else [],
    )
    write_csv(
        output_dir / "phaseB_variance_partitioning.csv",
        shapley_rows,
        list(shapley_rows[0].keys()) if shapley_rows else ["predictor", "shapley_r_squared", "total_r_squared", "n"],
    )

    # Within-subject path-level fit CSV: per-participant R^2 for the three
    # trial-level mechanisms predicting prefix uniqueness, plus standardised
    # betas and marginal Spearman rhos.
    within_fit_rows = [
        {
            "participant_id": pid,
            "group": next(
                (p.group for p in participants if p.participant_id == pid), ""
            ),
            **(within_fit.get(pid) or {}),
        }
        for pid in within_fit
    ]
    if within_fit_rows:
        keys = list(within_fit_rows[0].keys())
    else:
        keys = [
            "participant_id",
            "group",
            "within_fit_n_paths",
            "within_fit_r_squared",
            "within_fit_adjusted_r_squared",
            "within_fit_beta_consec_distance",
            "within_fit_beta_log_pause",
            "within_fit_beta_submovements",
            "within_fit_rho_consec_distance",
            "within_fit_rho_log_pause",
            "within_fit_rho_submovements",
        ]
    write_csv(output_dir / "phaseB_within_subject_fit.csv", within_fit_rows, keys)

    # Leave-one-out predictive fit: one row per held-out patient plus a
    # cohort-level summary at the top.
    loo_header_row: dict[str, object] = {
        "participant_id": "__SUMMARY__",
        "group": "patient",
        "target": loo_summary.get("target", args.shapley_target),
        "observed": math.nan,
        "predicted": math.nan,
        "residual": math.nan,
        "abs_residual": math.nan,
        "squared_standardised_residual": math.nan,
        "loo_r_squared_cohort": loo_summary.get("loo_r_squared", math.nan),
        "loo_rmse_cohort": loo_summary.get("loo_rmse", math.nan),
        "loo_mean_abs_error_cohort": loo_summary.get("loo_mean_abs_error", math.nan),
        "predicted_observed_spearman_rho_cohort": loo_summary.get(
            "predicted_observed_spearman_rho", math.nan
        ),
        "predicted_observed_spearman_p_cohort": loo_summary.get(
            "predicted_observed_spearman_p", math.nan
        ),
        "n_cohort": loo_summary.get("n", 0),
        "k_predictors": loo_summary.get("k_predictors", len(args.mechanism_columns)),
    }
    cohort_only_keys = [
        "loo_r_squared_cohort",
        "loo_rmse_cohort",
        "loo_mean_abs_error_cohort",
        "predicted_observed_spearman_rho_cohort",
        "predicted_observed_spearman_p_cohort",
        "n_cohort",
        "k_predictors",
    ]
    loo_rows_out: list[dict[str, object]] = [loo_header_row]
    for r in loo_per_rows:
        padded = dict(r)
        for k in cohort_only_keys:
            padded[k] = math.nan if k != "k_predictors" else loo_summary.get("k_predictors", len(args.mechanism_columns))
        loo_rows_out.append(padded)
    loo_keys = list(loo_header_row.keys())
    write_csv(output_dir / "phaseB_loo_predictive_fit.csv", loo_rows_out, loo_keys)

    # Strategy cluster summary (centroid sizes, per-cluster path counts).
    cluster_summary: list[dict[str, object]] = []
    assignments = cluster_info.get("assignments", np.empty((0,), dtype=int))
    n_clusters = int(cluster_info.get("n_clusters", 0))
    if assignments.size:
        for k in range(n_clusters):
            cluster_summary.append(
                {
                    "cluster": k,
                    "n_paths": int(np.count_nonzero(assignments == k)),
                }
            )
    write_csv(
        output_dir / "phaseB_strategy_cluster_sizes.csv",
        cluster_summary,
        ["cluster", "n_paths"],
    )

    if missing_rows:
        write_csv(
            output_dir / "phaseB_patients_missing_clinical_data.csv",
            missing_rows,
            list(missing_rows[0].keys()),
        )

    # Figures.
    save_mechanism_rainclouds(
        figures_dir / "phaseB_mechanism_rainclouds.png",
        participant_rows,
        mechanism_report_columns,
    )
    save_strategy_archetypes(
        figures_dir / "phaseB_strategy_archetypes.png",
        cluster_info,
        feature_cache,
        participants,
        np_points=args.np_points,
    )
    if clinical_correlation_rows:
        save_mechanism_clinical_heatmap(
            figures_dir / "phaseB_mechanism_clinical_heatmap.png",
            clinical_correlation_rows,
            mechanism_columns=mechanism_report_columns,
            clinical_columns=CLINICAL_CORRELATION_FIELDS,
        )

    print(f"Phase-B outputs written to {output_dir}")
    print(f"Shapley decomposition (target = {args.shapley_target}):")
    for row in shapley_rows:
        print(
            f"  {row['predictor']}: shapley R^2 = "
            f"{row.get('shapley_r_squared', math.nan):.4f} "
            f"(total = {row.get('total_r_squared', math.nan):.4f}, n = {row.get('n', 0)})"
        )
    print(
        "LOO predictive fit "
        f"(target = {args.shapley_target}, K = {loo_summary.get('k_predictors', 0)} "
        f"predictors, n = {loo_summary.get('n', 0)}): "
        f"R^2 = {loo_summary.get('loo_r_squared', math.nan):.4f}, "
        f"RMSE = {loo_summary.get('loo_rmse', math.nan):.4f}, "
        f"MAE = {loo_summary.get('loo_mean_abs_error', math.nan):.4f}, "
        f"pred-obs Spearman rho = "
        f"{loo_summary.get('predicted_observed_spearman_rho', math.nan):.3f}"
    )
    within_r2 = [
        v["within_fit_r_squared"]
        for v in within_fit.values()
        if v.get("within_fit_r_squared") is not None
        and isinstance(v.get("within_fit_r_squared"), float)
        and not math.isnan(v["within_fit_r_squared"])
    ]
    if within_r2:
        print(
            "Within-subject path-level fit (3 trial-level mechanisms -> "
            "prefix uniqueness): "
            f"median R^2 across participants = {float(np.median(within_r2)):.3f} "
            f"(IQR {float(np.quantile(within_r2, 0.25)):.3f}-"
            f"{float(np.quantile(within_r2, 0.75)):.3f}, "
            f"n participants with fit = {len(within_r2)})"
        )


if __name__ == "__main__":
    main()
