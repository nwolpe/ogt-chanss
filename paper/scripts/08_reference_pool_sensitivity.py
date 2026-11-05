#!/usr/bin/env python3
"""Revision analysis: is the patient-control uniqueness difference an artefact of the reference pool?

Route uniqueness (Ang et al., 2018) is each route's nearest-neighbour distance to every other test route
in the pooled dataset (100 controls + 150 patients). Because the pool contains 1.5 times as many patients as
controls, and patients' routes may resemble one another (e.g. by site or device), a patient route could have
a closer neighbour simply because there are more similar routes in the pool. This script recomputes
participant-level uniqueness under alternative reference sets that remove this possibility:

  pooled_original      : as in the paper (all other routes, including the participant's own other routes)
  leave_participant_out: nearest route drawn by a *different* participant (between-person distinctiveness)
  control_reference    : nearest route among control routes only (own routes excluded) -> common yardstick
  balanced_pool        : 100 controls + 100 randomly sampled patients per resample (50 resamples), averaged
  within_person_nn     : nearest neighbour among the participant's own routes only (reference-free)
  within_person_spread : mean pairwise distance among the participant's own routes (reference-free)

Run from the OGT repository root:
  python3 paper/scripts/08_reference_pool_sensitivity.py
Writes paper/outputs/revision/reference_pool_participant_uniqueness.csv
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "paper" / "outputs" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
N_BALANCED = 50
RNG = np.random.default_rng(20261017)

spec = importlib.util.spec_from_file_location("ru", ROOT / "paper" / "scripts" / "00_recompute_paper_route_uniqueness.py")
ru = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ru)


def nn_distances(Q: np.ndarray, QF: np.ndarray, R: np.ndarray, exclude: np.ndarray | None, block: int = 256) -> np.ndarray:
    """Nearest-neighbour distance from each query route (and its mirror) to reference routes.

    exclude: boolean matrix builder -> function(start, end) returning mask (block x n_ref) of disallowed pairs.
    """
    r_norm = np.einsum("ij,ij->i", R, R)
    out = np.full(Q.shape[0], np.inf, dtype=np.float64)
    for s in range(0, Q.shape[0], block):
        e = min(s + block, Q.shape[0])
        mask = exclude(s, e) if exclude is not None else None
        for q in (Q[s:e], QF[s:e]):
            qn = np.einsum("ij,ij->i", q, q)
            d2 = qn[:, None] + r_norm[None, :] - 2.0 * (q @ R.T)
            np.maximum(d2, 0, out=d2)
            if mask is not None:
                d2[mask] = np.inf
            out[s:e] = np.minimum(out[s:e], np.sqrt(d2.min(axis=1)))
    return out


def main() -> None:
    ns = ru.load_analyze_namespace()
    ns["select_control_rows"] = ru.select_control_rows_include_second_only
    controls = ns["load_controls"](ru.CONTROL_CSV, ru.CONTROL_DIR)
    patients = ns["load_patients"](ru.PATIENT_OCTAL_DIR, ru.PATIENT_JATOS_DIRS)
    participants = list(controls) + list(patients)

    recs, feats, flips = [], [], []
    for p_idx, p in enumerate(participants):
        for r_idx, route in enumerate(p.test_routes):
            pair = ru.route_feature_pair(route)
            if pair is None:
                continue
            recs.append({"p_idx": p_idx, "group": p.group, "participant_id": p.participant_id,
                         "source_id": p.source_id, "source_platform": p.source_platform})
            feats.append(pair[0]); flips.append(pair[1])
    routes = pd.DataFrame(recs)
    X = np.vstack(feats).astype(np.float64)
    XF = np.vstack(flips).astype(np.float64)
    pid = routes["p_idx"].to_numpy()
    is_ctrl = (routes["group"] == "control").to_numpy()
    n = len(routes)
    print(f"{n} routes from {len(participants)} participants")

    # 1. pooled original (exclude self-route only)
    idx_all = np.arange(n)
    routes["pooled_original"] = nn_distances(X, XF, X, lambda s, e: idx_all[s:e, None] == idx_all[None, :])
    # 2. leave participant out
    routes["leave_participant_out"] = nn_distances(X, XF, X, lambda s, e: pid[s:e, None] == pid[None, :])
    # 3. control reference (own participant's routes excluded)
    ctrl_idx = np.where(is_ctrl)[0]
    pid_c = pid[ctrl_idx]
    routes["control_reference"] = nn_distances(X, XF, X[ctrl_idx], lambda s, e: pid[s:e, None] == pid_c[None, :])
    # 4. within-person nearest neighbour and spread
    wnn = np.full(n, np.nan); spread = {}
    for p_idx, g in routes.groupby("p_idx").groups.items():
        g = np.asarray(g)
        if len(g) < 2:
            continue
        A, AF = X[g], XF[g]
        d_plain = np.sqrt(np.maximum(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1), 0))
        d_flip = np.sqrt(np.maximum(((AF[:, None, :] - A[None, :, :]) ** 2).sum(-1), 0))
        d = np.minimum(d_plain, d_flip)
        np.fill_diagonal(d, np.inf)
        wnn[g] = d.min(axis=1)
        iu = np.triu_indices(len(g), 1)
        spread[p_idx] = float(np.mean(d[iu]))
    routes["within_person_nn"] = wnn
    # 5. balanced pools
    patient_ids = np.array(sorted(set(pid[~is_ctrl])))
    acc = np.zeros(n); cnt = np.zeros(n)
    for b in range(N_BALANCED):
        chosen = RNG.choice(patient_ids, size=100, replace=False)
        keep = np.where(is_ctrl | np.isin(pid, chosen))[0]
        Xk, XFk = X[keep], XF[keep]
        loc = np.arange(len(keep))
        d = nn_distances(Xk, XFk, Xk, lambda s, e: loc[s:e, None] == loc[None, :])
        acc[keep] += d; cnt[keep] += 1
        if b % 10 == 0:
            print(f"  balanced resample {b + 1}/{N_BALANCED}")
    # per-route mean over resamples in which the route was included
    routes["balanced_pool"] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    measures = ["pooled_original", "leave_participant_out", "control_reference", "balanced_pool", "within_person_nn"]
    part = routes.groupby(["p_idx", "group", "participant_id", "source_id", "source_platform"])[measures].mean().reset_index()
    part["within_person_spread"] = part["p_idx"].map(spread)
    part["n_routes"] = part["p_idx"].map(routes.groupby("p_idx").size())
    part.drop(columns="p_idx").to_csv(OUT / "reference_pool_participant_uniqueness.csv", index=False)
    print(part.groupby("group")[measures + ["within_person_spread"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
