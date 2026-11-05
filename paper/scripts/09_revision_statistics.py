#!/usr/bin/env python3
"""Revision statistics (response to reviewers, October 2026).

Run from the OGT repository root after scripts 00-04, 07 (CCA) and 08 (reference pools):
  python3 paper/scripts/09_revision_statistics.py

All AMI scores use the CHANSS database coding (Completely true = 0 ... Completely untrue = 4), which is already
apathy-scored (higher = more apathy); no item reversal is applied.

Outputs (paper/outputs/revision/):
  R1_fluency_equivalence.csv           route count: effect size, CI, TOST equivalence, by site
  R2_uniqueness_robustness.csv         group difference in uniqueness across sites, coordinate frames, age/sex
                                       strata, reference pools; Hedges g with bootstrap CI
  R3_patient_glm.csv                   within-patient models: age, sex, site, motor, medication, illness stage
  R4_clinical_correlations.csv         patient uniqueness vs clinical/cognitive measures: unadjusted, site-
                                       adjusted (partial Spearman), excluding China; BH-FDR within each family
  R5_verbal_fluency_specificity.csv    bootstrap difference between VF correlation and other BACS domains
  R6_clinical_multivariable.csv        uniqueness ~ CDS + AMI behavioural + BNSS MAP (+ site)
  R7_shapley_by_group.csv              Shapley R2 of the four components, combined / controls / patients
  R8_cluster_stability.csv             k-means (k = 2..6) silhouettes and bootstrap Jaccard stability, site mix
  R9_site_descriptives.csv             patient characteristics by site
  R10_scale_validity.csv               convergent validity of AMI (as scored) against AES-S, BNSS, CDS, PSP
  revision_summary.md                  human-readable summary of the above
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(".")
OUT = ROOT / "paper" / "outputs" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261017)
N_BOOT = 5000

# ---------------------------------------------------------------- data assembly
s2 = pd.read_csv(ROOT / "paper/outputs/section2/section2_participant_metrics.csv", low_memory=False)
mech = pd.read_csv(ROOT / "analysis_outputs/phaseB_exploration_dynamics/phaseB_exploration_dynamics_participant_metrics.csv",
                   low_memory=False)
mech = mech[["group", "participant_id", "source_id", "roaming_entropy_normalised", "submovements_mean_per_path",
             "perseveration_ratio", "temporal_prefix_uniqueness_spearman_rho"]]
calib = pd.read_csv(ROOT / "paper/outputs/shared/paper_coordinate_calibration.csv")
calib["anisotropy"] = calib["calibration_y_scale"] / calib["calibration_x_scale"]
rp = pd.read_csv(OUT / "reference_pool_participant_uniqueness.csv")
for df in (s2, mech, calib, rp):
    for c in ("group", "participant_id", "source_id"):
        df[c] = df[c].astype(str)
keys = ["group", "participant_id", "source_id"]
d = (s2.merge(mech, on=keys, how="left")
       .merge(calib[keys + ["anisotropy"]], on=keys, how="left")
       .merge(rp.drop(columns=["source_platform"]), on=keys, how="left"))

clin = pd.read_csv(ROOT / "CHANSS_clinical_demog.csv", low_memory=False).copy()
num = lambda s: pd.to_numeric(s, errors="coerce")
clin["ID"] = clin["ID_subject"].astype(str).str.upper().str.strip()
items = lambda pre, idx: pd.concat([num(clin[f"{pre}{i}"]) for i in idx], axis=1)
clin_df = pd.DataFrame({
    "ID": clin["ID"],
    "age_clin": num(clin["Age"]), "sex_clin": num(clin["Gender"]),  # 0 = female, 1 = male
    "Age_FEP": num(clin["Age_FEP"]),
    "MAP": items("BNSS_", [1, 2, 3, 5, 6, 7, 8]).sum(axis=1, min_count=1),
    "EXP": items("BNSS_", [9, 10, 11, 12, 13]).sum(axis=1, min_count=1),
    "BNSS_anhedonia": items("BNSS_", [1, 2, 3]).sum(axis=1, min_count=1),
    "BNSS_asociality": items("BNSS_", [5, 6]).sum(axis=1, min_count=1),
    "BNSS_avolition": items("BNSS_", [7, 8]).sum(axis=1, min_count=1),
    "BNSS_blunted": items("BNSS_", [9, 10, 11]).sum(axis=1, min_count=1),
    "BNSS_alogia": items("BNSS_", [12, 13]).sum(axis=1, min_count=1),
    "AMI_beh": items("AMI_", [5, 9, 10, 11, 12, 15]).mean(axis=1),
    "AMI_soc": items("AMI_", [2, 3, 4, 8, 14, 17]).mean(axis=1),
    "AMI_emo": items("AMI_", [1, 6, 7, 13, 16, 18]).mean(axis=1),
    "CDS": items("CDS_", range(1, 10)).sum(axis=1, min_count=1),
    "PANSS_pos": pd.concat([num(clin[c]) for c in ["PANSS_p1", "PANSS_p3", "PANSS_pg9", "PANSS_pg10"]], axis=1).sum(axis=1, min_count=1),
    "PSP": num(clin["PSP_score"]),
    "VF": num(clin["BACS_VF_t"]), "VM": num(clin["BACS_VM_t"]), "DS": num(clin["BACS_DS_t"]),
    "TMT": num(clin["BACS_TMT_t"]), "SC": num(clin["BACS_SC_t"]), "ToL": num(clin["BACS_ToL_t"]),
    "BACS_comp": num(clin["BACS_comp_t"]),
    "MINI": clin["MINI"].astype(str),
    "AD_any": clin["AD_1"].notna() & (clin["AD_1"].astype(str).str.strip() != ""),
})
# AES (self-report; database coding 1 = not at all ... 4 = a lot); standard scoring reverses the 15 positively
# worded items so that higher = more apathy (items 6, 10, 11 are negatively worded).
aes = items("AES", range(1, 19))
aes.columns = range(1, 19)
for i in range(1, 19):
    if i not in (6, 10, 11):
        aes[i] = 5 - aes[i]
clin_df["AES_S"] = aes.sum(axis=1, min_count=18)
clin_df = clin_df.drop_duplicates("ID")
med = pd.read_excel(ROOT / "paper/CHANSS_clinical_demog_filled_APLoad.xlsx")
med = pd.DataFrame({"ID": med["ID_subject"].astype(str).str.upper().str.strip(),
                    "AP_BNFmax": num(med["AP_BNFmax"]),
                    "clozapine": med["AP_1"].astype(str).str.lower().str.contains("cloz").astype(float)
                                 + med["AP_2"].astype(str).str.lower().str.contains("cloz").astype(float) > 0,
                    "polypharmacy": med["AP_2"].notna() & ~med["AP_2"].astype(str).str.lower().str.contains("venfl|estazolam")}
                   ).drop_duplicates("ID")
corr = pd.read_csv(ROOT / "analysis_outputs/ogt_patients_missing_clinical_data_corrected.csv")
cmap = dict(zip(corr["participant_id"].astype(str).str.upper().str.strip(), corr["participant_id_corrected"].astype(str).str.upper().str.strip()))
d["cid"] = [cmap.get(p.upper().strip(), p.upper().strip()) for p in d["participant_id"]]
d = d.merge(clin_df, left_on="cid", right_on="ID", how="left").merge(med, on="ID", how="left")
pat = d["group"].eq("patient")
d["site"] = np.select([~pat, d["source_platform"].eq("OGT_Mandarin") | d["cid"].str.startswith("CH"),
                       d["source_platform"].eq("OGT_Spanish")], ["Control", "China", "Spain"], "UK")
d["age_all"] = np.where(pat, d["age_clin"], num(d["age"]))
sex_ctrl = d["sex"].astype(str).str.lower().map({"female": 0, "f": 0, "male": 1, "m": 1, "0": 0, "1": 1})
d["male"] = np.where(pat, d["sex_clin"], sex_ctrl)
d["illness_duration"] = d["age_clin"] - d["Age_FEP"]
d["early_psychosis"] = d["illness_duration"] <= 5
d["same_frame"] = d["anisotropy"] < 1.3
d["is_patient"] = pat.astype(int)
d.to_csv(OUT / "revision_analysis_dataset.csv", index=False)
P = d[pat].copy()
C = d[~pat].copy()

U_RAW, U_RES, U_PRES = "test_uniqueness_mean", "test_uniqueness_motor_residualised", "test_uniqueness_patient_motor_residualised"
MECH = ["roaming_entropy_normalised", "submovements_mean_per_path", "perseveration_ratio", "temporal_prefix_uniqueness_spearman_rho"]
MECH_LAB = dict(zip(MECH, ["Roaming entropy", "Option elaboration", "Perseveration ratio", "Novelty maintenance"]))


# ---------------------------------------------------------------- helpers
def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)
    return j * (a.mean() - b.mean()) / sp


def boot_g(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    g = [hedges_g(RNG.choice(a, len(a)), RNG.choice(b, len(b))) for _ in range(n)]
    return np.percentile(g, [2.5, 97.5])


def compare(label, ctrl, patv, metric):
    a, b = ctrl[metric].to_numpy(float), patv[metric].to_numpy(float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b)
    lo, hi = boot_g(a, b)
    return {"comparison": label, "metric": metric, "n_control": len(a), "n_patient": len(b),
            "mean_control": a.mean(), "mean_patient": b.mean(), "median_control": np.median(a), "median_patient": np.median(b),
            "ratio_patient_control": b.mean() / a.mean() if a.mean() > 0 else np.nan,
            "welch_t": t.statistic, "welch_df": t.df, "p": t.pvalue, "mannwhitney_p": u.pvalue,
            "hedges_g": hedges_g(a, b), "g_ci_low": lo, "g_ci_high": hi}


def rank(x):
    return stats.rankdata(x)


def partial_spearman(df, x, y, covars):
    sub = df[[x, y] + covars].dropna()
    if covars:
        Z = pd.get_dummies(sub[covars], drop_first=True, dtype=float)
        Z = np.column_stack([np.ones(len(sub)), Z.to_numpy(float)]) if Z.shape[1] else np.ones((len(sub), 1))
    else:
        Z = np.ones((len(sub), 1))
    rx, ry = rank(sub[x]), rank(sub[y])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    k = Z.shape[1] - 1
    dfree = len(sub) - 2 - k
    tval = r * math.sqrt(dfree / max(1e-12, 1 - r * r))
    return r, 2 * stats.t.sf(abs(tval), dfree), len(sub)


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        q[ok] = stats.false_discovery_control(p[ok])
    return q


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dfree = len(y) - X1.shape[1]
    s2v = resid @ resid / dfree
    cov = s2v * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, tval, 2 * stats.t.sf(np.abs(tval), dfree), r2, dfree


def glm_table(name, df, yv, xs, standardise=False):
    sub = df[[yv] + xs].dropna().copy()
    X = pd.get_dummies(sub[xs], drop_first=True, dtype=float)
    y = sub[yv].to_numpy(float)
    if standardise:
        y = (y - y.mean()) / y.std(ddof=1)
        for c in X.columns:
            if X[c].nunique() > 2:
                X[c] = (X[c] - X[c].mean()) / X[c].std(ddof=1)
    beta, se, tval, p, r2, dfree = ols(y, X.to_numpy(float))
    rows = [{"model": name, "term": t, "estimate": b, "se": s, "t": tv, "p": pv, "n": len(sub), "r2": r2, "df": dfree}
            for t, b, s, tv, pv in zip(["(Intercept)"] + list(X.columns), beta, se, tval, p)]
    return rows


# ---------------------------------------------------------------- R1 fluency
rows = []
sd_pooled = math.sqrt((C["test_route_count"].var(ddof=1) * (len(C) - 1) + P["test_route_count"].var(ddof=1) * (len(P) - 1)) / (len(C) + len(P) - 2))
for label, sub in [("All patients", P)] + [(s, P[P.site == s]) for s in ["UK", "Spain", "China"]] + [("Excluding China", P[P.site != "China"])]:
    r = compare(label, C, sub, "test_route_count")
    a, b = C["test_route_count"].dropna().to_numpy(), sub["test_route_count"].dropna().to_numpy()
    diff = b.mean() - a.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dfw = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    for d_bound in (0.3, 0.4, 0.5):
        bound = d_bound * sd_pooled
        p_low = stats.t.sf((diff + bound) / se, dfw)
        p_high = stats.t.cdf((diff - bound) / se, dfw)
        r[f"tost_p_d{d_bound}"] = max(p_low, p_high)
    r["mean_diff"] = diff
    r["diff_ci90_low"], r["diff_ci90_high"] = diff - stats.t.ppf(.95, dfw) * se, diff + stats.t.ppf(.95, dfw) * se
    r["diff_ci95_low"], r["diff_ci95_high"] = diff - stats.t.ppf(.975, dfw) * se, diff + stats.t.ppf(.975, dfw) * se
    rows.append(r)
R1 = pd.DataFrame(rows)
R1.to_csv(OUT / "R1_fluency_equivalence.csv", index=False)

# ---------------------------------------------------------------- R2 uniqueness robustness
ctrl_age = C["age_all"].dropna()
age_lo, age_hi = ctrl_age.min(), ctrl_age.max()
rows = []
rows.append(compare("All patients", C, P, U_RAW))
rows.append(compare("All patients (motor-residualised)", C, P, U_RES))
rows.append(compare("All patients (log uniqueness)", C, P, "test_log_uniqueness_mean"))
for s in ["UK", "Spain", "China"]:
    rows.append(compare(f"Site: {s}", C, P[P.site == s], U_RAW))
rows.append(compare("Excluding China", C, P[P.site != "China"], U_RAW))
rows.append(compare("Same coordinate frame as controls", C, P[P.same_frame], U_RAW))
rows.append(compare("Rescaled coordinate frame", C, P[~P.same_frame], U_RAW))
rows.append(compare(f"Age-overlap: patients aged {age_lo:.0f}-{age_hi:.0f}", C[C.age_all.notna()], P[P.age_all.between(age_lo, age_hi)], U_RAW))
rows.append(compare("Female participants only", C[C.male == 0], P[P.male == 0], U_RAW))
rows.append(compare("Male participants only", C[C.male == 1], P[P.male == 1], U_RAW))
for m, lab in [("leave_participant_out", "Reference: other participants only"), ("control_reference", "Reference: control routes only"),
               ("balanced_pool", "Reference: balanced 100 v 100 pool"), ("within_person_nn", "Within-person nearest neighbour"),
               ("within_person_spread", "Within-person mean pairwise distance")]:
    rows.append(compare(lab, C, P, m))
R2 = pd.DataFrame(rows)
R2.to_csv(OUT / "R2_uniqueness_robustness.csv", index=False)

# full-sample GLMs with age/sex in age-overlap subsample
glm_rows = []
both = d[d.age_all.notna() & d.male.notna()].copy()
glm_rows += glm_table("Full sample: uniqueness ~ motor + age + sex + group", both, U_RAW, ["control_mean_speed_px_per_s", "age_all", "male", "is_patient"])
ov = both[both.age_all.between(age_lo, age_hi)]
glm_rows += glm_table(f"Age-overlap sample ({age_lo:.0f}-{age_hi:.0f} y): uniqueness ~ motor + age + sex + group", ov, U_RAW,
                      ["control_mean_speed_px_per_s", "age_all", "male", "is_patient"])

# ---------------------------------------------------------------- R3 within-patient GLMs
glm_rows += glm_table("Patients: uniqueness ~ motor + age + sex + site", P, U_RAW, ["control_mean_speed_px_per_s", "age_all", "male", "site"])
glm_rows += glm_table("Patients: uniqueness ~ motor + age + sex + site + AP burden + clozapine + polypharmacy + antidepressant",
                      P.assign(clozapine=P.clozapine.astype(float), polypharmacy=P.polypharmacy.astype(float), AD_any=P.AD_any.astype(float)),
                      U_RAW, ["control_mean_speed_px_per_s", "age_all", "male", "site", "AP_BNFmax", "clozapine", "polypharmacy", "AD_any"])
glm_rows += glm_table("Patients: uniqueness ~ motor + site + illness duration", P, U_RAW, ["control_mean_speed_px_per_s", "site", "illness_duration"])
R3 = pd.DataFrame(glm_rows)
R3.to_csv(OUT / "R3_glms.csv", index=False)

# ---------------------------------------------------------------- R4 clinical correlations
CLIN = [("MAP", "BNSS motivation and pleasure"), ("EXP", "BNSS expressive deficit"), ("AMI_beh", "AMI behavioural"),
        ("AMI_soc", "AMI social"), ("AMI_emo", "AMI emotional"), ("AES_S", "AES (self-report)"), ("CDS", "CDS total"),
        ("PANSS_pos", "PANSS positive"), ("PSP", "PSP"), ("VM", "BACS verbal memory"), ("DS", "BACS digit sequencing"),
        ("TMT", "BACS token motor"), ("VF", "BACS verbal fluency"), ("SC", "BACS symbol coding"), ("ToL", "BACS Tower of London")]
rows = []
for xvar, xlab in [(U_PRES, "Motor-residualised uniqueness"), ("control_mean_speed_px_per_s", "Control-condition motor speed")]:
    for analysis, sub, cov in [("Unadjusted", P, []), ("Site-adjusted", P, ["site"]), ("Excluding China", P[P.site != "China"], []),
                               ("Excluding China, site-adjusted", P[P.site != "China"], ["site"])]:
        block = []
        for v, lab in CLIN:
            r, p, n = partial_spearman(sub, xvar, v, cov)
            ss = sub[[xvar, v]].dropna()
            bs = []
            for _ in range(2000 if analysis == "Unadjusted" else 0):
                i = RNG.integers(0, len(ss), len(ss))
                bs.append(stats.spearmanr(ss[xvar].to_numpy()[i], ss[v].to_numpy()[i])[0])
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
            block.append({"predictor": xlab, "analysis": analysis, "measure": lab, "variable": v, "n": n, "rho": r, "p": p,
                          "ci_low": lo, "ci_high": hi})
        q = bh([b["p"] for b in block])
        for b, qq in zip(block, q):
            b["q_fdr"] = qq
        rows += block
R4 = pd.DataFrame(rows)
R4.to_csv(OUT / "R4_clinical_correlations.csv", index=False)

# ---------------------------------------------------------------- R5 verbal fluency specificity
rows = []
cog = ["VM", "DS", "TMT", "SC", "ToL"]
for analysis, cov in [("Unadjusted", []), ("Site-adjusted", ["site"])]:
    ss = P[[U_PRES, "VF", "site"] + cog].dropna().reset_index(drop=True)
    obs = {v: partial_spearman(ss, U_PRES, v, cov)[0] for v in ["VF"] + cog}
    boots = {v: [] for v in cog}
    for _ in range(2000):
        i = RNG.integers(0, len(ss), len(ss))
        bsamp = ss.iloc[i].reset_index(drop=True)
        rv = partial_spearman(bsamp, U_PRES, "VF", cov)[0]
        for v in cog:
            boots[v].append(rv - partial_spearman(bsamp, U_PRES, v, cov)[0])
    for v in cog:
        b = np.asarray(boots[v])
        rows.append({"analysis": analysis, "comparison": f"VF vs {v}", "n": len(ss), "rho_VF": obs["VF"], "rho_other": obs[v],
                     "diff": obs["VF"] - obs[v], "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5),
                     "p_boot": min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))})
R5 = pd.DataFrame(rows)
R5.to_csv(OUT / "R5_verbal_fluency_specificity.csv", index=False)

# ---------------------------------------------------------------- R6 multivariable clinical model (rank-based)
mv = P[[U_PRES, "CDS", "AMI_beh", "MAP", "site", "VF"]].dropna().copy()
for c in [U_PRES, "CDS", "AMI_beh", "MAP", "VF"]:
    mv[c] = rank(mv[c])
R6 = pd.DataFrame(glm_table("Uniqueness ~ CDS + AMI behavioural + BNSS MAP (ranks, standardised)", mv, U_PRES, ["CDS", "AMI_beh", "MAP"], True)
                  + glm_table("Uniqueness ~ CDS + AMI behavioural + BNSS MAP + site (ranks, standardised)", mv, U_PRES, ["CDS", "AMI_beh", "MAP", "site"], True)
                  + glm_table("Uniqueness ~ CDS + AMI behavioural + BNSS MAP + verbal fluency + site (ranks, standardised)", mv, U_PRES,
                              ["CDS", "AMI_beh", "MAP", "VF", "site"], True))
R6.to_csv(OUT / "R6_clinical_multivariable.csv", index=False)


# ---------------------------------------------------------------- R7 Shapley by group
def shapley(df, yv, xs):
    sub = df[[yv] + xs].dropna()
    y = ((sub[yv] - sub[yv].mean()) / sub[yv].std()).to_numpy()
    Xz = ((sub[xs] - sub[xs].mean()) / sub[xs].std()).to_numpy()

    def r2(cols):
        if not cols:
            return 0.0
        return ols(y, Xz[:, cols])[4]
    k = len(xs)
    out = {}
    for j in range(k):
        others = [i for i in range(k) if i != j]
        val = 0.0
        for size in range(k):
            for S in itertools.combinations(others, size):
                w = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
                val += w * (r2(list(S) + [j]) - r2(list(S)))
        out[xs[j]] = val
    return out, r2(list(range(k))), len(sub)


rows = []
for lab, sub, yv in [("Combined sample", d, U_RES), ("Controls only", C, U_RES), ("Patients only", P, U_RES),
                     ("Patients only, site-residualised", None, None)]:
    if sub is None:
        sub = P[[U_RES] + MECH + ["site"]].dropna().copy()
        Z = pd.get_dummies(sub["site"], drop_first=True, dtype=float).to_numpy()
        Z = np.column_stack([np.ones(len(sub)), Z])
        for c in [U_RES] + MECH:
            v = sub[c].to_numpy(float)
            sub[c] = v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        yv = U_RES
    sh, tot, n = shapley(sub, yv, MECH)
    for m in MECH:
        rows.append({"sample": lab, "component": MECH_LAB[m], "shapley_r2": sh[m], "total_r2": tot, "n": n})
R7 = pd.DataFrame(rows)
R7.to_csv(OUT / "R7_shapley_by_group.csv", index=False)

# ---------------------------------------------------------------- R8 cluster stability (supplementary)
cl = P[MECH + ["site", "participant_id"]].dropna().reset_index(drop=True)
Xc = ((cl[MECH] - cl[MECH].mean()) / cl[MECH].std()).to_numpy()
rows = []
for k in range(2, 7):
    km = KMeans(k, n_init=100, random_state=1).fit(Xc)
    ref = km.labels_
    jac = np.zeros((300, k))
    for b in range(300):
        i = RNG.integers(0, len(Xc), len(Xc))
        kb = KMeans(k, n_init=20, random_state=b).fit(Xc[i])
        lab_b = kb.predict(Xc)
        for c in range(k):
            A = set(np.where(ref == c)[0])
            jac[b, c] = max(len(A & set(np.where(lab_b == cb)[0])) / len(A | set(np.where(lab_b == cb)[0])) for cb in range(k))
    sizes = np.bincount(ref, minlength=k)
    for c in range(k):
        sitemix = cl.loc[ref == c, "site"].value_counts().to_dict()
        rows.append({"k": k, "silhouette": silhouette_score(Xc, ref), "cluster": c + 1, "size": int(sizes[c]),
                     "jaccard_mean": jac[:, c].mean(), "jaccard_dissolved_pct": 100 * (jac[:, c] < 0.5).mean(),
                     "n_UK": sitemix.get("UK", 0), "n_Spain": sitemix.get("Spain", 0), "n_China": sitemix.get("China", 0)})
R8 = pd.DataFrame(rows)
R8.to_csv(OUT / "R8_cluster_stability.csv", index=False)

# ---------------------------------------------------------------- R9 site descriptives
desc_vars = ["age_all", "male", "illness_duration", "clozapine", "AP_BNFmax", "test_route_count", "control_mean_speed_px_per_s",
             U_RAW, "MAP", "EXP", "AMI_beh", "AES_S", "CDS", "PSP", "VF"]
rows = []
for s in ["UK", "Spain", "China"]:
    sub = P[P.site == s]
    row = {"site": s, "n": len(sub)}
    for v in desc_vars:
        x = pd.to_numeric(sub[v], errors="coerce")
        row[f"{v}_mean"] = x.mean(); row[f"{v}_sd"] = x.std()
    rows.append(row)
R9 = pd.DataFrame(rows)
kw = {v: stats.kruskal(*[pd.to_numeric(P.loc[P.site == s, v], errors="coerce").dropna() for s in ["UK", "Spain", "China"]]).pvalue
      for v in desc_vars if v not in ("male", "clozapine")}
R9 = pd.concat([R9, pd.DataFrame([{"site": "Kruskal-Wallis p", **{f"{v}_mean": p for v, p in kw.items()}}])])
R9.to_csv(OUT / "R9_site_descriptives.csv", index=False)

# ---------------------------------------------------------------- R10 scale validity
rows = []
for a, b in [("AMI_beh", "AES_S"), ("AMI_soc", "AES_S"), ("AMI_emo", "AES_S"), ("AMI_beh", "MAP"), ("AMI_soc", "MAP"),
             ("AMI_beh", "CDS"), ("AMI_beh", "PSP"), ("AES_S", "MAP"), ("AES_S", "PSP"), ("MAP", "PSP")]:
    for analysis, cov in [("Unadjusted", []), ("Site-adjusted", ["site"])]:
        r, p, n = partial_spearman(P, a, b, cov)
        rows.append({"x": a, "y": b, "analysis": analysis, "rho": r, "p": p, "n": n})
R10 = pd.DataFrame(rows)
R10.to_csv(OUT / "R10_scale_validity.csv", index=False)

# AMI descriptives and diagnosis
ami_desc = P[["AMI_beh", "AMI_soc", "AMI_emo", "AES_S"]].describe().T
ami_desc.to_csv(OUT / "R11_ami_descriptives.csv")
P["MINI"].value_counts().to_csv(OUT / "R12_mini_codes.csv")

# ---------------------------------------------------------------- summary
fmtp = lambda p: "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")
L = ["# Revision statistics summary", "", "AMI scored as entered in the CHANSS database (higher = more apathy).", ""]
L += ["## R1 Fluency (route count)", R1[["comparison", "mean_control", "mean_patient", "hedges_g", "g_ci_low", "g_ci_high", "p",
                                       "tost_p_d0.3", "tost_p_d0.4", "tost_p_d0.5"]].round(3).to_markdown(index=False), ""]
L += ["## R2 Uniqueness robustness", R2[["comparison", "metric", "n_patient", "mean_control", "mean_patient", "ratio_patient_control",
                                        "welch_t", "p", "hedges_g", "g_ci_low", "g_ci_high"]].round(3).to_markdown(index=False), ""]
L += ["## R3 GLMs", R3.round(4).to_markdown(index=False), ""]
L += ["## R4 Clinical correlations", R4.round(3).to_markdown(index=False), ""]
L += ["## R5 VF specificity", R5.round(3).to_markdown(index=False), ""]
L += ["## R6 Multivariable", R6.round(3).to_markdown(index=False), ""]
L += ["## R7 Shapley by group", R7.round(3).to_markdown(index=False), ""]
L += ["## R8 Cluster stability", R8.round(3).to_markdown(index=False), ""]
L += ["## R9 Site descriptives", R9.round(2).T.to_markdown(), ""]
L += ["## R10 Scale validity", R10.round(3).to_markdown(index=False), ""]
L += ["## R11 AMI/AES descriptives", ami_desc.round(2).to_markdown(), ""]
L += [f"Control age range: {age_lo:.0f}-{age_hi:.0f}; patients within range: {int(P.age_all.between(age_lo, age_hi).sum())}"]
(OUT / "revision_summary.md").write_text("\n".join(L))

# ---------------------------------------------------------------- R13 component-specific clinical correlates (within patients)
# partial Spearman of each task component with functioning, verbal fluency and BNSS domains, adjusting for site and for
# overall (motor-residualised) uniqueness; BH-FDR across the full 4 x 9 family.
P["uniq_for_partial"] = P[U_PRES]
rows = []
for analysis, cov in [("Site-adjusted", ["site"]), ("Site- and uniqueness-adjusted", ["site", "uniq_for_partial"]),
                      ("Site-, uniqueness-, motor-, fluency-, age- and sex-adjusted", ["site", "uniq_for_partial", "control_mean_speed_px_per_s", "test_route_count", "age_all", "male"])]:
    block = []
    for m in MECH:
        for v, lab in [("PSP", "PSP"), ("VF", "BACS verbal fluency"), ("MAP", "BNSS motivation and pleasure"), ("EXP", "BNSS expressive deficit"),
                       ("BNSS_anhedonia", "BNSS anhedonia"), ("BNSS_asociality", "BNSS asociality"), ("BNSS_avolition", "BNSS avolition"),
                       ("BNSS_blunted", "BNSS blunted affect"), ("BNSS_alogia", "BNSS alogia")]:
            sub = P.copy()
            # continuous covariates are entered as ranks alongside site dummies
            for c in cov:
                if c != "site":
                    sub[c] = sub[c].rank()
            sub2 = sub[[m, v] + cov].dropna()
            Z = np.column_stack([np.ones(len(sub2))] + [pd.get_dummies(sub2["site"], drop_first=True, dtype=float).to_numpy()]
                                + [sub2[[c for c in cov if c != "site"]].to_numpy(float)])
            rx, ry = rank(sub2[m]), rank(sub2[v])
            ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
            ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
            r = np.corrcoef(ex, ey)[0, 1]
            dfree = len(sub2) - Z.shape[1] - 1
            pv = 2 * stats.t.sf(abs(r * math.sqrt(dfree / (1 - r * r))), dfree)
            block.append({"analysis": analysis, "component": MECH_LAB[m], "measure": lab, "n": len(sub2), "rho": r, "p": pv})
    q = bh([b["p"] for b in block])
    for b, qq in zip(block, q):
        b["q_fdr"] = qq
    rows += block
R13 = pd.DataFrame(rows)
R13.to_csv(OUT / "R13_component_clinical_partial.csv", index=False)
with open(OUT / "revision_summary.md", "a") as fh:
    fh.write("\n\n## R13 Component-specific clinical correlates\n" + R13.round(3).to_markdown(index=False))

print("Wrote revision outputs to", OUT)
