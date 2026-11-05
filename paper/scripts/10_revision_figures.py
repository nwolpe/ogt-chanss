#!/usr/bin/env python3
"""Revision figures (Figures 3 and 5, Supplementary Figure S1).

Run from the OGT repository root after 07 (CCA) and 09 (revision statistics):
  python3 paper/scripts/10_revision_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(".")
REV = ROOT / "paper/outputs/revision"
CCA = ROOT / "paper/outputs/cca"
FIG = ROOT / "paper/outputs/revision/figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#222222", "#6b6b66", "#e6e5e0"
SITE_COL = {"Controls": "#8a8a85", "UK": "#2a78d6", "Spain": "#eb6834", "China": "#1baf7a"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left"})

d = pd.read_csv(REV / "revision_analysis_dataset.csv", low_memory=False)
P = d[d.group == "patient"].copy()
Ct = d[d.group == "control"].copy()
rng = np.random.default_rng(7)


def label(ax, s):
    ax.text(-0.13, 1.06, s, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, va="bottom")


def strip(ax, groups, values, colors, ylabel, log=False):
    for i, (g, v, c) in enumerate(zip(groups, values, colors)):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        x = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v, s=10, color=c, alpha=0.55, linewidths=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2, solid_capstyle="round")
        ax.plot([i, i], [q1, q3], color=INK, lw=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ================================================================ Figure 3
r4 = pd.read_csv(REV / "R4_clinical_correlations.csv")
r2 = pd.read_csv(REV / "R2_uniqueness_robustness.csv")
fig = plt.figure(figsize=(10.2, 7.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.42, wspace=0.42)

# A: uniqueness by patient site, healthy reference group shown for context
ax = fig.add_subplot(gs[0, 0])
groups = ["Healthy\nreference", "UK\n(n = 87)", "Spain\n(n = 28)", "China\n(n = 35)"]
vals = [Ct.test_uniqueness_mean, P[P.site == "UK"].test_uniqueness_mean, P[P.site == "Spain"].test_uniqueness_mean,
        P[P.site == "China"].test_uniqueness_mean]
strip(ax, groups, vals, [SITE_COL[k] for k in ["Controls", "UK", "Spain", "China"]], "Mean route uniqueness (log scale)", log=True)
ax.axvline(0.5, color=GRID, lw=1)
ax.set_title("Uniqueness by recruitment site")
label(ax, "A")

# B: fluency-uniqueness trade-off
ax = fig.add_subplot(gs[0, 1])
ax.scatter(Ct.test_route_count, Ct.test_uniqueness_mean, s=12, color=SITE_COL["Controls"], alpha=0.5, linewidths=0, label="Healthy reference")
for s in ["UK", "Spain", "China"]:
    sub = P[P.site == s]
    ax.scatter(sub.test_route_count, sub.test_uniqueness_mean, s=14, color=SITE_COL[s], alpha=0.8, linewidths=0.4,
               edgecolors="white", label=f"Patients, {s}")
ax.set_xscale("log"); ax.set_yscale("log")
rho = stats.spearmanr(P.test_route_count, P.test_uniqueness_mean, nan_policy="omit")[0]
ax.text(0.98, 0.96, f"Patients: ρ = {rho:.2f}", transform=ax.transAxes, ha="right", va="top", color=INK)
ax.set_xlabel("Number of routes (fluency, log scale)")
ax.set_ylabel("Mean route uniqueness (log scale)")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Fluency–uniqueness trade-off")
label(ax, "B")

# C: clinical correlates of uniqueness within patients, unadjusted vs site-adjusted
ax = fig.add_subplot(gs[1, 0])
sub = r4[r4.predictor == "Motor-residualised uniqueness"]
order = ["BACS verbal fluency", "BACS verbal memory", "BACS digit sequencing", "BACS token motor", "BACS symbol coding",
         "BACS Tower of London", "BNSS motivation and pleasure", "BNSS expressive deficit", "AMI behavioural", "AMI social",
         "AMI emotional", "AES (self-report)", "CDS total", "PANSS positive", "PSP"]
y = np.arange(len(order))[::-1]
for analysis, off, col, mk in [("Unadjusted", 0.17, "#9a9993", "o"), ("Site-adjusted", -0.17, INK, "D")]:
    s = sub[sub.analysis == analysis].set_index("measure").loc[order]
    if analysis == "Unadjusted":
        ax.hlines(y + off, s.ci_low, s.ci_high, color=col, lw=1)
    sig = s.q_fdr < .05
    ax.scatter(s.rho, y + off, s=22, marker=mk, facecolors=np.where(sig, col, "white"), edgecolors=col, linewidths=1,
               label=f"{analysis} (filled: FDR q < .05)", zorder=3)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(order)
ax.set_xlabel("Spearman ρ with motor-residualised uniqueness")
ax.set_xlim(-0.4, 0.65)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Clinical correlates within patients")
label(ax, "C")

# D: comparison with healthy reference group - robustness
ax = fig.add_subplot(gs[1, 1])
rows = [("All patients", "All patients"), ("All patients (motor-residualised)", "Motor-residualised"),
        ("Site: UK", "UK patients only"), ("Site: Spain", "Spain patients only"), ("Site: China", "China patients only"),
        ("Same coordinate frame as controls", "Same coordinate frame"), (r2.comparison[r2.comparison.str.startswith("Age-overlap")].iloc[0], "Age-overlap (18–39 y)"),
        ("Female participants only", "Women only"), ("Male participants only", "Men only"),
        ("Reference: other participants only", "Ref.: other participants"), ("Reference: control routes only", "Ref.: control routes"),
        ("Reference: balanced 100 v 100 pool", "Ref.: balanced pool"), ("Within-person mean pairwise distance", "Within-person spread")]
s = r2.set_index("comparison")
yy = np.arange(len(rows))[::-1]
g = [s.loc[k, "hedges_g"] for k, _ in rows]
lo = [s.loc[k, "g_ci_low"] for k, _ in rows]
hi = [s.loc[k, "g_ci_high"] for k, _ in rows]
ax.hlines(yy, lo, hi, color=SITE_COL["UK"], lw=1.2)
ax.scatter(g, yy, s=24, color=SITE_COL["UK"], zorder=3)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([lab for _, lab in rows])
ax.set_xlabel("Hedges g (healthy reference − patients), 95% CI")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Context: difference from healthy reference group")
label(ax, "D")
fig.savefig(FIG / "figure3_revised.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIG / "figure3_revised.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================================================================ Figure 5 (CCA)
dims = pd.read_csv(CCA / "cca_dimensions.csv")
loads = pd.read_csv(CCA / "cca_loadings_bootstrap.csv")
main = "Beyond uniqueness: also partialling residualised uniqueness"
L = loads[(loads.analysis == main) & (loads.dimension == 1)].copy()
# orient so that roaming loads positively
if L.loc[L.variable == "Roaming", "loading"].iloc[0] < 0:
    L[["loading", "ci_low", "ci_high"]] = -L[["loading", "ci_high", "ci_low"]].to_numpy()
lab = {"Roaming": "Roaming entropy", "Elaboration": "Option elaboration", "PerseverationRatio": "Perseveration ratio\n(higher = less perseverative)",
       "Novelty": "Novelty maintenance", "Anhedonia": "BNSS anhedonia", "Asociality": "BNSS asociality", "Avolition": "BNSS avolition",
       "BluntedAffect": "BNSS blunted affect", "Alogia": "BNSS alogia", "Distress_lack": "BNSS lack of distress",
       "AMI_beh": "AMI behavioural", "AMI_soc": "AMI social", "AMI_emo": "AMI emotional", "CDS": "CDS total",
       "VerbalFluency": "BACS verbal fluency", "BACS_composite": "BACS composite", "PSP": "PSP (functioning)"}
fig = plt.figure(figsize=(10.6, 9.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[1, 1.25], hspace=0.38, wspace=0.75)


def loadplot(ax, sub, title, letter):
    sub = sub.sort_values("loading")
    yy = np.arange(len(sub))
    sig = (sub.ci_low > 0) | (sub.ci_high < 0)
    col = np.where(sub.loading > 0, SITE_COL["UK"], SITE_COL["Spain"])
    ax.hlines(yy, sub.ci_low, sub.ci_high, color=col, lw=1)
    ax.scatter(sub.loading, yy, s=26, facecolors=np.where(sig, col, "white"), edgecolors=col, zorder=3, linewidths=1.1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([lab[v] for v in sub.variable])
    ax.set_xlim(-1, 1); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Structure correlation (95% bootstrap CI)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title); label(ax, letter)


ax = fig.add_subplot(gs[0, 0]); loadplot(ax, L[L.side == "mechanism"], "Canonical dimension: task components", "A")
ax = fig.add_subplot(gs[0, 1]); loadplot(ax, L[L.side == "clinical"], "Canonical dimension: clinical measures", "B")
# C: component-specific partial correlations (site- and uniqueness-adjusted)
r13 = pd.read_csv(REV / "R13_component_clinical_partial.csv")
r13 = r13[r13.analysis == "Site- and uniqueness-adjusted"]
comps = ["Roaming entropy", "Option elaboration", "Perseveration ratio", "Novelty maintenance"]
meas = ["PSP", "BACS verbal fluency", "BNSS motivation and pleasure", "BNSS anhedonia", "BNSS asociality", "BNSS avolition",
        "BNSS expressive deficit", "BNSS blunted affect", "BNSS alogia"]
M = r13.pivot(index="component", columns="measure", values="rho").loc[comps, meas]
Q = r13.pivot(index="component", columns="measure", values="q_fdr").loc[comps, meas]
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
for i in range(len(comps)):
    for j in range(len(meas)):
        v, q = M.iloc[i, j], Q.iloc[i, j]
        ax.text(j, i, f"{v:.2f}".replace("0.", ".") + ("*" if q < .05 else ""), ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) > 0.22 else INK, fontweight="bold" if q < .05 else "normal")
ax.set_xticks(range(len(meas))); ax.set_xticklabels(meas, rotation=50, ha="right")
ax.set_yticks(range(len(comps))); ax.set_yticklabels(comps)
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Partial Spearman ρ")
ax.set_title("Component-specific correlates\n(site- and uniqueness-adjusted; * FDR q < .05)"); label(ax, "C")
ax = fig.add_subplot(gs[1, 1])
an = [("Main: all patients, site-adjusted", "Components only"), (main, "Beyond uniqueness"),
      ("Beyond uniqueness, without AMI (larger n)", "  without AMI"), ("Beyond uniqueness, excluding China", "  excluding China"),
      ("Beyond uniqueness, UK only", "  UK only"), ("Beyond uniqueness + motor speed, route count, age, sex", "  + motor, fluency,\n    age, sex")]
dd = dims[dims.dimension == 1].set_index("analysis")
yy = np.arange(len(an))[::-1]
for yv, (k, nm) in zip(yy, an):
    r, p, n = dd.loc[k, "canonical_r"], dd.loc[k, "p_perm"], dd.loc[k, "n"]
    ax.barh(yv, r, height=0.55, color=SITE_COL["UK"] if p < .05 else "#b9b8b2")
    ax.text(r + 0.02, yv, f"p = {p:.3f}".replace("0.", ".") + f"  (n = {n})", va="center", fontsize=7.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([nm for _, nm in an])
ax.set_xlim(0, 1.0); ax.set_xlabel("First canonical correlation")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Canonical correlation: robustness\n(within-site permutation p)"); label(ax, "D")
fig.savefig(FIG / "figure5_revised.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIG / "figure5_revised.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================================================================ Supplementary Figure S1 (apathy distributions, corrected AMI)
fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2))
for ax, (v, t, rngx) in zip(axes.ravel(), [("AMI_beh", "AMI behavioural", (0, 4)), ("AMI_soc", "AMI social", (0, 4)),
                                           ("AMI_emo", "AMI emotional", (0, 4)), ("AES_S", "AES (self-report)", (18, 72)),
                                           ("MAP", "BNSS motivation and pleasure", (0, 42)), ("EXP", "BNSS expressive deficit", (0, 30))]):
    x = pd.to_numeric(P[v], errors="coerce").dropna()
    ax.hist(x, bins=16, range=rngx, color=SITE_COL["UK"], alpha=0.85, edgecolor="white", linewidth=1)
    ax.axvline(x.median(), color=INK, lw=1.4)
    ax.set_title(t); ax.set_xlim(*rngx)
    ax.text(0.98, 0.95, f"n = {len(x)}\nmedian = {x.median():.2f}\nmean = {x.mean():.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    ax.set_xlabel("Score (higher = more apathy / symptoms)")
axes[0, 0].set_ylabel("Patients"); axes[1, 0].set_ylabel("Patients")
fig.tight_layout()
fig.savefig(FIG / "figureS1_apathy_distributions_revised.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Wrote figures to", FIG)
