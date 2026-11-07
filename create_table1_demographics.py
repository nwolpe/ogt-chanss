#!/usr/bin/env python3
"""Create Table 1 demographic/clinical summary for the OGT paper.

The table is anchored to the primary analysis sample: participants in the
Phase B participant-level table with non-missing test uniqueness. Controls have
age/sex/hand extracted from JATOS demographics; patient clinical/cognitive rows
come from CHANSS_clinical_demog.csv via the same ID-correction file used by the
main analyses.
"""

from __future__ import annotations

import re
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

PHASEB_CSV = Path("analysis_outputs/phaseB/phaseB_participant_metrics.csv")
PARTICIPANT_METRICS_CSV = Path("analysis_outputs/ogt_participant_metrics.csv")
CLINICAL_CSV = Path("CHANSS_clinical_demog.csv")
CORRECTIONS_CSV = Path("analysis_outputs/ogt_patients_missing_clinical_data_corrected.csv")
OUTPUT_DIR = Path("analysis_outputs/table1")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def to_float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    s = str(value).strip()
    if not s or s.lower() in {"na", "nan", "none", "-"}:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def is_finite(value: float) -> bool:
    return not math.isnan(value) and math.isfinite(value)


def compute_is_fep_5y(age: float, age_fep: float, window_years: float = 5.0) -> float:
    if not is_finite(age) or not is_finite(age_fep):
        return math.nan
    illness_duration = age - age_fep
    return 1.0 if 0.0 <= illness_duration <= window_years else 0.0


def compute_is_trs(ap_1: object, keyword: str = "clozapine") -> float:
    s = str(ap_1).strip().lower()
    if not s or s in {"nan", "na", "none", "-"}:
        return math.nan
    return 1.0 if keyword in s else 0.0


def numeric_values(rows: list[dict[str, object]], field: str) -> list[float]:
    vals = [to_float(row.get(field, math.nan)) for row in rows]
    return [v for v in vals if is_finite(v)]


def mean_sd(rows: list[dict[str, object]], field: str, digits: int = 1) -> str:
    vals = numeric_values(rows, field)
    if not vals:
        return ""
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean(vals):.{digits}f} ({sd:.{digits}f})"


def mean_sd_n(rows: list[dict[str, object]], field: str, denom: int, digits: int = 1) -> str:
    vals = numeric_values(rows, field)
    if not vals:
        return ""
    suffix = f", n = {len(vals)}" if len(vals) < denom else ""
    return f"{mean_sd(rows, field, digits)}{suffix}"


def median_iqr(rows: list[dict[str, object]], field: str, digits: int = 1) -> str:
    vals = sorted(numeric_values(rows, field))
    if not vals:
        return ""
    q1, med, q3 = percentile(vals, 25), percentile(vals, 50), percentile(vals, 75)
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def n_percent(n: int, denom: int) -> str:
    return f"{n} ({100.0 * n / denom:.1f}%)" if denom else ""


def counts_with_n(parts: list[int], denom: int, labels: list[str] | None = None) -> str:
    value = " / ".join(str(part) for part in parts)
    if labels:
        value = f"{value} {' / '.join(labels)}"
    observed = sum(parts)
    if observed < denom:
        value = f"{value}, n = {observed}"
    return value


def format_p(p: float) -> str:
    if not is_finite(p):
        return ""
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def normal_sf(z: float) -> float:
    """Survival function for the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ttest_welch_approx(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t-test with a normal-tail p-value approximation.

    The table uses these p-values descriptively. This avoids a SciPy dependency
    while keeping the inferential column reproducible in the local environment.
    """
    ma, mb = mean(a), mean(b)
    va = stdev(a) ** 2 if len(a) > 1 else 0.0
    vb = stdev(b) ** 2 if len(b) > 1 else 0.0
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return math.nan, math.nan
    t_value = (ma - mb) / se
    p_value = 2.0 * normal_sf(abs(t_value))
    return t_value, p_value


def format_t_test(a: list[float], b: list[float]) -> str:
    if len(a) < 2 or len(b) < 2:
        return ""
    t_value, p_value = ttest_welch_approx(a, b)
    return f"t = {t_value:.2f}, {format_p(float(p_value))}"


def format_chi2(table: list[list[int]]) -> str:
    if not table or any(sum(row) == 0 for row in table):
        return ""
    rows = len(table)
    cols = len(table[0])
    total = sum(sum(row) for row in table)
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[r][c] for r in range(rows)) for c in range(cols)]
    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            expected = row_totals[r] * col_totals[c] / total
            if expected > 0:
                chi2 += (table[r][c] - expected) ** 2 / expected
    if rows == 2 and cols == 2:
        p_value = math.erfc(math.sqrt(chi2 / 2.0))
    else:
        p_value = math.nan
    return f"chi2 = {chi2:.2f}, {format_p(float(p_value))}"


def patient_site(row: dict[str, object]) -> str:
    platform = str(row.get("source_platform", ""))
    subject_id = str(row.get("clinical_lookup_id", row.get("participant_id", ""))).upper()
    if platform == "OGT_Mandarin" or subject_id.startswith("CH"):
        return "China"
    # Spanish sites: Barcelona (C3), Bilbao (C7; English-language task), Santander (C8)
    if platform == "OGT_Spanish" or re.match(r"^C[378]", subject_id):
        return "Spain"
    if platform in {"OGT_English", "OCTAL"} or subject_id.startswith("C"):
        return "UK"
    return "Unknown"


def add_patient_flags(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        merged = dict(row)
        age = to_float(merged.get("Age", math.nan))
        age_fep = to_float(merged.get("Age_FEP", math.nan))
        merged["illness_duration"] = age - age_fep if is_finite(age) and is_finite(age_fep) else math.nan
        merged["is_fep_5y"] = compute_is_fep_5y(age, age_fep)
        merged["is_trs"] = compute_is_trs(merged.get("AP_1", ""))
        merged["site"] = patient_site(merged)
        out.append(merged)
    return out


def control_demo_lookup() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(PARTICIPANT_METRICS_CSV)
    return {(row["group"], row["participant_id"]): row for row in rows}


def load_clinical_id_corrections(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    corrections = {}
    for row in read_csv(path):
        ogt_id = str(row.get("participant_id", "")).strip().upper()
        corrected_id = str(row.get("participant_id_corrected", "")).strip().upper()
        if ogt_id and corrected_id:
            corrections[ogt_id] = corrected_id
    return corrections


def load_clinical_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(path):
        enriched: dict[str, object] = dict(row)
        enriched["ID_subject"] = str(row.get("ID_subject", "")).strip().upper()

        map_terms = [to_float(row.get(key, "")) for key in ["BNSS_1", "BNSS_2", "BNSS_3", "BNSS_5", "BNSS_6", "BNSS_7", "BNSS_8"]]
        exp_terms = [to_float(row.get(key, "")) for key in ["BNSS_9", "BNSS_10", "BNSS_11", "BNSS_12", "BNSS_13"]]
        enriched["MAP"] = sum(v for v in map_terms if is_finite(v)) if any(is_finite(v) for v in map_terms) else math.nan
        enriched["EXP"] = sum(v for v in exp_terms if is_finite(v)) if any(is_finite(v) for v in exp_terms) else math.nan

        ami_scored = {}
        for i in range(1, 19):
            raw = to_float(row.get(f"AMI_{i}", ""))
            ami_scored[i] = raw if is_finite(raw) else math.nan  # CHANSS database already apathy-scored (Completely true = 0 ... Completely untrue = 4); no reversal

        def item_mean(indices: list[int]) -> float:
            vals = [ami_scored[i] for i in indices if is_finite(ami_scored[i])]
            return mean(vals) if vals else math.nan

        enriched["AMI_behavioural"] = item_mean([5, 9, 10, 11, 12, 15])
        enriched["AMI_social"] = item_mean([2, 3, 4, 8, 14, 17])
        enriched["AMI_emotional"] = item_mean([1, 6, 7, 13, 16, 18])

        cds_items = [to_float(row.get(f"CDS_{i}", "")) for i in range(1, 10)]
        enriched["CDS_total"] = sum(v for v in cds_items if is_finite(v)) if any(is_finite(v) for v in cds_items) else math.nan

        marder_items = [to_float(row.get(key, "")) for key in ("PANSS_p1", "PANSS_p3", "PANSS_pg9", "PANSS_pg10")]
        enriched["PANSS_marder_positive"] = (
            sum(v for v in marder_items if is_finite(v)) if any(is_finite(v) for v in marder_items) else math.nan
        )
        rows.append(enriched)
    return rows


def build_clinical_match_rows(
    participant_rows: list[dict[str, object]],
    clinical_rows: list[dict[str, object]],
    id_corrections: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clinical_by_id = {
        str(row.get("ID_subject", "")).upper(): row
        for row in clinical_rows
        if str(row.get("ID_subject", "")).strip()
    }
    matched = []
    missing = []
    for row in participant_rows:
        participant_id = str(row.get("participant_id", "")).strip().upper()
        clinical_lookup_id = id_corrections.get(participant_id, participant_id)
        clinical = clinical_by_id.get(clinical_lookup_id)
        if clinical is None:
            missing.append(row)
            continue
        merged = dict(row)
        merged["clinical_lookup_id"] = clinical_lookup_id
        merged.update(clinical)
        matched.append(merged)
    return matched, missing


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["section", "measure", "patients", "controls", "inferential_statistic"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], n_patient: int, n_control: int) -> None:
    lines = [
        "# Table 1. Demographic, cognitive, and clinical measures in the primary analysis sample",
        "",
        f"| Measure | Patients (n = {n_patient}) | Controls (n = {n_control}) | Inferential statistic |",
        "|---|---:|---:|---:|",
    ]
    current_section = ""
    for row in rows:
        if row["section"] != current_section:
            current_section = row["section"]
            lines.append(f"| **{current_section}** |  |  |  |")
        lines.append(
            "| {measure} | {patients} | {controls} | {inferential_statistic} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Values are mean (SD), median [IQR], or n (%). The sample is restricted to participants with non-missing test-session mean uniqueness in `analysis_outputs/phaseB/phaseB_participant_metrics.csv`.",
            "Inferential tests are Welch t-tests for continuous measures and chi-square tests for categorical measures; p-values are large-sample approximations so that the table can be regenerated without SciPy.",
            "Clinical and cognitive measures were available for patients only in the current extracted datasets. Cells show measure-specific n where it is smaller than the column-header sample size.",
            "FEP = first episode psychosis within 5 years of illness onset; TRS = treatment-resistant schizophrenia, operationalised as current clozapine in `AP_1`; AMI = Apathy Motivation Index; BACS = Brief Assessment of Cognition in Schizophrenia; BNSS MAP/EXP = motivation-and-pleasure / expressive-deficit subscales.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    phaseb_rows = read_csv(PHASEB_CSV)
    primary_rows = [
        row for row in phaseb_rows if is_finite(to_float(row.get("test_uniqueness_mean", math.nan)))
    ]

    demo_by_key = control_demo_lookup()
    enriched_rows = []
    for row in primary_rows:
        merged = dict(row)
        merged.update(demo_by_key.get((row["group"], row["participant_id"]), {}))
        enriched_rows.append(merged)

    controls = [row for row in enriched_rows if row["group"] == "control"]
    patient_task_rows = [row for row in enriched_rows if row["group"] == "patient"]
    clinical_rows = load_clinical_rows(CLINICAL_CSV)
    corrections = load_clinical_id_corrections(CORRECTIONS_CSV)
    patients, missing = build_clinical_match_rows(patient_task_rows, clinical_rows, corrections)
    patients = add_patient_flags(patients)

    n_patient = len(patient_task_rows)
    n_control = len(controls)
    n_patient_clinical = len(patients)

    patient_gender = Counter("Male" if str(r.get("Gender")) == "1" else "Female" for r in patients)
    control_gender = Counter(str(r.get("sex", "")).strip().title() for r in controls)
    patient_sites = Counter(r["site"] for r in patients)
    control_hand = Counter(str(r.get("hand", "")).strip().title() for r in controls if str(r.get("hand", "")).strip())

    patient_fep_known = [r for r in patients if is_finite(to_float(r.get("is_fep_5y", math.nan)))]
    patient_trs_known = [r for r in patients if is_finite(to_float(r.get("is_trs", math.nan)))]

    rows: list[dict[str, str]] = []

    def add(section: str, measure: str, patients_value: str, controls_value: str = "", stat: str = "") -> None:
        rows.append(
            {
                "section": section,
                "measure": measure,
                "patients": patients_value,
                "controls": controls_value,
                "inferential_statistic": stat,
            }
        )

    add("Sample", "Primary OGT analysis sample", str(n_patient), str(n_control))
    add("Sample", "Matched patient clinical data", f"{n_patient_clinical}/{n_patient}", "", "")
    if missing:
        add("Sample", "Unmatched patient clinical data", str(len(missing)), "", "")

    add(
        "Demographic",
        "Age, years",
        mean_sd_n(patients, "Age", n_patient),
        mean_sd_n(controls, "age", n_control),
        format_t_test(numeric_values(patients, "Age"), numeric_values(controls, "age")),
    )
    add(
        "Demographic",
        "Sex, female / male",
        counts_with_n([patient_gender["Female"], patient_gender["Male"]], n_patient),
        counts_with_n([control_gender["Female"], control_gender["Male"]], n_control),
        format_chi2(
            [
                [patient_gender["Female"], patient_gender["Male"]],
                [control_gender["Female"], control_gender["Male"]],
            ]
        ),
    )
    add(
        "Demographic",
        "Patient site, UK / Spain / China",
        counts_with_n([patient_sites["UK"], patient_sites["Spain"], patient_sites["China"]], n_patient),
        "",
        "",
    )
    add(
        "Demographic",
        "Control handedness, right / left",
        "",
        counts_with_n([control_hand["Right"], control_hand["Left"]], n_control),
        "",
    )
    add("Demographic", "Education, years", mean_sd_n(patients, "pEducation", n_patient), "", "")

    add("Illness", "Age at first episode, years", mean_sd_n(patients, "Age_FEP", n_patient), "", "")
    add("Illness", "Illness duration, years", mean_sd_n(patients, "illness_duration", n_patient), "", "")
    add(
        "Illness",
        "FEP within 5 years",
        f"{n_percent(sum(int(to_float(r['is_fep_5y'])) for r in patient_fep_known), len(patient_fep_known))}, n = {len(patient_fep_known)}",
        "",
        "",
    )
    add(
        "Illness",
        "TRS / current clozapine",
        f"{n_percent(sum(int(to_float(r['is_trs'])) for r in patient_trs_known), len(patient_trs_known))}, n = {len(patient_trs_known)}",
        "",
        "",
    )

    add("Cognitive", "BACS verbal fluency T-score", mean_sd_n(patients, "BACS_VF_t", n_patient), "", "")
    add("Cognitive", "BACS composite T-score", mean_sd_n(patients, "BACS_comp_t", n_patient), "", "")
    add("Cognitive", "BACS digit sequencing T-score", mean_sd_n(patients, "BACS_DS_t", n_patient), "", "")
    add("Cognitive", "BACS trail-making T-score", mean_sd_n(patients, "BACS_TMT_t", n_patient), "", "")

    add("Clinical", "PANSS Marder positive", mean_sd_n(patients, "PANSS_marder_positive", n_patient), "", "")
    add("Clinical", "BNSS MAP", mean_sd_n(patients, "MAP", n_patient), "", "")
    add("Clinical", "BNSS EXP", mean_sd_n(patients, "EXP", n_patient), "", "")
    add("Clinical", "AMI behavioural", mean_sd_n(patients, "AMI_behavioural", n_patient), "", "")
    add("Clinical", "AMI social", mean_sd_n(patients, "AMI_social", n_patient), "", "")
    add("Clinical", "AMI emotional", mean_sd_n(patients, "AMI_emotional", n_patient), "", "")
    add("Clinical", "CDSS total", mean_sd_n(patients, "CDS_total", n_patient), "", "")
    add("Clinical", "PSP score", mean_sd_n(patients, "PSP_score", n_patient), "", "")

    add(
        "Task",
        "Test route count",
        mean_sd_n(patient_task_rows, "test_route_count", n_patient),
        mean_sd_n(controls, "test_route_count", n_control),
        format_t_test(numeric_values(patient_task_rows, "test_route_count"), numeric_values(controls, "test_route_count")),
    )
    add(
        "Task",
        "Test mean uniqueness",
        mean_sd_n(patient_task_rows, "test_uniqueness_mean", n_patient),
        mean_sd_n(controls, "test_uniqueness_mean", n_control),
        format_t_test(
            numeric_values(patient_task_rows, "test_uniqueness_mean"),
            numeric_values(controls, "test_uniqueness_mean"),
        ),
    )

    write_csv(OUTPUT_DIR / "table1_demographics.csv", rows)
    write_markdown(OUTPUT_DIR / "table1_demographics.md", rows, n_patient, n_control)
    print(f"Wrote {OUTPUT_DIR / 'table1_demographics.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'table1_demographics.md'}")


if __name__ == "__main__":
    main()
