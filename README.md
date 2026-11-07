# Option generation in schizophrenia (OGT, CHANSS)

Analysis code for a study of option generation in schizophrenia. The reference will be added on publication.

The study uses the option generation task (OGT; Ang et al., *Curr Biol* 2018), in which participants draw as
many different paths as possible between two points on a touchscreen. Data come from 150 patients with
schizophrenia in the multisite CHANSS study (UK, Spain, China; protocol: Wolpe et al., *BJPsych Open* 2025)
and 100 healthy reference participants tested at Tel Aviv University.

The code computes route-level uniqueness (nearest-neighbour distance in a position–velocity–acceleration feature
space, mirror-invariant), four task-derived components (roaming entropy, option elaboration, perseveration,
novelty maintenance), and the statistical analyses reported in the paper: within-patient general linear models,
site-adjusted partial Spearman correlations with FDR correction, Shapley R² decomposition, canonical correlation
analysis with within-site permutation tests, and reference-group sensitivity analyses.

## Data availability

**No participant data are included in this repository.** Raw task recordings and clinical data from CHANSS
are available from the corresponding author on reasonable request, subject to the CHANSS data-sharing
agreements and ethics approvals.

To run the pipeline, place the data in the repository root with this layout:

```
data_controls/                  reference-group JATOS exports (one study_result_* folder per participant)
data_patients/OCTAL/            OCTAL task exports (.txt), one UK site
data_patients/OGT_English/      web-based task exports (JATOS study_result_* folders)
data_patients/OGT_Spanish/
data_patients/OGT_Mandarin/
control_demographics_sorted.csv reference-group demographics (subID, study_result_id, age, sex, hand, ...)
control_demographics_additional.csv  demographics for NC_*/NH_* reference participants (subID, age, sex, education_years, testing_date)
CHANSS_clinical_demog.csv       CHANSS clinical, cognitive and demographic database export
paper/CHANSS_clinical_demog_filled_APLoad.xlsx   antipsychotic load (% BNF maximum) per patient
analysis_outputs/ogt_patients_missing_clinical_data_corrected.csv   task-ID to clinical-ID corrections
```

Reference participants are those tested before the data freeze of 1 May 2026 (`CONTROL_DATA_FREEZE_MS` in
`analyze_ogt_groups.py`). Where a participant restarted the task, the attempt with the most completed test paths is
used; `paper/outputs/shared/extra_control_attempts.csv` logs every attempt.

AMI items in the CHANSS database are entered already apathy-scored (Completely true = 0 … Completely untrue = 4);
the scripts use them without reversal.

## Requirements

- Python ≥ 3.10: `pip install -r requirements.txt`
- R ≥ 4.1 (≥ 4.4 recommended): `Rscript install_r_packages.R`

Computing route uniqueness and the components builds distance matrices over ~13,800 routes and needs about
8 GB of RAM.

## Running the pipeline

Run everything from the repository root (`bash run_all.sh`), or step by step:

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `analyze_ogt_groups.py` | Loads raw task data, harmonises coordinate frames, participant-level route counts and speeds | `analysis_outputs/ogt_participant_metrics.csv` |
| 2 | `paper/scripts/00_recompute_paper_route_uniqueness.py` | Route uniqueness for all 250 participants; control-condition motor features | `paper/outputs/shared/` |
| 3 | `phase_B_exploration_dynamics.py` | Four task-derived components (uses `phase_B_mechanisms.py`, `motor_purification.py`) | `analysis_outputs/phaseB_exploration_dynamics/` |
| 4 | `phase_B_mechanisms.py` | Participant-level component table used for Table 1 | `analysis_outputs/phaseB/` |
| 5 | `create_table1_demographics.py` | Sample characteristics | `analysis_outputs/table1/` |
| 6 | `paper/scripts/01_section1_sample_performance.R` | Raw task performance | `paper/outputs/section1/` |
| 7 | `paper/scripts/02_section2_motor_adjusted_uniqueness.R` | Motor-residualised uniqueness; reference-group GLM | `paper/outputs/section2/` |
| 8 | `paper/scripts/03_section3_cognitive_mechanisms.R` | Component correlations, Shapley R² (combined sample), attenuation | `paper/outputs/section3/` |
| 9 | `paper/scripts/07_cca_mechanisms_clinical.R` | Canonical correlation analyses with within-site permutations | `paper/outputs/cca/` |
| 9b | `paper/scripts/07b_cca_canonical_scores.R` | Patient scores on the first canonical dimension; distribution and influence diagnostics | `paper/outputs/cca/` |
| 10 | `paper/scripts/08_reference_pool_sensitivity.py` | Uniqueness with alternative reference sets | `paper/outputs/revision/` |
| 11 | `paper/scripts/09_revision_statistics.py` | Within-patient models, FDR, TOST, Shapley by group, cluster stability, site descriptives | `paper/outputs/revision/` |
| 12 | `paper/scripts/10_revision_figures.py` | Figures 2, 4 and S1 | `paper/outputs/revision/figures/` |

Additional checks, not required for the manuscript results: `04_section4_kmeans_profiles.R` and
`04b_section4_supplementary_cluster_characteristics.R` (original k-means analysis), `05_sensitivity_checks.R`
(AMI scoring direction, site and coordinate-frame checks) and `06_exploratory_mechanisms_bnss.R`
(component × BNSS domain correlations). `create_methods_mechanism_schematic.py` draws the component schematics
in Figure 1C–F.

## Where each result comes from

| Manuscript item | Source |
|---|---|
| Table 1 | `create_table1_demographics.py`; treatment variables (clozapine, polypharmacy, antidepressants) from `09_revision_statistics.py` |
| Figure 1C–F | `create_methods_mechanism_schematic.py` (panels A–B are illustrations) |
| Figure 2 | `10_revision_figures.py` (`figure3_revised`) |
| Figure 3 | `03_section3_cognitive_mechanisms.R` (`figure4_cognitive_mechanisms`) |
| Figure 4 | `10_revision_figures.py` (`figure5_revised`) |
| Figure S1 | `10_revision_figures.py` (`figureS1_apathy_distributions_revised`) |
| Figure S2 | `01_section1_sample_performance.R` (`figure2_raw_task_performance`) |
| Table S1 | `09_revision_statistics.py` (`R3_glms.csv`) |
| Table S2 | `09_revision_statistics.py` (`R9_site_descriptives.csv`) |
| Table S3 | `09_revision_statistics.py` (`R2_uniqueness_robustness.csv`), using `08_reference_pool_sensitivity.py` |
| Table S4 | `09_revision_statistics.py` (`R4_clinical_correlations.csv`, `R6_clinical_multivariable.csv`) |
| Table S5 | `09_revision_statistics.py` (`R7_shapley_by_group.csv`) |
| Table S6 | `07_cca_mechanisms_clinical.R` (`cca_dimensions.csv`, `cca_omnibus.csv`) |
| Figure S3 | `07b_cca_canonical_scores.R` (`figureS3_cca_dim1_scores`, `cca_dim1_distribution_diagnostics.csv`) |
| Table S7 | `09_revision_statistics.py` (`R8_cluster_stability.csv`) |
| Component–clinical partial correlations | `09_revision_statistics.py` (`R13_component_clinical_partial.csv`) |
| Fluency equivalence (TOST) | `09_revision_statistics.py` (`R1_fluency_equivalence.csv`) |

Random seeds are fixed in each script. With different numerical-library versions, submovement counts (option
elaboration) can differ by one or two for a few participants, because of near-ties in the detection of speed
minima. This changes downstream estimates only in the third decimal place (e.g., canonical correlations and
permutation p-values); roaming entropy, perseveration, novelty maintenance and uniqueness are unaffected.

## AI assistance

This repository was prepared with assistance from Claude (Anthropic), an AI assistant, which was used to
review the analysis code against the manuscript, write parts of the analysis scripts, and organise and document
the repository. The authors reviewed and tested the code and take full responsibility for it and for the results
reported in the paper.

## Citation

Citation details will be added on publication.

## Licence

MIT (see `LICENSE`).
