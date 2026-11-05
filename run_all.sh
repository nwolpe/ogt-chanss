#!/usr/bin/env bash
# Reproduce all manuscript analyses. Run from the repository root with the data in place (see README).
set -euo pipefail
PY=${PYTHON:-python3}

$PY analyze_ogt_groups.py
$PY paper/scripts/00_recompute_paper_route_uniqueness.py
$PY phase_B_exploration_dynamics.py
$PY phase_B_mechanisms.py
$PY create_table1_demographics.py

Rscript paper/scripts/01_section1_sample_performance.R
Rscript paper/scripts/02_section2_motor_adjusted_uniqueness.R
Rscript paper/scripts/03_section3_cognitive_mechanisms.R
Rscript paper/scripts/07_cca_mechanisms_clinical.R

$PY paper/scripts/08_reference_pool_sensitivity.py
$PY paper/scripts/09_revision_statistics.py
$PY paper/scripts/10_revision_figures.py

echo "Done. Outputs are in analysis_outputs/ and paper/outputs/."
