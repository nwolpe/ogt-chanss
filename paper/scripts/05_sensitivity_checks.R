#!/usr/bin/env Rscript
# Sensitivity checks for the OGT paper (code review, Oct 2026).
# Run from the OGT repository root AFTER scripts 02 and 04:
#   Rscript paper/scripts/05_sensitivity_checks.R
# Writes to paper/outputs/sensitivity/. Does not modify any other output.
#
# Checks:
#   A. AMI scoring direction (raw items vs the 4 - item reversal used in 01/02/04/04b)
#   B. Patient clinical correlations: as reported, pooled residualiser, AMI corrected,
#      site-adjusted (partial Spearman), and excluding China; BH-FDR within each set
#   C. Group difference in uniqueness by patient platform / coordinate frame
#   D. Mechanism attenuation on a common complete-case sample
#   E. K-means clusters: site composition, site-adjusted clinical tests, re-clustering without China

if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a  # compatibility with older R/ggplot2
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(readxl)
  library(tidyr)
})

out_dir <- file.path("paper", "outputs", "sensitivity")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

to_num <- function(x) suppressWarnings(as.numeric(x))
sum_obs <- function(df) { o <- rowSums(df, na.rm = TRUE); o[rowSums(!is.na(df)) == 0] <- NA_real_; o }
row_mean <- function(df) { o <- rowMeans(df, na.rm = TRUE); o[is.nan(o)] <- NA_real_; o }

# ---------------------------------------------------------------- data
s2 <- read_csv(file.path("paper", "outputs", "section2", "section2_participant_metrics.csv"),
               show_col_types = FALSE, guess_max = 10000) %>%
  mutate(across(c(group, participant_id, source_id), as.character))
calib <- read_csv(file.path("paper", "outputs", "shared", "paper_coordinate_calibration.csv"),
                  show_col_types = FALSE) %>%
  mutate(across(c(group, participant_id, source_id), as.character)) %>%
  transmute(group, participant_id, source_id,
            anisotropy = calibration_y_scale / calibration_x_scale)
mech <- read_csv(file.path("analysis_outputs", "phaseB_exploration_dynamics",
                           "phaseB_exploration_dynamics_participant_metrics.csv"),
                 show_col_types = FALSE, guess_max = 10000) %>%
  mutate(across(c(group, participant_id, source_id), as.character)) %>%
  select(group, participant_id, source_id, roaming_entropy_normalised, submovements_mean_per_path,
         perseveration_ratio, temporal_prefix_uniqueness_spearman_rho)
mech_cols <- c("roaming_entropy_normalised", "submovements_mean_per_path",
               "perseveration_ratio", "temporal_prefix_uniqueness_spearman_rho")

raw_clin <- read_csv("CHANSS_clinical_demog.csv", show_col_types = FALSE, name_repair = "unique_quiet")
# The file has two "BACS_ToL_t" columns (T-score, then z-score); keep the first, as script 02 does.
if (!"BACS_ToL_t" %in% names(raw_clin)) raw_clin$BACS_ToL_t <- raw_clin[[grep("^BACS_ToL_t", names(raw_clin))[1]]]
ami <- function(items, reverse) {
  m <- as.data.frame(lapply(paste0("AMI_", items), function(c) to_num(raw_clin[[c]])))
  if (reverse) m <- 4 - m
  row_mean(m)
}
clin <- raw_clin %>%
  transmute(
    ID_subject = toupper(trimws(ID_subject)),
    MAP = sum_obs(across(c(BNSS_1, BNSS_2, BNSS_3, BNSS_5, BNSS_6, BNSS_7, BNSS_8), to_num)),
    EXP = sum_obs(across(c(BNSS_9, BNSS_10, BNSS_11, BNSS_12, BNSS_13), to_num)),
    CDS_total = sum_obs(across(paste0("CDS_", 1:9), to_num)),
    PANSS_pos = sum_obs(across(c(PANSS_p1, PANSS_p3, PANSS_pg9, PANSS_pg10), to_num)),
    PSP_score = to_num(PSP_score),
    BACS_VM_t = to_num(BACS_VM_t), BACS_DS_t = to_num(BACS_DS_t), BACS_TMT_t = to_num(BACS_TMT_t),
    BACS_VF_t = to_num(BACS_VF_t), BACS_SC_t = to_num(BACS_SC_t), BACS_ToL_t = to_num(BACS_ToL_t),
    AMI_beh_as_coded = ami(c(5, 9, 10, 11, 12, 15), TRUE),
    AMI_soc_as_coded = ami(c(2, 3, 4, 8, 14, 17), TRUE),
    AMI_emo_as_coded = ami(c(1, 6, 7, 13, 16, 18), TRUE),
    AMI_beh_raw = ami(c(5, 9, 10, 11, 12, 15), FALSE),
    AMI_soc_raw = ami(c(2, 3, 4, 8, 14, 17), FALSE),
    AMI_emo_raw = ami(c(1, 6, 7, 13, 16, 18), FALSE)
  ) %>%
  distinct(ID_subject, .keep_all = TRUE)

corr_path <- file.path("analysis_outputs", "ogt_patients_missing_clinical_data_corrected.csv")
corrections <- read_csv(corr_path, show_col_types = FALSE) %>%
  transmute(pid = toupper(trimws(participant_id)), cid = toupper(trimws(participant_id_corrected)))

rows <- s2 %>%
  left_join(calib, by = c("group", "participant_id", "source_id")) %>%
  left_join(mech, by = c("group", "participant_id", "source_id")) %>%
  mutate(
    pid = toupper(trimws(participant_id)),
    cid = coalesce(corrections$cid[match(pid, corrections$pid)], pid),
    site = case_when(
      group == "control" ~ "Control",
      source_platform == "OGT_Mandarin" | grepl("^CH", cid) ~ "China",
      source_platform == "OGT_Spanish" ~ "Spain",
      TRUE ~ "UK"
    ),
    same_frame_as_controls = anisotropy < 1.3
  )
pat <- rows %>% filter(group == "patient") %>% left_join(clin, by = c("cid" = "ID_subject"))

# ---------------------------------------------------------------- A. AMI direction
validity <- bind_rows(lapply(c("AMI_beh", "AMI_soc", "AMI_emo"), function(a) {
  bind_rows(lapply(c("MAP", "CDS_total", "PSP_score"), function(y) {
    bind_rows(lapply(c("as_coded", "raw"), function(v) {
      x <- pat[[paste0(a, "_", v)]]; yy <- pat[[y]]; ok <- is.finite(x) & is.finite(yy)
      ct <- suppressWarnings(cor.test(x[ok], yy[ok], method = "spearman", exact = FALSE))
      tibble(ami_subscale = a, scoring = v, versus = y, n = sum(ok),
             rho = unname(ct$estimate), p = ct$p.value, mean_ami = mean(x, na.rm = TRUE))
    }))
  }))
}))
write_csv(validity, file.path(out_dir, "A_ami_direction_validity.csv"))

# ---------------------------------------------------------------- B. clinical correlations
clin_vars <- c(EXP = "BNSS EXP", MAP = "BNSS MAP",
               AMI_beh_as_coded = "AMI behavioural (as coded, 4 - item)",
               AMI_beh_raw = "AMI behavioural (raw = apathy)",
               AMI_soc_raw = "AMI social (raw)", AMI_emo_raw = "AMI emotional (raw)",
               PSP_score = "PSP", BACS_VM_t = "BACS verbal memory", BACS_DS_t = "BACS digit sequencing",
               BACS_TMT_t = "BACS token motor", BACS_VF_t = "BACS verbal fluency",
               BACS_SC_t = "BACS symbol coding", BACS_ToL_t = "BACS Tower of London",
               CDS_total = "CDS total", PANSS_pos = "PANSS positive (P1,P3,G9,G10)")

partial_spearman <- function(x, y, site) {
  ok <- is.finite(x) & is.finite(y) & !is.na(site)
  x <- x[ok]; y <- y[ok]; site <- factor(site[ok])
  rx <- resid(lm(rank(x) ~ site)); ry <- resid(lm(rank(y) ~ site))
  r <- cor(rx, ry)
  df <- length(x) - 2 - (nlevels(site) - 1)
  t <- r * sqrt(df / (1 - r^2))
  c(n = length(x), rho = r, p = 2 * pt(-abs(t), df))
}
spear <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman", exact = FALSE))
  c(n = sum(ok), rho = unname(ct$estimate), p = ct$p.value)
}

corr_sets <- list(
  "1_as_reported_patient_residual" = function(y) spear(pat$test_uniqueness_patient_motor_residualised, pat[[y]]),
  "2_pooled_residual_per_methods" = function(y) spear(pat$test_uniqueness_motor_residualised, pat[[y]]),
  "3_site_adjusted_partial" = function(y) partial_spearman(pat$test_uniqueness_patient_motor_residualised, pat[[y]], pat$site),
  "4_excluding_China" = function(y) { k <- pat$site != "China"; spear(pat$test_uniqueness_patient_motor_residualised[k], pat[[y]][k]) },
  "5_motor_speed_as_reported" = function(y) spear(pat$control_mean_speed_px_per_s, pat[[y]]),
  "6_motor_speed_site_adjusted" = function(y) partial_spearman(pat$control_mean_speed_px_per_s, pat[[y]], pat$site)
)
corr_out <- bind_rows(lapply(names(corr_sets), function(set) {
  res <- bind_rows(lapply(names(clin_vars), function(v) {
    r <- corr_sets[[set]](v)
    tibble(analysis = set, measure = clin_vars[[v]], variable = v, n = r[["n"]], rho = r[["rho"]], p = r[["p"]])
  }))
  # FDR over the 14 measures actually used in the paper (exclude the duplicate as-coded AMI row)
  fam <- res$variable != "AMI_beh_as_coded"
  res$q_bh <- NA_real_
  res$q_bh[fam] <- p.adjust(res$p[fam], "BH")
  res
}))
write_csv(corr_out, file.path(out_dir, "B_clinical_correlations_sensitivity.csv"))

site_means <- pat %>% group_by(site) %>%
  summarise(n = n(),
            resid_uniqueness = mean(test_uniqueness_patient_motor_residualised, na.rm = TRUE),
            raw_uniqueness = mean(test_uniqueness_mean, na.rm = TRUE),
            CDS_total = mean(CDS_total, na.rm = TRUE), BACS_VF_t = mean(BACS_VF_t, na.rm = TRUE),
            AMI_beh_raw = mean(AMI_beh_raw, na.rm = TRUE), MAP = mean(MAP, na.rm = TRUE), .groups = "drop")
write_csv(site_means, file.path(out_dir, "B_site_means.csv"))

# ---------------------------------------------------------------- C. group difference by platform
ctl <- rows %>% filter(group == "control")
welch <- function(a, b) { tt <- t.test(a, b); c(t = unname(tt$statistic), df = unname(tt$parameter), p = tt$p.value) }
subsets <- list(
  "All patients" = pat,
  "Same coordinate frame as controls" = pat %>% filter(same_frame_as_controls),
  "Rescaled frame (OCTAL/Mandarin-type)" = pat %>% filter(!same_frame_as_controls),
  "Excluding China" = pat %>% filter(site != "China"),
  "UK" = pat %>% filter(site == "UK"), "Spain" = pat %>% filter(site == "Spain"),
  "China" = pat %>% filter(site == "China")
)
grp_out <- bind_rows(lapply(names(subsets), function(nm) {
  p <- subsets[[nm]]
  bind_rows(lapply(c("test_uniqueness_mean", "test_uniqueness_motor_residualised"), function(m) {
    w <- welch(ctl[[m]], p[[m]])
    tibble(patient_subset = nm, metric = m, n_patients = sum(is.finite(p[[m]])),
           mean_controls = mean(ctl[[m]], na.rm = TRUE), mean_patients = mean(p[[m]], na.rm = TRUE),
           ratio_patient_to_control = mean(p[[m]], na.rm = TRUE) / mean(ctl[[m]], na.rm = TRUE),
           welch_t = w[["t"]], df = w[["df"]], p = w[["p"]])
  }))
}))
write_csv(grp_out, file.path(out_dir, "C_group_difference_by_subset.csv"))

# ---------------------------------------------------------------- D. attenuation, common sample
att_df <- rows %>% mutate(group_patient = as.integer(group == "patient")) %>%
  filter(if_all(all_of(c("test_uniqueness_mean", "control_mean_speed_px_per_s", mech_cols)), is.finite))
att_fit <- function(f) { m <- lm(f, data = att_df); s <- summary(m)$coefficients
  c(beta = s["group_patient", 1], se = s["group_patient", 2], p = s["group_patient", 4], r2 = summary(m)$r.squared) }
forms <- list("Group only" = test_uniqueness_mean ~ group_patient,
              "Group + motor" = test_uniqueness_mean ~ group_patient + control_mean_speed_px_per_s,
              "Group + motor + mechanisms" = reformulate(c("group_patient", "control_mean_speed_px_per_s", mech_cols), "test_uniqueness_mean"))
att <- bind_rows(lapply(names(forms), function(nm) { r <- att_fit(forms[[nm]])
  tibble(model = nm, n = nrow(att_df), group_beta = r[["beta"]], se = r[["se"]], p = r[["p"]], r2 = r[["r2"]]) })) %>%
  mutate(attenuation_pct = 100 * (1 - group_beta / group_beta[1]))
write_csv(att, file.path(out_dir, "D_attenuation_common_sample.csv"))

# ---------------------------------------------------------------- E. clusters
memb <- read_csv(file.path("paper", "outputs", "section4", "section4_kmeans_memberships.csv"),
                 show_col_types = FALSE, guess_max = 10000) %>%
  mutate(across(c(group, participant_id, source_id), as.character)) %>%
  select(group, participant_id, source_id, kmeans_cluster)
pc <- pat %>% inner_join(memb, by = c("group", "participant_id", "source_id")) %>%
  mutate(kmeans_cluster = factor(kmeans_cluster))
write_csv(as.data.frame.matrix(table(pc$kmeans_cluster, pc$site)) %>% tibble::rownames_to_column("cluster"),
          file.path(out_dir, "E_cluster_by_site.csv"))

mech_profile <- pc %>% group_by(kmeans_cluster) %>%
  summarise(n = n(), across(all_of(mech_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
write_csv(mech_profile, file.path(out_dir, "E_cluster_raw_mechanism_means.csv"))

kw_vars <- c("test_uniqueness_motor_residualised", "AMI_beh_as_coded", "AMI_beh_raw", "BACS_VF_t", "CDS_total")
kw <- function(d, v, adjust_site) {
  y <- d[[v]]; ok <- is.finite(y)
  d <- d[ok, ]; y <- y[ok]
  if (adjust_site) y <- resid(lm(rank(y) ~ factor(d$site)))
  k <- kruskal.test(y, d$kmeans_cluster)
  tibble(variable = v, n = length(y), H = unname(k$statistic), p = k$p.value)
}
kw_out <- bind_rows(
  bind_rows(lapply(kw_vars, function(v) kw(pc, v, FALSE))) %>% mutate(analysis = "as reported (all patients)"),
  bind_rows(lapply(kw_vars, function(v) kw(pc, v, TRUE))) %>% mutate(analysis = "site-adjusted ranks"),
  bind_rows(lapply(kw_vars, function(v) kw(pc %>% filter(site != "China"), v, FALSE))) %>% mutate(analysis = "excluding China")
)
write_csv(kw_out, file.path(out_dir, "E_cluster_clinical_tests_sensitivity.csv"))

# Re-cluster without China (same procedure as script 04)
nc <- pat %>% filter(site != "China") %>% filter(if_all(all_of(mech_cols), is.finite))
Z <- scale(as.matrix(nc[, mech_cols]))
sil <- bind_rows(lapply(2:6, function(k) { set.seed(42); km <- kmeans(Z, k, nstart = 100)
  tibble(k = k, silhouette = mean(cluster::silhouette(km$cluster, dist(Z))[, "sil_width"])) }))
best_k <- sil$k[which.max(sil$silhouette)]
set.seed(42); km <- kmeans(Z, best_k, nstart = 500)
nc$cluster_noCN <- factor(km$cluster)
prof_nc <- nc %>% group_by(cluster_noCN) %>%
  summarise(n = n(), across(all_of(mech_cols), ~ mean(.x, na.rm = TRUE)),
            resid_uniqueness = mean(test_uniqueness_motor_residualised, na.rm = TRUE),
            CDS_total = mean(CDS_total, na.rm = TRUE), BACS_VF_t = mean(BACS_VF_t, na.rm = TRUE),
            AMI_beh_raw = mean(AMI_beh_raw, na.rm = TRUE), .groups = "drop")
kw_nc <- bind_rows(lapply(c("test_uniqueness_motor_residualised", "AMI_beh_raw", "BACS_VF_t", "CDS_total"), function(v) {
  ok <- is.finite(nc[[v]]); k <- kruskal.test(nc[[v]][ok], nc$cluster_noCN[ok])
  tibble(variable = v, n = sum(ok), H = unname(k$statistic), p = k$p.value) }))
write_csv(sil, file.path(out_dir, "E_recluster_noChina_silhouette.csv"))
write_csv(prof_nc, file.path(out_dir, "E_recluster_noChina_profiles.csv"))
write_csv(kw_nc, file.path(out_dir, "E_recluster_noChina_clinical_tests.csv"))

message("Wrote sensitivity outputs to ", out_dir)
