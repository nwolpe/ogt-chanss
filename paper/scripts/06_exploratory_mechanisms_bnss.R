#!/usr/bin/env Rscript
# Exploratory: do the four uniqueness mechanisms ("paths" to reduced uniqueness) map onto
# distinct negative-symptom domains (BNSS), self-reported apathy (AMI, raw = apathy-scored),
# depression (CDS) or cognition (BACS)?
# Run from the OGT repository root after scripts 02 and 04:
#   Rscript paper/scripts/06_exploratory_mechanisms_bnss.R
# Writes to paper/outputs/exploratory_bnss/.

if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a  # compatibility with older R/ggplot2
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})

out_dir <- file.path("paper", "outputs", "exploratory_bnss")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
nn <- function(x) suppressWarnings(as.numeric(x))
sum_obs <- function(m) { o <- rowSums(m, na.rm = TRUE); o[rowSums(!is.na(m)) == 0] <- NA_real_; o }
mean_obs <- function(m) { o <- rowMeans(m, na.rm = TRUE); o[is.nan(o)] <- NA_real_; o }
items <- function(df, prefix, idx) as.data.frame(lapply(paste0(prefix, idx), function(c) nn(df[[c]])))

# ---------------------------------------------------------------- data
s2 <- read_csv(file.path("paper", "outputs", "section2", "section2_participant_metrics.csv"),
               show_col_types = FALSE, guess_max = 10000) %>%
  filter(group == "patient") %>%
  mutate(across(c(group, participant_id, source_id), as.character)) %>%
  select(group, participant_id, source_id, source_platform,
         uniqueness_resid = test_uniqueness_patient_motor_residualised)
mech <- read_csv(file.path("analysis_outputs", "phaseB_exploration_dynamics",
                           "phaseB_exploration_dynamics_participant_metrics.csv"),
                 show_col_types = FALSE, guess_max = 10000) %>%
  mutate(across(c(group, participant_id, source_id), as.character)) %>%
  transmute(group, participant_id, source_id,
            roaming = roaming_entropy_normalised,
            elaboration = submovements_mean_per_path,
            consecutive_distinctness = perseveration_ratio,   # higher = LESS perseverative
            novelty = temporal_prefix_uniqueness_spearman_rho)
memb <- read_csv(file.path("paper", "outputs", "section4", "section4_kmeans_memberships.csv"),
                 show_col_types = FALSE, guess_max = 10000) %>%
  mutate(across(c(group, participant_id, source_id), as.character)) %>%
  select(group, participant_id, source_id, kmeans_cluster)

raw <- read_csv("CHANSS_clinical_demog.csv", show_col_types = FALSE, name_repair = "unique_quiet")
if (!"BACS_ToL_t" %in% names(raw)) raw$BACS_ToL_t <- raw[[grep("^BACS_ToL_t", names(raw))[1]]]
clin <- tibble(
  ID = toupper(trimws(raw$ID_subject)),
  # BNSS domains (Kirkpatrick 2011; Strauss five-factor structure)
  BNSS_anhedonia = sum_obs(items(raw, "BNSS_", 1:3)),
  BNSS_distress = nn(raw$BNSS_4),
  BNSS_asociality = sum_obs(items(raw, "BNSS_", 5:6)),
  BNSS_avolition = sum_obs(items(raw, "BNSS_", 7:8)),
  BNSS_blunted_affect = sum_obs(items(raw, "BNSS_", 9:11)),
  BNSS_alogia = sum_obs(items(raw, "BNSS_", 12:13)),
  BNSS_MAP = sum_obs(items(raw, "BNSS_", c(1, 2, 3, 5, 6, 7, 8))),
  BNSS_EXP = sum_obs(items(raw, "BNSS_", 9:13)),
  # AMI, raw items (already apathy-scored: higher = more apathy)
  AMI_behavioural = mean_obs(items(raw, "AMI_", c(5, 9, 10, 11, 12, 15))),
  AMI_social = mean_obs(items(raw, "AMI_", c(2, 3, 4, 8, 14, 17))),
  AMI_emotional = mean_obs(items(raw, "AMI_", c(1, 6, 7, 13, 16, 18))),
  CDS_total = sum_obs(items(raw, "CDS_", 1:9)),
  BACS_VF_t = nn(raw$BACS_VF_t), BACS_comp_t = nn(raw$BACS_comp_t),
  PSP_score = nn(raw$PSP_score)
)
for (i in 1:13) clin[[paste0("BNSS_item", i)]] <- nn(raw[[paste0("BNSS_", i)]])
clin <- clin %>% distinct(ID, .keep_all = TRUE)

co <- read_csv(file.path("analysis_outputs", "ogt_patients_missing_clinical_data_corrected.csv"), show_col_types = FALSE)
d <- s2 %>%
  left_join(mech, by = c("group", "participant_id", "source_id")) %>%
  left_join(memb, by = c("group", "participant_id", "source_id")) %>%
  mutate(pid = toupper(trimws(participant_id)),
         cid = coalesce(toupper(trimws(co$participant_id_corrected))[match(pid, toupper(trimws(co$participant_id)))], pid),
         site = case_when(source_platform == "OGT_Mandarin" | grepl("^CH", cid) ~ "China",
                          source_platform == "OGT_Spanish" | grepl("^C[378]", cid) ~ "Spain", TRUE ~ "UK")) %>%
  left_join(clin, by = c("cid" = "ID"))

task_vars <- c(uniqueness_resid = "Residualised uniqueness", roaming = "Roaming entropy",
               elaboration = "Option elaboration", consecutive_distinctness = "Perseveration ratio (high = less perseverative)",
               novelty = "Novelty maintenance")
domain_vars <- c("BNSS_anhedonia", "BNSS_asociality", "BNSS_avolition", "BNSS_blunted_affect", "BNSS_alogia",
                 "BNSS_distress", "BNSS_MAP", "BNSS_EXP", "AMI_behavioural", "AMI_social", "AMI_emotional",
                 "CDS_total", "BACS_VF_t", "BACS_comp_t", "PSP_score")
item_vars <- paste0("BNSS_item", 1:13)

spear <- function(x, y) { ok <- is.finite(x) & is.finite(y)
  ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman", exact = FALSE))
  c(n = sum(ok), rho = unname(ct$estimate), p = ct$p.value) }
pspear <- function(x, y, s) { ok <- is.finite(x) & is.finite(y); s <- factor(s[ok])
  rx <- resid(lm(rank(x[ok]) ~ s)); ry <- resid(lm(rank(y[ok]) ~ s)); r <- cor(rx, ry)
  df <- sum(ok) - 2 - (nlevels(s) - 1); t <- r * sqrt(df / (1 - r^2))
  c(n = sum(ok), rho = r, p = 2 * pt(-abs(t), df)) }

run_grid <- function(dat, yvars, label) {
  bind_rows(lapply(names(task_vars), function(tv) bind_rows(lapply(yvars, function(cv) {
    a <- spear(dat[[tv]], dat[[cv]]); b <- pspear(dat[[tv]], dat[[cv]], dat$site)
    tibble(sample = label, task = task_vars[[tv]], clinical = cv, n = a[["n"]],
           rho = a[["rho"]], p = a[["p"]], rho_site_adj = b[["rho"]], p_site_adj = b[["p"]])
  }))))
}
grid <- bind_rows(run_grid(d, domain_vars, "All patients"),
                  run_grid(d %>% filter(site != "China"), domain_vars, "Excluding China")) %>%
  group_by(sample) %>%
  mutate(q = p.adjust(p, "BH"), q_site_adj = p.adjust(p_site_adj, "BH")) %>% ungroup()
write_csv(grid, file.path(out_dir, "mechanism_by_clinical_domain.csv"))

items_grid <- run_grid(d, item_vars, "All patients") %>%
  mutate(q = p.adjust(p, "BH"), q_site_adj = p.adjust(p_site_adj, "BH"))
write_csv(items_grid, file.path(out_dir, "mechanism_by_bnss_item.csv"))

# Clusters vs BNSS domains (raw and site-adjusted Kruskal-Wallis)
kw <- bind_rows(lapply(domain_vars, function(v) {
  dd <- d %>% filter(is.finite(.data[[v]]), !is.na(kmeans_cluster))
  k1 <- kruskal.test(dd[[v]], factor(dd$kmeans_cluster))
  k2 <- kruskal.test(resid(lm(rank(dd[[v]]) ~ factor(dd$site))), factor(dd$kmeans_cluster))
  means <- tapply(dd[[v]], dd$kmeans_cluster, mean)
  tibble(clinical = v, n = nrow(dd), mean_c1 = means[["1"]], mean_c2 = means[["2"]], mean_c3 = means[["3"]],
         H = unname(k1$statistic), p = k1$p.value, H_site_adj = unname(k2$statistic), p_site_adj = k2$p.value)
})) %>% mutate(q = p.adjust(p, "BH"), q_site_adj = p.adjust(p_site_adj, "BH"))
write_csv(kw, file.path(out_dir, "clusters_by_clinical_domain.csv"))

site_means <- d %>% group_by(site) %>%
  summarise(n = n(), across(c(all_of(names(task_vars)), all_of(domain_vars[1:8])), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
write_csv(site_means, file.path(out_dir, "site_means.csv"))
message("Wrote exploratory outputs to ", out_dir)
