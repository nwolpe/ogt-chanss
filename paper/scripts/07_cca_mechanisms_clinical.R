#!/usr/bin/env Rscript
# Formal multivariate test: is there mechanism-specific clinical structure beyond chance?
# Canonical correlation analysis (CCA) between the four uniqueness mechanisms (X) and a
# prespecified set of clinical measures (Y), within patients.
#   - All variables rank-transformed (Spearman-like), then site regressed out, then z-scored.
#   - Significance by permutation of Y rows *within site* (10,000 permutations).
#   - Q1 omnibus: Wilks' lambda over all dimensions, and sum of squared X-Y correlations.
#   - Q2 dimensionality: Wilks' lambda for dimensions k..K (sequential permutation test).
#   - Q3 beyond uniqueness: same CCA after also regressing residualised uniqueness out of X and Y.
#   - Stability: bootstrap of structure correlations (loadings) for significant dimensions.
#   - Sensitivity: excluding China; without AMI (keeps n higher).
# Run from the OGT repository root after scripts 02 and 04:
#   Rscript paper/scripts/07_cca_mechanisms_clinical.R
# AMI is scored from the raw items (already apathy-scored: higher = more apathy).

if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a  # compatibility with older R/ggplot2
suppressPackageStartupMessages({ library(dplyr); library(readr); library(tidyr) })
set.seed(20261016)
N_PERM <- 10000; N_BOOT <- 2000
out_dir <- file.path("paper", "outputs", "cca")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
nn <- function(x) suppressWarnings(as.numeric(x))
sum_obs <- function(m) { o <- rowSums(m, na.rm = TRUE); o[rowSums(!is.na(m)) == 0] <- NA_real_; o }
mean_obs <- function(m) { o <- rowMeans(m, na.rm = TRUE); o[is.nan(o)] <- NA_real_; o }
items <- function(df, prefix, idx) as.data.frame(lapply(paste0(prefix, idx), function(c) nn(df[[c]])))

# ---------------------------------------------------------------- data
s2 <- read_csv(file.path("paper", "outputs", "section2", "section2_participant_metrics.csv"),
               show_col_types = FALSE, guess_max = 10000) %>%
  filter(group == "patient") %>% mutate(across(c(group, participant_id, source_id), as.character)) %>%
  select(group, participant_id, source_id, source_platform, uniq = test_uniqueness_patient_motor_residualised,
         motor = control_mean_speed_px_per_s, nroutes = test_route_count)
mech <- read_csv(file.path("analysis_outputs", "phaseB_exploration_dynamics",
                           "phaseB_exploration_dynamics_participant_metrics.csv"), show_col_types = FALSE, guess_max = 10000) %>%
  mutate(across(c(group, participant_id, source_id), as.character)) %>%
  transmute(group, participant_id, source_id, Roaming = roaming_entropy_normalised, Elaboration = submovements_mean_per_path,
            PerseverationRatio = perseveration_ratio, Novelty = temporal_prefix_uniqueness_spearman_rho)
raw <- read_csv("CHANSS_clinical_demog.csv", show_col_types = FALSE, name_repair = "unique_quiet")
clin <- tibble(
  ID = toupper(trimws(raw$ID_subject)), age = nn(raw$Age), male = nn(raw$Gender),
  Anhedonia = sum_obs(items(raw, "BNSS_", 1:3)), Distress_lack = nn(raw$BNSS_4),
  Asociality = sum_obs(items(raw, "BNSS_", 5:6)), Avolition = sum_obs(items(raw, "BNSS_", 7:8)),
  BluntedAffect = sum_obs(items(raw, "BNSS_", 9:11)), Alogia = sum_obs(items(raw, "BNSS_", 12:13)),
  AMI_beh = mean_obs(items(raw, "AMI_", c(5, 9, 10, 11, 12, 15))), AMI_soc = mean_obs(items(raw, "AMI_", c(2, 3, 4, 8, 14, 17))),
  AMI_emo = mean_obs(items(raw, "AMI_", c(1, 6, 7, 13, 16, 18))),
  CDS = sum_obs(items(raw, "CDS_", 1:9)), VerbalFluency = nn(raw$BACS_VF_t), BACS_composite = nn(raw$BACS_comp_t),
  PSP = nn(raw$PSP_score)
) %>% distinct(ID, .keep_all = TRUE)
co <- read_csv(file.path("analysis_outputs", "ogt_patients_missing_clinical_data_corrected.csv"), show_col_types = FALSE)
d0 <- s2 %>% left_join(mech, by = c("group", "participant_id", "source_id")) %>%
  mutate(pid = toupper(trimws(participant_id)),
         cid = coalesce(toupper(trimws(co$participant_id_corrected))[match(pid, toupper(trimws(co$participant_id)))], pid),
         site = case_when(source_platform == "OGT_Mandarin" | grepl("^CH", cid) ~ "China",
                          source_platform == "OGT_Spanish" | grepl("^C[378]", cid) ~ "Spain", TRUE ~ "UK")) %>%
  left_join(clin, by = c("cid" = "ID"))

Xv <- c("Roaming", "Elaboration", "PerseverationRatio", "Novelty")
Yv_full <- c("Anhedonia", "Asociality", "Avolition", "BluntedAffect", "Alogia", "Distress_lack",
             "AMI_beh", "AMI_soc", "AMI_emo", "CDS", "VerbalFluency", "BACS_composite", "PSP")

# ---------------------------------------------------------------- helpers
prep <- function(d, vars, covars) {
  # rank -> residualise on covariates -> z-score
  sapply(vars, function(v) {
    r <- rank(d[[v]])
    f <- if (length(covars)) reformulate(covars, "r") else r ~ 1
    as.numeric(scale(resid(lm(f, data = cbind(d, r = r)))))
  })
}
cca_stats <- function(X, Y) {
  cc <- cancor(X, Y, xcenter = FALSE, ycenter = FALSE)$cor
  K <- length(cc)
  wilks <- sapply(seq_len(K), function(k) prod(1 - cc[k:K]^2))
  list(cor = cc, wilks = wilks, ssr = sum(cor(X, Y)^2))
}
perm_within <- function(site) { idx <- seq_along(site)
  for (s in unique(site)) { w <- which(site == s); idx[w] <- w[sample.int(length(w))] }; idx }

run_cca <- function(d, Yv, label, partial_uniq = FALSE, extra = character()) {
  d <- d %>% filter(if_all(all_of(c(Xv, Yv, if (partial_uniq) "uniq", extra)), is.finite))
  covars <- if (length(unique(d$site)) > 1) "site" else character()
  if (partial_uniq) { d$uniq_r <- rank(d$uniq); covars <- c(covars, "uniq_r") }
  for (e in extra) { d[[paste0(e, "_r")]] <- rank(d[[e]]); covars <- c(covars, paste0(e, "_r")) }
  X <- prep(d, Xv, covars); Y <- prep(d, Yv, covars)
  obs <- cca_stats(X, Y); K <- length(obs$cor)
  perm_w <- matrix(NA_real_, N_PERM, K); perm_ssr <- numeric(N_PERM)
  for (i in seq_len(N_PERM)) {
    s <- cca_stats(X, Y[perm_within(d$site), , drop = FALSE])
    perm_w[i, ] <- s$wilks; perm_ssr[i] <- s$ssr
  }
  dims <- tibble(analysis = label, n = nrow(d), n_Y = length(Yv), dimension = seq_len(K),
                 canonical_r = obs$cor, wilks_k_to_K = obs$wilks,
                 p_perm = sapply(seq_len(K), function(k) (1 + sum(perm_w[, k] <= obs$wilks[k])) / (N_PERM + 1)))
  omni <- tibble(analysis = label, n = nrow(d), sum_sq_corr = obs$ssr,
                 sum_sq_corr_null_mean = mean(perm_ssr),
                 p_perm_sum_sq_corr = (1 + sum(perm_ssr >= obs$ssr)) / (N_PERM + 1),
                 p_perm_wilks_all = dims$p_perm[1])
  # structure correlations (loadings) and bootstrap CIs
  cc <- cancor(X, Y, xcenter = FALSE, ycenter = FALSE)
  U <- X %*% cc$xcoef[, seq_len(K), drop = FALSE]; V <- Y %*% cc$ycoef[, seq_len(K), drop = FALSE]
  load_x <- cor(X, U); load_y <- cor(Y, V)
  boot_load <- function(k) {
    ref_x <- load_x[, k]; ref_y <- load_y[, k]
    b <- replicate(N_BOOT, {
      i <- sample.int(nrow(X), replace = TRUE)
      cb <- cancor(X[i, ], Y[i, ], xcenter = FALSE, ycenter = FALSE)
      u <- X[i, ] %*% cb$xcoef[, k]; v <- Y[i, ] %*% cb$ycoef[, k]
      lx <- cor(X[i, ], u)[, 1]; ly <- cor(Y[i, ], v)[, 1]
      sgn <- sign(sum(lx * ref_x) + sum(ly * ref_y)); c(lx, ly) * ifelse(sgn == 0, 1, sgn)  # align sign
    })
    tibble(variable = c(Xv, Yv), side = c(rep("mechanism", length(Xv)), rep("clinical", length(Yv))),
           loading = c(ref_x, ref_y), ci_low = apply(b, 1, quantile, 0.025), ci_high = apply(b, 1, quantile, 0.975))
  }
  loads <- bind_rows(lapply(seq_len(min(2, K)), function(k) boot_load(k) %>% mutate(dimension = k))) %>%
    mutate(analysis = label, .before = 1)
  list(dims = dims, omni = omni, loads = loads)
}

analyses <- list(
  list(label = "Main: all patients, site-adjusted", d = d0, Yv = Yv_full, pu = FALSE),
  list(label = "Beyond uniqueness: also partialling residualised uniqueness", d = d0, Yv = Yv_full, pu = TRUE),
  list(label = "Sensitivity: excluding China", d = d0 %>% filter(site != "China"), Yv = Yv_full, pu = FALSE),
  list(label = "Sensitivity: without AMI (larger n)", d = d0, Yv = setdiff(Yv_full, c("AMI_beh", "AMI_soc", "AMI_emo")), pu = FALSE),
  list(label = "Beyond uniqueness, excluding China", d = d0 %>% filter(site != "China"), Yv = Yv_full, pu = TRUE),
  list(label = "Beyond uniqueness, without AMI (larger n)", d = d0, Yv = setdiff(Yv_full, c("AMI_beh", "AMI_soc", "AMI_emo")), pu = TRUE),
  list(label = "Beyond uniqueness + motor speed, route count, age, sex", d = d0, Yv = Yv_full, pu = TRUE, extra = c("motor", "nroutes", "age", "male")),
  list(label = "Beyond uniqueness, UK only", d = d0 %>% filter(site == "UK"), Yv = Yv_full, pu = TRUE)
)
res <- lapply(analyses, function(a) { message("Running: ", a$label); run_cca(a$d, a$Yv, a$label, a$pu, if (is.null(a$extra)) character() else a$extra) })
dims <- bind_rows(lapply(res, `[[`, "dims")); omni <- bind_rows(lapply(res, `[[`, "omni")); loads <- bind_rows(lapply(res, `[[`, "loads"))
write_csv(dims, file.path(out_dir, "cca_dimensions.csv"))
write_csv(omni, file.path(out_dir, "cca_omnibus.csv"))
write_csv(loads, file.path(out_dir, "cca_loadings_bootstrap.csv"))
message("Wrote CCA outputs to ", out_dir)
