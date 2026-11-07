#!/usr/bin/env Rscript
# Patient scores on the first canonical dimension (CCA of script 07).
# Question: is the first canonical correlation carried evenly across patients (a continuum),
# or by a small group at one end (a tail)?
#   - Per-patient scores on the task variate (U1) and clinical variate (V1), same data
#     preparation as script 07 (ranks, residualised on site [+ uniqueness], z-scored).
#   - Figure S3: U1 vs V1 coloured by site, with marginal histograms; out-of-sample
#     (leave-one-out) scores alongside.
#   - Diagnostics: per-site r, Spearman r, leave-one-out refits, trimming the most extreme
#     patients, share of the covariance carried by the top 10% of patients compared with a
#     bivariate-normal benchmark, distribution shape (skewness, dip test of unimodality).
# Run from the repository root after script 07:  Rscript paper/scripts/07b_cca_canonical_scores.R

if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr); library(ggplot2); library(patchwork); library(diptest)
})
set.seed(20261106)
out_dir <- file.path("paper", "outputs", "cca")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
nn <- function(x) suppressWarnings(as.numeric(x))
sum_obs <- function(m) { o <- rowSums(m, na.rm = TRUE); o[rowSums(!is.na(m)) == 0] <- NA_real_; o }
mean_obs <- function(m) { o <- rowMeans(m, na.rm = TRUE); o[is.nan(o)] <- NA_real_; o }
items <- function(df, prefix, idx) as.data.frame(lapply(paste0(prefix, idx), function(c) nn(df[[c]])))

# ---------------------------------------------------------------- data (identical to script 07)
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
Yv <- c("Anhedonia", "Asociality", "Avolition", "BluntedAffect", "Alogia", "Distress_lack",
        "AMI_beh", "AMI_soc", "AMI_emo", "CDS", "VerbalFluency", "BACS_composite", "PSP")
prep <- function(d, vars, covars) sapply(vars, function(v) {
  r <- rank(d[[v]]); f <- if (length(covars)) reformulate(covars, "r") else r ~ 1
  as.numeric(scale(resid(lm(f, data = cbind(d, r = r)))))
})

# ---------------------------------------------------------------- first-dimension scores
first_dim <- function(X, Y) {
  cc <- cancor(X, Y, xcenter = FALSE, ycenter = FALSE)
  a <- cc$xcoef[, 1]; b <- cc$ycoef[, 1]
  if (cor(X[, "Roaming"], X %*% a) < 0) { a <- -a; b <- -b }   # orient: broader roaming = higher U1
  list(a = a, b = b, r = cc$cor[1])
}
skew <- function(x) { x <- x - mean(x); mean(x^3) / mean(x^2)^1.5 }
top_share <- function(u, v, q = 0.10) {   # share of sum(u*v) carried by the top q of patients
  p <- as.numeric(scale(u)) * as.numeric(scale(v)); p <- sort(p, decreasing = TRUE)
  sum(p[seq_len(ceiling(q * length(p)))]) / sum(p)
}

analyse <- function(d, label, partial_uniq) {
  d <- d %>% filter(if_all(all_of(c(Xv, Yv, if (partial_uniq) "uniq")), is.finite))
  covars <- if (length(unique(d$site)) > 1) "site" else character()
  if (partial_uniq) { d$uniq_r <- rank(d$uniq); covars <- c(covars, "uniq_r") }
  X <- prep(d, Xv, covars); Y <- prep(d, Yv, covars); n <- nrow(d)
  f <- first_dim(X, Y)
  U <- as.numeric(scale(X %*% f$a)); V <- as.numeric(scale(Y %*% f$b))

  # leave-one-out: refit without patient i -> (1) r of the refit (influence),
  # (2) out-of-sample scores for patient i from the refit weights
  loo <- t(sapply(seq_len(n), function(i) {
    g <- first_dim(X[-i, ], Y[-i, ])
    if (sum(g$a * f$a) < 0) { g$a <- -g$a; g$b <- -g$b }
    c(r_without = g$r, U_oos = sum(X[i, ] * g$a), V_oos = sum(Y[i, ] * g$b))
  }))
  # trimming: drop the patients farthest from the centre of the (U, V) cloud and refit
  dist <- sqrt(U^2 + V^2)
  trim_r <- function(q) { keep <- dist <= quantile(dist, 1 - q); first_dim(X[keep, ], Y[keep, ])$r }
  # benchmark for the top-10% share under an evenly spread (bivariate normal) association
  # and for how much r falls when the 10% most extreme patients are dropped (range restriction alone)
  trim_fixed <- function(u, v, q = 0.10) { dd <- sqrt(as.numeric(scale(u))^2 + as.numeric(scale(v))^2)
    keep <- dd <= quantile(dd, 1 - q); cor(u[keep], v[keep]) }
  bench <- replicate(5000, { z1 <- rnorm(n); z2 <- f$r * z1 + sqrt(1 - f$r^2) * rnorm(n)
    c(share = top_share(z1, z2), trim = trim_fixed(z1, z2)) })
  # influence of each patient on the in-sample correlation of the fixed variates
  infl <- sapply(seq_len(n), function(i) cor(U[-i], V[-i])) - cor(U, V)

  scores <- tibble(analysis = label, participant_id = d$participant_id, site = d$site, U1_task = U, V1_clinical = V,
                   U1_task_loo = as.numeric(scale(loo[, "U_oos"])), V1_clinical_loo = as.numeric(scale(loo[, "V_oos"])),
                   r_refit_without = loo[, "r_without"])
  per_site <- scores %>% group_by(site) %>% summarise(n = n(), r = cor(U1_task, V1_clinical), .groups = "drop")
  diag <- tibble(
    analysis = label, n = n, canonical_r = f$r, spearman_r = cor(U, V, method = "spearman"),
    r_out_of_sample_loo = cor(loo[, "U_oos"], loo[, "V_oos"]),
    r_out_of_sample_loo_spearman = cor(loo[, "U_oos"], loo[, "V_oos"], method = "spearman"),
    r_refit_min_dropping_one = min(loo[, "r_without"]), r_refit_max_dropping_one = max(loo[, "r_without"]),
    r_refit_trim_5pct = trim_r(0.05), r_refit_trim_10pct = trim_r(0.10),
    max_abs_change_r_dropping_one_fixed_variates = max(abs(infl)),
    top10pct_share_of_covariance = top_share(U, V), top10pct_share_benchmark_median = median(bench["share", ]),
    top10pct_share_benchmark_p95 = unname(quantile(bench["share", ], 0.95)),
    r_fixed_variates_trim_10pct = trim_fixed(U, V), r_trim_10pct_benchmark_median = median(bench["trim", ]),
    r_trim_10pct_benchmark_p05 = unname(quantile(bench["trim", ], 0.05)),
    skew_U1 = skew(U), skew_V1 = skew(V), dip_p_U1 = dip.test(U)$p.value, dip_p_V1 = dip.test(V)$p.value,
    shapiro_p_U1 = shapiro.test(U)$p.value, shapiro_p_V1 = shapiro.test(V)$p.value,
    per_site_r = paste(sprintf("%s %.2f (n = %d)", per_site$site, per_site$r, per_site$n), collapse = "; ")
  )
  weights <- tibble(analysis = label, variable = c(Xv, Yv), side = rep(c("task", "clinical"), c(length(Xv), length(Yv))),
                    structure_correlation = c(cor(X, U)[, 1], cor(Y, V)[, 1]))
  list(scores = scores, diag = diag, weights = weights)
}

res <- list(
  analyse(d0, "Beyond uniqueness: also partialling residualised uniqueness", TRUE),
  analyse(d0, "Main: all patients, site-adjusted", FALSE),
  analyse(d0 %>% filter(site == "UK"), "Beyond uniqueness, UK only", TRUE)
)
scores <- bind_rows(lapply(res, `[[`, "scores")); diag <- bind_rows(lapply(res, `[[`, "diag"))
weights <- bind_rows(lapply(res, `[[`, "weights"))
write_csv(scores, file.path(out_dir, "cca_dim1_patient_scores.csv"))
write_csv(diag, file.path(out_dir, "cca_dim1_distribution_diagnostics.csv"))
write_csv(weights, file.path(out_dir, "cca_dim1_structure_correlations.csv"))
print(as.data.frame(t(diag)))

# ---------------------------------------------------------------- Figure S3
site_cols <- c(UK = "#2a78d6", Spain = "#eb6834", China = "#1baf7a")
site_shapes <- c(UK = 21, Spain = 24, China = 22)
ink <- "#2b2b2a"; muted <- "#6f6e69"; grid_col <- "#e6e5e0"
theme_s <- theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = grid_col, linewidth = 0.3),
        axis.text = element_text(colour = muted), axis.title = element_text(colour = ink),
        plot.title = element_text(colour = ink, face = "bold", size = 9.5),
        plot.subtitle = element_text(colour = muted, size = 8), legend.position = "none")

panel <- function(df, ucol, vcol, title, r_txt, lab_x, lab_y) {
  df <- df %>% mutate(u = .data[[ucol]], v = .data[[vcol]], site = factor(site, levels = names(site_cols)))
  lim <- range(c(df$u, df$v)) + c(-0.2, 0.2)
  br <- seq(floor(lim[1] * 2) / 2, ceiling(lim[2] * 2) / 2, by = 0.5)
  sc <- ggplot(df, aes(u, v)) +
    geom_smooth(method = "lm", formula = y ~ x, se = TRUE, colour = ink, fill = "#d9d8d2", linewidth = 0.6) +
    geom_point(aes(fill = site, shape = site), colour = "white", stroke = 0.5, size = 2.3, alpha = 0.9) +
    scale_fill_manual(values = site_cols, name = NULL) + scale_shape_manual(values = site_shapes, name = NULL) +
    annotate("text", x = lim[1], y = lim[2], label = r_txt, hjust = 0, vjust = 1, size = 3, colour = ink) +
    coord_cartesian(xlim = lim, ylim = lim) + labs(x = lab_x, y = lab_y) + theme_s
  hx <- ggplot(df, aes(u, fill = site)) + geom_histogram(breaks = br, colour = "white", linewidth = 0.3) +
    scale_fill_manual(values = site_cols) + coord_cartesian(xlim = lim) + labs(title = title, x = NULL, y = NULL) +
    theme_s + theme(axis.text.x = element_blank(), panel.grid.major.x = element_blank())
  hy <- ggplot(df, aes(v, fill = site)) + geom_histogram(breaks = br, colour = "white", linewidth = 0.3) +
    scale_fill_manual(values = site_cols) + coord_flip(xlim = lim) + labs(x = NULL, y = NULL) +
    theme_s + theme(axis.text.y = element_blank(), panel.grid.major.y = element_blank())
  hx + plot_spacer() + sc + hy + plot_layout(ncol = 2, widths = c(4, 1), heights = c(1, 4))
}
b <- scores %>% filter(analysis == "Beyond uniqueness: also partialling residualised uniqueness")
db <- diag %>% filter(analysis == unique(b$analysis))
pA <- panel(b, "U1_task", "V1_clinical", "A  In-sample scores",
            sprintf("canonical r = %.2f\nSpearman r = %.2f", db$canonical_r, db$spearman_r),
            "Task variate (z; higher = broader roaming)", "Clinical variate (z; higher = better)")
pB <- panel(b, "U1_task_loo", "V1_clinical_loo", "B  Out-of-sample (leave-one-out) scores",
            sprintf("r = %.2f\nSpearman r = %.2f", db$r_out_of_sample_loo, db$r_out_of_sample_loo_spearman),
            "Task variate (z)", "Clinical variate (z)")
legend_df <- tibble(site = factor(names(site_cols), levels = names(site_cols)),
                    n = as.integer(table(factor(b$site, levels = names(site_cols)))))
leg <- ggplot(legend_df, aes(x = 1.6 + seq_along(site) * 1.5, y = 1)) +
  geom_point(aes(fill = site, shape = site), colour = "white", size = 3) +
  geom_text(aes(label = sprintf("%s (n = %d)", site, n)), hjust = 0, nudge_x = 0.12, size = 3, colour = ink) +
  scale_fill_manual(values = site_cols) + scale_shape_manual(values = site_shapes) +
  coord_cartesian(xlim = c(1.5, 8.9)) + theme_void() + theme(legend.position = "none")
fig <- wrap_elements(pA) + wrap_elements(pB) + leg + plot_layout(design = "AB\nCC", heights = c(12, 1))
ggsave(file.path(out_dir, "figureS3_cca_dim1_scores.png"), fig, width = 7.2, height = 3.9, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "figureS3_cca_dim1_scores.pdf"), fig, width = 7.2, height = 3.9, bg = "white")
message("Wrote canonical scores, diagnostics and Figure S3 to ", out_dir)
