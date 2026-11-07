#!/usr/bin/env Rscript

if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a  # compatibility with older R/ggplot2
suppressPackageStartupMessages({
  library(cowplot)
  library(dplyr)
  library(ggplot2)
  library(readxl)
  library(readr)
  library(scales)
  library(tidyr)
})

output_dir <- file.path("paper", "outputs", "section2")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

phaseA_path <- file.path("paper", "outputs", "shared", "paper_phaseA_participant_metrics.csv")
section1_path <- file.path("paper", "outputs", "section1", "section1_participant_metrics.csv")
clinical_path <- "CHANSS_clinical_demog.csv"
medication_path <- file.path("paper", "CHANSS_clinical_demog_filled_APLoad.xlsx")
corrections_path <- file.path("analysis_outputs", "ogt_patients_missing_clinical_data_corrected.csv")

motor_residualiser_terms <- c(
  "control_mean_speed_px_per_s"
)

control_fill <- "#7FA9C6"
patient_fill <- "#D67C59"
control_point <- "#5A7D9A"
patient_point <- "#C26D52"

to_num <- function(x) suppressWarnings(as.numeric(x))
sum_observed_items <- function(items) {
  items <- as.data.frame(items)
  out <- rowSums(items, na.rm = TRUE)
  out[rowSums(!is.na(items)) == 0] <- NA_real_
  out
}
format_p <- function(p) {
  if (!is.finite(p)) return("")
  if (p < 0.001) return("p < .001")
  sub("^0", "", sprintf("p = %.3f", p))
}
fmt <- function(x, digits = 2) formatC(x, digits = digits, format = "f", big.mark = ",")
finite_values <- function(df, field) {
  vals <- to_num(df[[field]])
  vals[is.finite(vals)]
}
glm_term_label <- function(term) {
  recode(
    term,
    "(Intercept)" = "Intercept",
    "control_mean_speed_px_per_s" = "Control motor speed",
    "sex_covariatemale" = "Sex: male",
    "age_covariate" = "Age",
    "group_covariatepatient" = "Group: patient",
    "countrySpain" = "Country: Spain",
    "countryChina" = "Country: China",
    "countryUnknown" = "Country: unknown",
    "AP_BNFmax" = "Antipsychotic burden",
    "AP_Load2026" = "Antipsychotic load 2026",
    .default = term
  )
}

load_medication_burden <- function(path) {
  read_excel(path, sheet = 1, .name_repair = "unique_quiet") %>%
    transmute(
      ID_subject = toupper(trimws(ID_subject)),
      AP_BNFmax = to_num(AP_BNFmax),
      AP_Load2026 = to_num(AP_Load2026)
    ) %>%
    distinct(ID_subject, .keep_all = TRUE)
}

load_clinical <- function(path) {
  clinical <- read_csv(path, show_col_types = FALSE, name_repair = "unique_quiet")
  tol_cols <- grep("^BACS_ToL_t", names(clinical), value = TRUE)
  if (!"BACS_ToL_t" %in% names(clinical) && length(tol_cols)) {
    clinical$BACS_ToL_t <- clinical[[tol_cols[[1]]]]
  }

  clinical %>%
    mutate(
      ID_subject = toupper(trimws(ID_subject)),
      MAP = sum_observed_items(across(c(BNSS_1, BNSS_2, BNSS_3, BNSS_5, BNSS_6, BNSS_7, BNSS_8), to_num)),
      EXP = sum_observed_items(across(c(BNSS_9, BNSS_10, BNSS_11, BNSS_12, BNSS_13), to_num)),
      CDS_total = sum_observed_items(across(paste0("CDS_", 1:9), to_num)),
      PANSS_marder_positive = sum_observed_items(across(c(PANSS_p1, PANSS_p3, PANSS_pg9, PANSS_pg10), to_num))
    ) %>%
    rowwise() %>%
    mutate(
      AMI_behavioural = mean(c_across(c(AMI_5, AMI_9, AMI_10, AMI_11, AMI_12, AMI_15)), na.rm = TRUE),
      AMI_social = mean(c_across(c(AMI_2, AMI_3, AMI_4, AMI_8, AMI_14, AMI_17)), na.rm = TRUE),
      AMI_emotional = mean(c_across(c(AMI_1, AMI_6, AMI_7, AMI_13, AMI_16, AMI_18)), na.rm = TRUE)
    ) %>%
    ungroup() %>%
    mutate(
      AMI_behavioural = if_else(is.nan(AMI_behavioural), NA_real_, as.numeric(AMI_behavioural)),
      AMI_social = if_else(is.nan(AMI_social), NA_real_, as.numeric(AMI_social)),
      AMI_emotional = if_else(is.nan(AMI_emotional), NA_real_, as.numeric(AMI_emotional))
    ) %>%
    distinct(ID_subject, .keep_all = TRUE)
}

patient_country <- function(platform, subject_id) {
  case_when(
    platform == "OGT_Mandarin" | grepl("^CH", subject_id) ~ "China",
    platform == "OGT_Spanish" | grepl("^C[378]", subject_id) ~ "Spain",  # Barcelona, Bilbao, Santander
    platform %in% c("OGT_English", "OCTAL") | grepl("^C", subject_id) ~ "UK",
    TRUE ~ "Unknown"
  )
}

clinical_specs <- tibble::tribble(
  ~clinical_metric, ~clinical_label,
  "EXP", "BNSS EXP",
  "MAP", "BNSS MAP",
  "AMI_behavioural", "AMI behavioural",
  "AMI_social", "AMI social",
  "AMI_emotional", "AMI emotional",
  "PSP_score", "PSP",
  "BACS_VM_t", "BACS verbal memory",
  "BACS_DS_t", "BACS digit sequencing",
  "BACS_TMT_t", "BACS token motor",
  "BACS_VF_t", "BACS verbal fluency",
  "BACS_SC_t", "BACS symbol coding",
  "BACS_ToL_t", "BACS Tower of London",
  "CDS_total", "CDS total",
  "PANSS_marder_positive", "PANSS positive"
)

match_patient_clinical <- function(rows) {
  clinical <- load_clinical(clinical_path)
  corrections <- if (file.exists(corrections_path)) {
    read_csv(corrections_path, show_col_types = FALSE) %>%
      transmute(participant_id = toupper(trimws(participant_id)), participant_id_corrected = toupper(trimws(participant_id_corrected)))
  } else {
    tibble(participant_id = character(), participant_id_corrected = character())
  }

  rows %>%
    filter(group == "patient") %>%
    mutate(
      participant_id_upper = toupper(trimws(participant_id)),
      clinical_lookup_id = coalesce(corrections$participant_id_corrected[match(participant_id_upper, corrections$participant_id)], participant_id_upper)
    ) %>%
    left_join(clinical, by = c("clinical_lookup_id" = "ID_subject")) %>%
    mutate(
      age = to_num(Age),
      sex = factor(if_else(as.character(Gender) == "1", "male", "female"), levels = c("female", "male")),
      country = factor(patient_country(source_platform, clinical_lookup_id), levels = c("UK", "Spain", "China", "Unknown")),
      illness_duration = age - to_num(Age_FEP),
      early_psychosis = if_else(is.finite(illness_duration) & illness_duration >= 0 & illness_duration <= 5, 1, 0, missing = NA_real_),
      trs = case_when(is.na(AP_1) | trimws(as.character(AP_1)) == "" ~ NA_real_, grepl("cloz", AP_1, ignore.case = TRUE) | grepl("cloz", AP_2, ignore.case = TRUE) ~ 1, TRUE ~ 0)
    )
}

read_section2_data <- function() {
  phaseA <- read_csv(phaseA_path, show_col_types = FALSE, guess_max = 10000) %>%
    mutate(group = as.character(group), participant_id = as.character(participant_id), source_id = as.character(source_id))

  section1 <- if (file.exists(section1_path)) {
    read_csv(section1_path, show_col_types = FALSE, guess_max = 10000) %>%
      mutate(group = as.character(group), participant_id = as.character(participant_id), source_id = as.character(source_id))
  } else {
    tibble()
  }

  missing_from_phaseA <- section1 %>%
    filter(group == "control") %>%
    anti_join(phaseA %>% filter(group == "control"), by = c("participant_id", "source_id")) %>%
    transmute(
      group,
      participant_id,
      source_id,
      source_platform,
      raw_source = NA_character_,
      control_path_count = practice_route_count,
      control_mean_speed_px_per_s = practice_mean_speed_px_per_s,
      control_peak_speed_px_per_s = NA_real_,
      control_mean_straightness = NA_real_,
      control_tremor_rms_px_per_s = NA_real_,
      control_jerk_rms = NA_real_,
      control_endpoint_scatter_px = NA_real_,
      control_path_duration_cv = NA_real_,
      test_uniqueness_mean = NA_real_,
      test_log_uniqueness_mean = NA_real_,
      test_mds_hull_area = NA_real_,
      test_uniqueness_route_count = NA_real_,
      test_position_uniqueness_mean = NA_real_,
      test_log_position_uniqueness_mean = NA_real_,
      test_position_uniqueness_route_count = NA_real_,
      test_uniqueness_residualised = NA_real_
    )

  rows <- bind_rows(phaseA, missing_from_phaseA)

  audit <- tibble(
    item = c(
      "task_sample_controls",
      "task_sample_patients",
      "controls_in_phaseA_uniqueness_file",
      "patients_in_phaseA_uniqueness_file",
      "controls_with_finite_uniqueness",
      "patients_with_finite_uniqueness",
      "controls_with_complete_motor_residualiser",
      "patients_with_complete_motor_residualiser",
      "controls_added_without_recomputed_uniqueness"
    ),
    n = c(
      if (nrow(section1)) sum(section1$group == "control") else NA_integer_,
      if (nrow(section1)) sum(section1$group == "patient") else NA_integer_,
      sum(phaseA$group == "control"),
      sum(phaseA$group == "patient"),
      sum(rows$group == "control" & is.finite(rows$test_uniqueness_mean)),
      sum(rows$group == "patient" & is.finite(rows$test_uniqueness_mean)),
      sum(rows$group == "control" & complete.cases(rows[, motor_residualiser_terms])),
      sum(rows$group == "patient" & complete.cases(rows[, motor_residualiser_terms])),
      nrow(missing_from_phaseA)
    )
  )
  write_csv(audit, file.path(output_dir, "section2_inclusion_audit.csv"))
  missing_report <- missing_from_phaseA %>%
    select(participant_id, source_id, control_path_count, control_mean_speed_px_per_s, test_uniqueness_mean)
  write_csv(missing_report, file.path(output_dir, "section2_controls_without_recomputed_uniqueness.csv"))
  rows
}

fit_motor_residualiser <- function(rows) {
  pooled_fit_df <- rows %>%
    filter(group %in% c("control", "patient"), is.finite(test_uniqueness_mean)) %>%
    filter(if_all(all_of(motor_residualiser_terms), is.finite))
  patient_fit_df <- rows %>%
    filter(group == "patient", is.finite(test_uniqueness_mean)) %>%
    filter(if_all(all_of(motor_residualiser_terms), is.finite))

  pooled_model <- lm(
    test_uniqueness_mean ~ control_mean_speed_px_per_s,
    data = pooled_fit_df
  )
  patient_model <- lm(
    test_uniqueness_mean ~ control_mean_speed_px_per_s,
    data = patient_fit_df
  )

  pred_df <- rows %>%
    mutate(.complete_motor = if_all(all_of(motor_residualiser_terms), is.finite) & is.finite(test_uniqueness_mean))
  pooled_pred <- rep(NA_real_, nrow(pred_df))
  pooled_pred[pred_df$.complete_motor] <- predict(pooled_model, newdata = pred_df[pred_df$.complete_motor, ])
  patient_pred <- rep(NA_real_, nrow(pred_df))
  patient_complete <- pred_df$.complete_motor & pred_df$group == "patient"
  patient_pred[patient_complete] <- predict(patient_model, newdata = pred_df[patient_complete, ])
  pred_df %>%
    mutate(
      test_uniqueness_motor_fitted = pooled_pred,
      test_uniqueness_motor_residualised = test_uniqueness_mean - test_uniqueness_motor_fitted,
      test_uniqueness_patient_motor_fitted = patient_pred,
      test_uniqueness_patient_motor_residualised = test_uniqueness_mean - test_uniqueness_patient_motor_fitted
    ) %>%
    select(-.complete_motor) -> out

  model_rows <- bind_rows(
    broom_like_lm(pooled_model) %>%
      mutate(model = "pooled residualiser: uniqueness ~ control_condition_motor_speed"),
    broom_like_lm(patient_model) %>%
      mutate(model = "patient residualiser: uniqueness ~ control_condition_motor_speed")
  )
  write_csv(model_rows, file.path(output_dir, "section2_motor_residualiser_model.csv"))
  out
}

broom_like_lm <- function(model) {
  coef_mat <- summary(model)$coefficients
  conf <- confint(model)
  tibble(
    term = rownames(coef_mat),
    estimate = coef_mat[, "Estimate"],
    std_error = coef_mat[, "Std. Error"],
    statistic = coef_mat[, "t value"],
    p_value = coef_mat[, "Pr(>|t|)"],
    conf_low = conf[, 1],
    conf_high = conf[, 2],
    n = length(model$residuals),
    r_squared = summary(model)$r.squared,
    adj_r_squared = summary(model)$adj.r.squared,
    residual_df = df.residual(model)
  )
}

compare_groups <- function(rows, metric, label) {
  controls <- finite_values(rows %>% filter(group == "control"), metric)
  patients <- finite_values(rows %>% filter(group == "patient"), metric)
  tt <- t.test(controls, patients, var.equal = FALSE)
  wt <- suppressWarnings(wilcox.test(controls, patients, exact = FALSE))
  tibble(
    metric = metric,
    label = label,
    n_control = length(controls),
    mean_control = mean(controls),
    sd_control = sd(controls),
    median_control = median(controls),
    n_patient = length(patients),
    mean_patient = mean(patients),
    sd_patient = sd(patients),
    median_patient = median(patients),
    welch_t = unname(tt$statistic),
    welch_df = unname(tt$parameter),
    welch_p = tt$p.value,
    wilcoxon_p = wt$p.value
  )
}

GeomFlatViolin <- ggproto(
  "GeomFlatViolin",
  Geom,
  setup_data = function(data, params) {
    data$width <- data$width %||% params$width %||% (resolution(data$x, FALSE) * 0.9)
    transform(data, ymin = y, ymax = y, xmin = x - violinwidth * width / 2, xmax = x)
  },
  draw_group = function(data, panel_scales, coord) {
    newdata <- rbind(arrange(transform(data, x = xmin), y), arrange(data, -y))
    newdata <- rbind(newdata, newdata[1, ])
    GeomPolygon$draw_panel(newdata, panel_scales, coord)
  },
  draw_key = draw_key_polygon,
  default_aes = aes(weight = 1, colour = "grey20", fill = "white", linewidth = 0.5, alpha = NA, linetype = "solid"),
  required_aes = c("x", "y")
)

geom_flat_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ..., trim = TRUE, scale = "area", show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomFlatViolin, position = position, show.legend = show.legend, inherit.aes = inherit.aes, params = list(trim = trim, scale = scale, ...))
}

panel_plot <- function(rows, metric, title, ylabel, ylim = NULL, zero_line = FALSE) {
  plot_df <- rows %>%
    transmute(group = factor(group, levels = c("control", "patient")), value = to_num(.data[[metric]])) %>%
    filter(is.finite(value)) %>%
    mutate(
      x_center = if_else(group == "control", 1, 2),
      x_violin = x_center - 0.06,
      x_box = x_center,
      x_point = x_center + 0.17
    )

  p <- ggplot() +
    geom_flat_violin(data = plot_df, aes(x = x_violin, y = value, fill = group, group = interaction(group, x_violin)), width = 0.85, alpha = 0.28, colour = NA, trim = FALSE) +
    geom_boxplot(data = plot_df, aes(x = x_box, y = value, fill = group, group = interaction(group, x_box)), width = 0.22, outlier.shape = NA, colour = "black", linewidth = 0.7, fatten = 1.1) +
    geom_point(data = plot_df, aes(x = x_point, y = value, colour = group), position = position_jitter(width = 0.055, height = 0, seed = 42), size = 1.8, alpha = 0.45, stroke = 0) +
    scale_x_continuous(breaks = c(1, 2), labels = c("Controls", "Patients"), limits = c(0.45, 2.55)) +
    scale_fill_manual(values = c(control = control_fill, patient = patient_fill), guide = "none") +
    scale_colour_manual(values = c(control = control_point, patient = patient_point), guide = "none") +
    labs(title = title, x = NULL, y = ylabel) +
    theme_classic(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11, hjust = 0.5), axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), axis.title.y = element_text(size = 10), plot.margin = margin(8, 10, 8, 10))

  if (grepl("speed", metric)) p <- p + scale_y_continuous(labels = label_number(big.mark = ","))
  if (zero_line) {
    p <- p + geom_hline(yintercept = 0, colour = "#9A9A9A", linewidth = 0.45, linetype = "dashed")
  }
  if (!is.null(ylim)) p <- p + coord_cartesian(ylim = ylim)
  p
}

bootstrap_spearman <- function(df, x_metric, y_metric, n_boot = 5000, seed = 1001) {
  x <- to_num(df[[x_metric]])
  y <- to_num(df[[y_metric]])
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]
  y <- y[keep]
  n <- length(x)
  if (n < 4 || stats::sd(x) == 0 || stats::sd(y) == 0) {
    return(tibble(n = n, spearman_rho = NA_real_, spearman_p = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_))
  }
  observed <- suppressWarnings(cor.test(x, y, method = "spearman", exact = FALSE))
  set.seed(seed)
  boot <- replicate(n_boot, {
    idx <- sample.int(n, n, replace = TRUE)
    if (stats::sd(x[idx]) == 0 || stats::sd(y[idx]) == 0) return(NA_real_)
    suppressWarnings(cor(x[idx], y[idx], method = "spearman"))
  })
  boot <- boot[is.finite(boot)]
  list(
    stats = tibble(
      n = n,
      spearman_rho = unname(observed$estimate),
      spearman_p = observed$p.value,
      ci_lower = unname(quantile(boot, 0.025, na.rm = TRUE)),
      ci_upper = unname(quantile(boot, 0.975, na.rm = TRUE))
    ),
    boot = boot
  )
}

make_bootstrap_correlations <- function(rows, n_boot = 5000) {
  patient_clinical <- match_patient_clinical(rows)
  boot_rows <- list()
  corr_rows <- bind_rows(lapply(seq_len(nrow(clinical_specs)), function(i) {
    clinical_metric <- clinical_specs$clinical_metric[[i]]
    motor_boot <- bootstrap_spearman(patient_clinical, "control_mean_speed_px_per_s", clinical_metric, n_boot = n_boot, seed = 2000 + i)
    uniqueness_boot <- bootstrap_spearman(patient_clinical, "test_uniqueness_patient_motor_residualised", clinical_metric, n_boot = n_boot, seed = 3000 + i)
    boot_rows[[length(boot_rows) + 1]] <<- tibble(
      metric = "control_mean_speed_px_per_s",
      clinical_metric = clinical_metric,
      bootstrap_rho = motor_boot$boot
    )
    boot_rows[[length(boot_rows) + 1]] <<- tibble(
      metric = "test_uniqueness_patient_motor_residualised",
      clinical_metric = clinical_metric,
      bootstrap_rho = uniqueness_boot$boot
    )
    motor <- motor_boot$stats %>%
      mutate(metric = "control_mean_speed_px_per_s", metric_label = "Control motor speed", clinical_metric = clinical_metric)
    uniqueness <- uniqueness_boot$stats %>%
      mutate(metric = "test_uniqueness_patient_motor_residualised", metric_label = "Patient motor-residualised uniqueness", clinical_metric = clinical_metric)
    bind_rows(motor, uniqueness)
  })) %>%
    left_join(clinical_specs, by = "clinical_metric") %>%
    mutate(clinical_label = factor(clinical_label, levels = rev(clinical_specs$clinical_label)))

  write_csv(
    corr_rows %>% mutate(clinical_label = as.character(clinical_label)),
    file.path(output_dir, "section2_bootstrap_clinical_correlations.csv")
  )
  list(stats = corr_rows, boot = bind_rows(boot_rows) %>% left_join(clinical_specs, by = "clinical_metric"))
}

bootstrap_density_ribbons <- function(corr_rows, boot_rows, metric_name) {
  plot_stats <- corr_rows %>%
    filter(metric == metric_name) %>%
    mutate(y = as.numeric(clinical_label))

  bind_rows(lapply(seq_len(nrow(plot_stats)), function(i) {
    row <- plot_stats[i, ]
    boot <- boot_rows %>%
      filter(metric == metric_name, clinical_metric == row$clinical_metric) %>%
      pull(bootstrap_rho)
    boot <- boot[is.finite(boot)]
    if (length(boot) < 10) {
      return(tibble())
    }
    boot_low <- max(-0.65, min(boot, na.rm = TRUE))
    boot_high <- min(0.65, max(boot, na.rm = TRUE))
    dens <- density(boot, from = boot_low, to = boot_high, n = 220, adjust = 1.0)
    dens_df <- tibble(x = dens$x, density = dens$y) %>%
      filter(x >= boot_low, x <= boot_high)
    if (nrow(dens_df) < 2) {
      return(tibble())
    }
    dens_df %>%
      mutate(
        xmin = x,
        xmax = lead(x),
        alpha = density / max(density, na.rm = TRUE),
        y = row$y,
        ymin = y - 0.35,
        ymax = y + 0.35
      ) %>%
      filter(is.finite(xmax))
  }))
}

clinical_forest_plot <- function(corr_rows, boot_rows, metric_name, title) {
  plot_df <- corr_rows %>%
    filter(metric == metric_name) %>%
    mutate(
      y = as.numeric(clinical_label),
      ci_width = ci_upper - ci_lower
    )
  ribbon_df <- bootstrap_density_ribbons(corr_rows, boot_rows, metric_name)

  ggplot(plot_df, aes(y = clinical_label)) +
    geom_rect(
      data = ribbon_df,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = alpha),
      inherit.aes = FALSE
    ) +
    scale_fill_gradientn(
      colours = c("#FFFFFF", "#F9EAE4", "#F0C4B5", "#DD8E72"),
      values = c(0, 0.25, 0.65, 1),
      limits = c(0, 1),
      guide = "none"
    ) +
    geom_vline(xintercept = 0, colour = "#B8B8B8", linewidth = 0.45) +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0, colour = "black", linewidth = 0.55) +
    geom_point(aes(x = spearman_rho), size = 1.9, colour = "black") +
    scale_x_continuous(
      limits = c(-0.65, 0.65),
      breaks = seq(-0.6, 0.6, 0.3),
      labels = label_number(accuracy = 0.1)
    ) +
    labs(title = title, x = "Spearman rho", y = NULL) +
    theme_classic(base_size = 9.5) +
    theme(
      plot.title = element_text(face = "bold", size = 10, hjust = 0.5),
      axis.text.y = element_text(size = 8.5),
      axis.text.x = element_text(size = 8.5),
      axis.title.x = element_text(size = 9),
      plot.margin = margin(6, 8, 6, 8)
    )
}

make_group_figure <- function(rows) {
  stats <- bind_rows(
    compare_groups(rows, "control_mean_speed_px_per_s", "Control motor speed"),
    compare_groups(rows, "test_uniqueness_mean", "Raw uniqueness"),
    compare_groups(rows, "test_uniqueness_motor_residualised", "Uniqueness residualised for control-condition motor speed (pooled)")
  )
  write_csv(stats, file.path(output_dir, "section2_group_stats.csv"))

  correlations <- make_bootstrap_correlations(rows)
  corr_rows <- correlations$stats
  boot_rows <- correlations$boot

  tryCatch({
  p1 <- panel_plot(
    rows,
    "test_uniqueness_mean",
    "Mean uniqueness",
    "Mean uniqueness",
    ylim = c(0, 2300)
  )
  p2 <- panel_plot(
    rows,
    "test_uniqueness_motor_residualised",
    "Residualised uniqueness",
    "Motor-residualised uniqueness",
    ylim = c(-1000, 2300),
    zero_line = TRUE
  )
  p3 <- clinical_forest_plot(
    corr_rows,
    boot_rows,
    "control_mean_speed_px_per_s",
    "Clinical correlates of motor speed"
  )
  p4 <- clinical_forest_plot(
    corr_rows,
    boot_rows,
    "test_uniqueness_patient_motor_residualised",
    "Clinical correlates of patient-residualised uniqueness"
  )

  combined <- cowplot::plot_grid(
    p1, p2, p3, p4,
    labels = c("A", "B", "C", "D"),
    ncol = 2,
    rel_heights = c(0.8, 1.35),
    label_size = 13,
    label_fontface = "bold",
    hjust = -0.35,
    vjust = 1.15
  )
  ggsave(file.path(output_dir, "figure3_motor_adjusted_uniqueness.png"), combined, width = 9.6, height = 7.0, dpi = 300, bg = "white")
  ggsave(file.path(output_dir, "figure3_motor_adjusted_uniqueness.pdf"), combined, width = 9.6, height = 7.0, device = cairo_pdf, bg = "white")
  }, error = function(e) message("Figure 3 not rendered (plotting backend): ", conditionMessage(e)))
  stats
}

prepare_covariate_glm_data <- function(rows) {
  clinical <- load_clinical(clinical_path) %>%
    select(ID_subject, Age, Gender, pEducation)
  corrections <- if (file.exists(corrections_path)) {
    read_csv(corrections_path, show_col_types = FALSE) %>%
      transmute(participant_id = toupper(trimws(participant_id)), participant_id_corrected = toupper(trimws(participant_id_corrected)))
  } else {
    tibble(participant_id = character(), participant_id_corrected = character())
  }

  rows %>%
    mutate(
      participant_id_upper = toupper(trimws(participant_id)),
      clinical_lookup_id = if_else(
        group == "patient",
        coalesce(corrections$participant_id_corrected[match(participant_id_upper, corrections$participant_id)], participant_id_upper),
        NA_character_
      )
    ) %>%
    left_join(clinical, by = c("clinical_lookup_id" = "ID_subject")) %>%
    mutate(
      age_covariate = if_else(group == "patient" & is.finite(to_num(Age)), to_num(Age), to_num(age)),
      sex_covariate = case_when(
        group == "patient" & as.character(Gender) == "1" ~ "male",
        group == "patient" & as.character(Gender) == "0" ~ "female",
        group == "control" ~ tolower(trimws(as.character(sex))),
        TRUE ~ NA_character_
      ),
      sex_covariate = factor(sex_covariate, levels = c("female", "male")),
      group_covariate = factor(group, levels = c("control", "patient")),
      education_years = if_else(group == "patient", to_num(pEducation), NA_real_)
    )
}

make_covariate_glm <- function(rows) {
  covar_rows <- prepare_covariate_glm_data(rows)
  glm_df <- covar_rows %>%
    filter(
      is.finite(test_uniqueness_mean),
      is.finite(control_mean_speed_px_per_s),
      is.finite(age_covariate),
      !is.na(sex_covariate),
      !is.na(group_covariate)
    )

  fit <- lm(
    test_uniqueness_mean ~ control_mean_speed_px_per_s + sex_covariate + age_covariate + group_covariate,
    data = glm_df
  )

  model_label <- "full-sample GLM: uniqueness ~ motor_speed + sex + age + group"
  coef_rows <- broom_like_lm(fit) %>%
    mutate(model = model_label, .before = 1)
  write_csv(coef_rows, file.path(output_dir, "section2_covariate_glm_coefficients.csv"))

  model_summary <- summary(fit)
  fstat <- model_summary$fstatistic
  glance <- tibble(
    model = model_label,
    n = length(fit$residuals),
    n_controls = sum(glm_df$group_covariate == "control"),
    n_patients = sum(glm_df$group_covariate == "patient"),
    r_squared = model_summary$r.squared,
    adj_r_squared = model_summary$adj.r.squared,
    residual_df = df.residual(fit),
    f_statistic = unname(fstat[["value"]]),
    f_num_df = unname(fstat[["numdf"]]),
    f_den_df = unname(fstat[["dendf"]]),
    f_p_value = pf(fstat[["value"]], fstat[["numdf"]], fstat[["dendf"]], lower.tail = FALSE)
  )
  write_csv(glance, file.path(output_dir, "section2_covariate_glm_summary.csv"))
  write_csv(
    glm_df %>%
      select(participant_id, clinical_lookup_id, group_covariate, test_uniqueness_mean, control_mean_speed_px_per_s, age_covariate, sex_covariate, education_years),
    file.path(output_dir, "section2_covariate_glm_analysis_sample.csv")
  )

  education_audit <- covar_rows %>%
    filter(is.finite(test_uniqueness_mean)) %>%
    group_by(group) %>%
    summarise(
      n_with_uniqueness = n(),
      n_with_education = sum(is.finite(education_years)),
      .groups = "drop"
    ) %>%
    mutate(
      requested_model = "uniqueness ~ motor_speed + sex + age + group + education",
      estimable = all(n_with_education > 0),
      note = if_else(
        n_with_education == 0,
        "Education is unavailable for this group in the current metadata.",
        "Education is available for this group."
      )
    )
  write_csv(education_audit, file.path(output_dir, "section2_education_covariate_audit.csv"))

  list(coefs = coef_rows, glance = glance, education_audit = education_audit)
}

make_medication_sensitivity_glm <- function(rows, medication_metric = "AP_BNFmax", medication_label = "AP_BNFmax", output_suffix = "ap_bnfmax") {
  covar_rows <- prepare_covariate_glm_data(rows)
  medication <- load_medication_burden(medication_path)
  medication_rows <- covar_rows %>%
    filter(group == "patient") %>%
    left_join(medication, by = c("clinical_lookup_id" = "ID_subject")) %>%
    mutate(
      country = factor(patient_country(source_platform, clinical_lookup_id), levels = c("UK", "Spain", "China", "Unknown"))
    )

  glm_df <- medication_rows %>%
    filter(
      is.finite(test_uniqueness_motor_residualised),
      is.finite(age_covariate),
      !is.na(sex_covariate),
      !is.na(country),
      is.finite(.data[[medication_metric]])
    )

  formula <- as.formula(paste0("test_uniqueness_motor_residualised ~ age_covariate + sex_covariate + country + ", medication_metric))
  fit <- lm(formula, data = glm_df)

  model_label <- paste0("patient-only GLM: motor-residualised uniqueness ~ age + sex + country + ", medication_label)
  coef_rows <- broom_like_lm(fit) %>%
    mutate(model = model_label, .before = 1)
  write_csv(coef_rows, file.path(output_dir, paste0("section2_medication_sensitivity_glm_", output_suffix, "_coefficients.csv")))

  model_summary <- summary(fit)
  fstat <- model_summary$fstatistic
  glance <- tibble(
    model = model_label,
    n = length(fit$residuals),
    n_patients = nrow(glm_df),
    n_patients_with_residualised_uniqueness = sum(medication_rows$group == "patient" & is.finite(medication_rows$test_uniqueness_motor_residualised)),
    medication_metric = medication_metric,
    n_patients_with_medication_metric = sum(is.finite(medication_rows[[medication_metric]])),
    r_squared = model_summary$r.squared,
    adj_r_squared = model_summary$adj.r.squared,
    residual_df = df.residual(fit),
    f_statistic = unname(fstat[["value"]]),
    f_num_df = unname(fstat[["numdf"]]),
    f_den_df = unname(fstat[["dendf"]]),
    f_p_value = pf(fstat[["value"]], fstat[["numdf"]], fstat[["dendf"]], lower.tail = FALSE)
  )
  write_csv(glance, file.path(output_dir, paste0("section2_medication_sensitivity_glm_", output_suffix, "_summary.csv")))
  write_csv(
    glm_df %>%
      select(participant_id, clinical_lookup_id, test_uniqueness_motor_residualised, age_covariate, sex_covariate, country, all_of(medication_metric)),
    file.path(output_dir, paste0("section2_medication_sensitivity_glm_", output_suffix, "_analysis_sample.csv"))
  )

  list(coefs = coef_rows, glance = glance)
}

make_table_s1_medication_glm <- function(rows) {
  covar_rows <- prepare_covariate_glm_data(rows)
  medication <- load_medication_burden(medication_path)
  medication_rows <- covar_rows %>%
    filter(group == "patient") %>%
    left_join(medication, by = c("clinical_lookup_id" = "ID_subject"))

  glm_df <- medication_rows %>%
    filter(
      is.finite(test_uniqueness_mean),
      is.finite(control_mean_speed_px_per_s),
      is.finite(age_covariate),
      !is.na(sex_covariate),
      is.finite(AP_BNFmax)
    )

  fit <- lm(
    test_uniqueness_mean ~ control_mean_speed_px_per_s + sex_covariate + age_covariate + AP_BNFmax,
    data = glm_df
  )

  model_label <- "patient-only GLM: uniqueness ~ motor_speed + sex + age + AP_BNFmax"
  coef_rows <- broom_like_lm(fit) %>%
    mutate(model = model_label, .before = 1)
  write_csv(coef_rows, file.path(output_dir, "table_s1_medication_effect_on_uniqueness_coefficients.csv"))

  model_summary <- summary(fit)
  fstat <- model_summary$fstatistic
  glance <- tibble(
    model = model_label,
    n = length(fit$residuals),
    n_patients_with_uniqueness = sum(medication_rows$group == "patient" & is.finite(medication_rows$test_uniqueness_mean)),
    n_patients_with_AP_BNFmax = sum(is.finite(medication_rows$AP_BNFmax)),
    r_squared = model_summary$r.squared,
    adj_r_squared = model_summary$adj.r.squared,
    residual_df = df.residual(fit),
    f_statistic = unname(fstat[["value"]]),
    f_num_df = unname(fstat[["numdf"]]),
    f_den_df = unname(fstat[["dendf"]]),
    f_p_value = pf(fstat[["value"]], fstat[["numdf"]], fstat[["dendf"]], lower.tail = FALSE)
  )
  write_csv(glance, file.path(output_dir, "table_s1_medication_effect_on_uniqueness_summary.csv"))
  write_csv(
    glm_df %>%
      select(participant_id, clinical_lookup_id, test_uniqueness_mean, control_mean_speed_px_per_s, sex_covariate, age_covariate, AP_BNFmax),
    file.path(output_dir, "table_s1_medication_effect_on_uniqueness_analysis_sample.csv")
  )

  lines <- c(
    "# Table S1. Medication effect on uniqueness",
    "",
    paste0(
      "Patient-only general linear model predicting uniqueness from control-condition motor speed, sex, age, ",
      "and antipsychotic burden expressed as percentage of BNF maximum dose. n = ", glance$n,
      "; adjusted R2 = ", fmt(glance$adj_r_squared, 3), "."
    ),
    "",
    "| Predictor | Estimate | SE | t | 95% CI | p |",
    "|---|---:|---:|---:|---:|---:|"
  )
  for (row in split(coef_rows, seq_len(nrow(coef_rows)))) {
    lines <- c(lines, paste0(
      "| ", glm_term_label(row$term), " | ",
      fmt(row$estimate), " | ",
      fmt(row$std_error), " | ",
      fmt(row$statistic), " | ",
      "[", fmt(row$conf_low), ", ", fmt(row$conf_high), "] | ",
      format_p(row$p_value), " |"
    ))
  }
  lines <- c(
    lines,
    "",
    "AP_BNFmax is the summed antipsychotic dose burden, expressed as percentage of BNF maximum dose."
  )
  writeLines(lines, file.path(output_dir, "table_s1_medication_effect_on_uniqueness.md"))

  list(coefs = coef_rows, glance = glance)
}

write_summary <- function(group_stats, glm, medication_glms) {
  raw <- group_stats %>% filter(metric == "test_uniqueness_mean") %>% slice(1)
  resid <- group_stats %>% filter(metric == "test_uniqueness_motor_residualised") %>% slice(1)

  lines <- c(
    "# Section 2. Motor-adjusted uniqueness",
    "",
    "## Outputs",
    "",
    "- Figure PNG: `paper/outputs/section2/figure3_motor_adjusted_uniqueness.png`",
    "- Figure PDF: `paper/outputs/section2/figure3_motor_adjusted_uniqueness.pdf`",
    "- Group statistics: `paper/outputs/section2/section2_group_stats.csv`",
    "- Bootstrapped clinical correlations: `paper/outputs/section2/section2_bootstrap_clinical_correlations.csv`",
    "- Covariate GLM coefficients: `paper/outputs/section2/section2_covariate_glm_coefficients.csv`",
    "- Covariate GLM summary: `paper/outputs/section2/section2_covariate_glm_summary.csv`",
    "- Medication sensitivity GLM coefficients: `paper/outputs/section2/section2_medication_sensitivity_glm_ap_bnfmax_coefficients.csv`",
    "- Medication sensitivity GLM summary: `paper/outputs/section2/section2_medication_sensitivity_glm_ap_bnfmax_summary.csv`",
    "- Table S1 medication GLM: `paper/outputs/section2/table_s1_medication_effect_on_uniqueness.md`",
    "- Education covariate audit: `paper/outputs/section2/section2_education_covariate_audit.csv`",
    "",
    "## Group Statistics",
    "",
    "| Measure | Controls mean (SD) | Patients mean (SD) | Inferential statistic |",
    "|---|---:|---:|---|"
  )

  for (row in split(group_stats, seq_len(nrow(group_stats)))) {
    lines <- c(lines, paste0(
      "| ", row$label, " | ",
      fmt(row$mean_control), " (", fmt(row$sd_control), "), n = ", row$n_control, " | ",
      fmt(row$mean_patient), " (", fmt(row$sd_patient), "), n = ", row$n_patient, " | ",
      "t = ", fmt(row$welch_t), ", df = ", fmt(row$welch_df, 1), ", ", format_p(row$welch_p),
      "; Wilcoxon ", format_p(row$wilcoxon_p), " |"
    ))
  }

  lines <- c(
    lines,
    "",
    "## Covariate GLM",
    "",
    paste0(
      "Full-sample model: uniqueness ~ control motor speed + sex + age + group. ",
      "n = ", glm$glance$n, " (controls = ", glm$glance$n_controls, ", patients = ", glm$glance$n_patients,
      "), adjusted R2 = ", fmt(glm$glance$adj_r_squared, 3), "."
    ),
    "",
    "| Predictor | Estimate | SE | t | 95% CI | p |",
    "|---|---:|---:|---:|---:|---:|"
  )

  for (row in split(glm$coefs, seq_len(nrow(glm$coefs)))) {
    lines <- c(lines, paste0(
      "| ", glm_term_label(row$term), " | ",
      fmt(row$estimate), " | ",
      fmt(row$std_error), " | ",
      fmt(row$statistic), " | ",
      "[", fmt(row$conf_low), ", ", fmt(row$conf_high), "] | ",
      format_p(row$p_value), " |"
    ))
  }

  for (medication_glm in medication_glms) {
    lines <- c(
      lines,
      "",
      "## Medication Sensitivity GLM",
      "",
      paste0(
        medication_glm$glance$model, ". ",
        "n = ", medication_glm$glance$n,
        ", adjusted R2 = ", fmt(medication_glm$glance$adj_r_squared, 3), "."
      ),
      "",
      "| Predictor | Estimate | SE | t | 95% CI | p |",
      "|---|---:|---:|---:|---:|---:|"
    )

    for (row in split(medication_glm$coefs, seq_len(nrow(medication_glm$coefs)))) {
      lines <- c(lines, paste0(
        "| ", glm_term_label(row$term), " | ",
        fmt(row$estimate), " | ",
        fmt(row$std_error), " | ",
        fmt(row$statistic), " | ",
        "[", fmt(row$conf_low), ", ", fmt(row$conf_high), "] | ",
        format_p(row$p_value), " |"
      ))
    }
  }

  lines <- c(
    lines,
    "",
    "## Paragraph Draft",
    "",
    paste0(
      "Controls showed higher raw uniqueness than patients (controls: M = ", fmt(raw$mean_control), ", SD = ", fmt(raw$sd_control), ", n = ", raw$n_control,
      "; patients: M = ", fmt(raw$mean_patient), ", SD = ", fmt(raw$sd_patient), ", n = ", raw$n_patient,
      "; Welch t = ", fmt(raw$welch_t), ", ", format_p(raw$welch_p), "). ",
      "The group difference remained after residualising uniqueness for control-condition motor speed using a pooled control-patient model (controls: M = ", fmt(resid$mean_control), ", SD = ", fmt(resid$sd_control), ", n = ", resid$n_control,
      "; patients: M = ", fmt(resid$mean_patient), ", SD = ", fmt(resid$sd_patient), ", n = ", resid$n_patient,
      "; Welch t = ", fmt(resid$welch_t), ", ", format_p(resid$welch_p), "). ",
      "Because patient and control groups differed in age and sex, we further tested the group effect in a full-sample GLM adjusting for control-block motor speed, age, and sex. ",
      "The group effect remained in this model (n = ", glm$glance$n, "; adjusted R2 = ", fmt(glm$glance$adj_r_squared, 3), "). ",
      "Education was not included in the full-sample GLM because education data are unavailable for controls in the current metadata; see the education covariate audit."
    )
  )
  writeLines(lines, file.path(output_dir, "section2_motor_adjusted_uniqueness_summary.md"))
}

rows <- read_section2_data()
rows <- fit_motor_residualiser(rows)
write_csv(rows, file.path(output_dir, "section2_participant_metrics.csv"))
group_stats <- make_group_figure(rows)
glm <- make_covariate_glm(rows)
medication_glms <- list(
  make_medication_sensitivity_glm(rows, "AP_BNFmax", "AP_BNFmax", "ap_bnfmax")
)
table_s1 <- make_table_s1_medication_glm(rows)
write_summary(group_stats, glm, medication_glms)

message("Wrote section 2 outputs to ", output_dir)
