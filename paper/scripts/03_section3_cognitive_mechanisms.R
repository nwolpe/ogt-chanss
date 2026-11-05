#!/usr/bin/env Rscript

if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a  # compatibility with older R/ggplot2
suppressPackageStartupMessages({
  library(cowplot)
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
  library(tidyr)
})

output_dir <- file.path("paper", "outputs", "section3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

section2_path <- file.path("paper", "outputs", "section2", "section2_participant_metrics.csv")
legacy_mechanism_path <- file.path("analysis_outputs", "phaseB_exploration_dynamics", "phaseB_exploration_dynamics_participant_metrics.csv")

mechanisms <- tibble::tribble(
  ~metric, ~label,
  "roaming_entropy_normalised", "Roaming entropy",
  "submovements_mean_per_path", "Option elaboration",
  "perseveration_ratio", "Perseveration",
  "temporal_prefix_uniqueness_spearman_rho", "Novelty maintenance"
)

control_fill <- "#7FA9C6"
patient_fill <- "#D67C59"
control_point <- "#5A7D9A"
patient_point <- "#C26D52"

to_num <- function(x) suppressWarnings(as.numeric(x))
fmt <- function(x, digits = 2) formatC(x, digits = digits, format = "f", big.mark = ",")
format_p <- function(p) {
  if (!is.finite(p)) return("")
  if (p < 0.001) return("p < .001")
  sub("^0", "", sprintf("p = %.3f", p))
}

read_section3_data <- function() {
  paper_rows <- read_csv(section2_path, show_col_types = FALSE, guess_max = 10000) %>%
    mutate(
      group = as.character(group),
      participant_id = as.character(participant_id),
      source_id = as.character(source_id)
    )

  mechanism_rows <- read_csv(legacy_mechanism_path, show_col_types = FALSE, guess_max = 10000) %>%
    mutate(
      group = as.character(group),
      participant_id = as.character(participant_id),
      source_id = as.character(source_id)
    ) %>%
    select(
      group,
      participant_id,
      source_id,
      mechanism_residualised_uniqueness = test_uniqueness_residualised,
      all_of(mechanisms$metric)
    )

  rows <- paper_rows %>%
    left_join(mechanism_rows, by = c("group", "participant_id", "source_id"))

  audit <- tibble(
    item = c(
      "paper_controls",
      "paper_patients",
      "mechanism_complete_controls",
      "mechanism_complete_patients",
      "mechanism_missing_controls",
      "mechanism_missing_patients"
    ),
    n = c(
      sum(rows$group == "control"),
      sum(rows$group == "patient"),
      sum(rows$group == "control" & complete.cases(rows[, mechanisms$metric])),
      sum(rows$group == "patient" & complete.cases(rows[, mechanisms$metric])),
      sum(rows$group == "control" & !complete.cases(rows[, mechanisms$metric])),
      sum(rows$group == "patient" & !complete.cases(rows[, mechanisms$metric]))
    )
  )
  write_csv(audit, file.path(output_dir, "section3_mechanism_sample_audit.csv"))
  rows
}

finite_values <- function(df, field) {
  vals <- to_num(df[[field]])
  vals[is.finite(vals)]
}

compare_groups <- function(rows) {
  stats <- bind_rows(lapply(seq_len(nrow(mechanisms)), function(i) {
    metric <- mechanisms$metric[[i]]
    controls <- finite_values(rows %>% filter(group == "control"), metric)
    patients <- finite_values(rows %>% filter(group == "patient"), metric)
    tt <- t.test(controls, patients, var.equal = FALSE)
    wt <- suppressWarnings(wilcox.test(controls, patients, exact = FALSE))
    tibble(
      metric = metric,
      label = mechanisms$label[[i]],
      n_control = length(controls),
      n_patient = length(patients),
      mean_control = mean(controls),
      sd_control = sd(controls),
      median_control = median(controls),
      mean_patient = mean(patients),
      sd_patient = sd(patients),
      median_patient = median(patients),
      welch_t = unname(tt$statistic),
      welch_df = unname(tt$parameter),
      welch_p = tt$p.value,
      wilcoxon_p = wt$p.value
    )
  }))
  write_csv(stats, file.path(output_dir, "section3_mechanism_group_stats.csv"))
  stats
}

make_correlation_matrix <- function(rows) {
  patient_rows <- rows %>%
    filter(group == "patient") %>%
    select(all_of(mechanisms$metric))

  mat <- cor(patient_rows, use = "pairwise.complete.obs", method = "spearman")
  out <- as.data.frame(as.table(mat)) %>%
    transmute(
      mechanism_1 = mechanisms$label[match(Var1, mechanisms$metric)],
      mechanism_2 = mechanisms$label[match(Var2, mechanisms$metric)],
      spearman_rho = Freq
    )
  write_csv(out, file.path(output_dir, "section3_mechanism_correlation_matrix.csv"))

  out %>%
    mutate(
      mechanism_1 = factor(mechanism_1, levels = mechanisms$label),
      mechanism_2 = factor(mechanism_2, levels = rev(mechanisms$label))
    )
}

ols_fit <- function(df, target, predictors, standardise = FALSE) {
  model_df <- df %>%
    select(all_of(c(target, predictors))) %>%
    mutate(across(everything(), to_num)) %>%
    filter(if_all(everything(), is.finite))
  if (standardise) {
    model_df <- model_df %>% mutate(across(everything(), ~ as.numeric(scale(.x))))
  }
  if (nrow(model_df) <= length(predictors) + 2) return(list(r2 = NA_real_, n = nrow(model_df), model = NULL, df = model_df))
  model <- lm(reformulate(predictors, response = target), data = model_df)
  list(r2 = summary(model)$r.squared, n = nrow(model_df), model = model, df = model_df)
}

shapley_r2 <- function(rows, target, predictors) {
  model_rows <- rows
  predictors <- as.character(predictors)
  cache <- new.env(parent = emptyenv())
  key <- function(items) {
    if (length(items) == 0) return("__empty__")
    paste(sort(items), collapse = "|")
  }
  assign("__empty__", 0, envir = cache)

  for (size in seq_along(predictors)) {
    combs <- combn(predictors, size, simplify = FALSE)
    for (subset in combs) {
      assign(key(subset), ols_fit(model_rows, target, subset, standardise = TRUE)$r2, envir = cache)
    }
  }

  full_key <- key(predictors)
  total_r2 <- get(full_key, envir = cache)
  complete_n <- ols_fit(model_rows, target, predictors, standardise = TRUE)$n
  p <- length(predictors)

  bind_rows(lapply(predictors, function(pred) {
    others <- setdiff(predictors, pred)
    value <- 0
    for (size in 0:length(others)) {
      subsets <- if (size == 0) list(character()) else combn(others, size, simplify = FALSE)
      for (subset in subsets) {
        subset_key <- key(subset)
        with_key <- key(c(subset, pred))
        weight <- factorial(size) * factorial(p - size - 1) / factorial(p)
        value <- value + weight * (get(with_key, envir = cache) - get(subset_key, envir = cache))
      }
    }
    tibble(
      predictor = pred,
      label = mechanisms$label[match(pred, mechanisms$metric)],
      shapley_r_squared = value,
      total_r_squared = total_r2,
      n = complete_n
    )
  })) %>%
    arrange(desc(shapley_r_squared))
}

make_attenuation <- function(rows) {
  model_rows <- rows %>%
    mutate(group_patient = if_else(group == "patient", 1, 0))
  specs <- list(
    list(model = "Group only", predictors = c("group_patient")),
    list(model = "Group + motor", predictors = c("group_patient", "control_mean_speed_px_per_s")),
    list(model = "Group + motor + mechanisms", predictors = c("group_patient", "control_mean_speed_px_per_s", mechanisms$metric))
  )

  base_beta <- NA_real_
  out <- bind_rows(lapply(specs, function(spec) {
    fit <- ols_fit(model_rows, "test_uniqueness_mean", spec$predictors, standardise = FALSE)
    if (is.null(fit$model)) {
      return(tibble(model = spec$model, n = fit$n))
    }
    coef_summary <- summary(fit$model)$coefficients
    conf <- confint(fit$model)
    beta <- coef_summary["group_patient", "Estimate"]
    if (spec$model == "Group only") base_beta <<- beta
    tibble(
      model = spec$model,
      n = fit$n,
      r_squared = summary(fit$model)$r.squared,
      adj_r_squared = summary(fit$model)$adj.r.squared,
      group_beta = beta,
      group_se = coef_summary["group_patient", "Std. Error"],
      group_t = coef_summary["group_patient", "t value"],
      group_p = coef_summary["group_patient", "Pr(>|t|)"],
      group_ci_lower = conf["group_patient", 1],
      group_ci_upper = conf["group_patient", 2],
      attenuation_pct = if_else(is.finite(base_beta), 100 * (1 - beta / base_beta), NA_real_)
    )
  }))
  out$model <- factor(out$model, levels = vapply(specs, `[[`, character(1), "model"))
  write_csv(out %>% mutate(model = as.character(model)), file.path(output_dir, "section3_group_effect_attenuation.csv"))
  out
}

make_raincloud_data <- function(rows) {
  rows %>%
    select(group, participant_id, source_id, all_of(mechanisms$metric)) %>%
    pivot_longer(all_of(mechanisms$metric), names_to = "metric", values_to = "value") %>%
    left_join(mechanisms, by = "metric") %>%
    group_by(metric) %>%
    mutate(value_z = as.numeric(scale(value))) %>%
    ungroup() %>%
    mutate(
      label = factor(label, levels = mechanisms$label),
      group = factor(group, levels = c("control", "patient"), labels = c("Controls", "Patients"))
    )
}

panel_a_corr <- function(corr_df) {
  ggplot(corr_df, aes(x = mechanism_1, y = mechanism_2, fill = spearman_rho)) +
    geom_tile(colour = "white", linewidth = 0.8) +
    geom_text(aes(label = sprintf("%.2f", spearman_rho)), size = 3.0) +
    scale_fill_gradient2(low = "#5A7D9A", mid = "white", high = "#D67C59", midpoint = 0, limits = c(-1, 1), name = "rho") +
    coord_equal() +
    labs(title = "Mechanism correlations", x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
      axis.text.x = element_text(angle = 35, hjust = 1, size = 8.5),
      axis.text.y = element_text(size = 8.5),
      panel.grid = element_blank(),
      legend.position = "right"
    )
}

panel_b_groups <- function(plot_df) {
  ggplot(plot_df, aes(x = label, y = value_z, fill = group, colour = group)) +
    geom_violin(
      position = position_dodge(width = 0.72),
      width = 0.68,
      alpha = 0.24,
      colour = NA,
      trim = FALSE,
      scale = "width"
    ) +
    geom_boxplot(
      position = position_dodge(width = 0.72),
      width = 0.18,
      outlier.shape = NA,
      colour = "black",
      linewidth = 0.35,
      alpha = 0.78
    ) +
    geom_point(
      position = position_jitterdodge(jitter.width = 0.09, dodge.width = 0.72, seed = 7),
      size = 0.65,
      alpha = 0.28,
      stroke = 0
    ) +
    scale_fill_manual(values = c(Controls = control_fill, Patients = patient_fill), guide = "none") +
    scale_colour_manual(values = c(Controls = control_point, Patients = patient_point), guide = "none") +
    labs(title = "Mechanism group differences", x = NULL, y = "Mechanism score (z)") +
    theme_classic(base_size = 9.5) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
      axis.text.x = element_text(angle = 25, hjust = 1, size = 8),
      axis.text.y = element_text(size = 8),
      plot.margin = margin(6, 6, 6, 6)
    )
}

panel_c_shapley <- function(shapley_df) {
  plot_df <- shapley_df %>%
    mutate(label = factor(label, levels = rev(label)))
  ggplot(plot_df, aes(x = shapley_r_squared, y = label)) +
    geom_col(fill = "#7FA36B", width = 0.62) +
    geom_text(aes(label = sprintf("%.3f", shapley_r_squared)), hjust = -0.08, size = 2.8) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(title = paste0("Variance partitioning across the full sample"), x = "Shapley R2", y = NULL) +
    theme_classic(base_size = 9.5) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10), axis.text.y = element_text(size = 8.5), axis.text.x = element_text(size = 8.5))
}

panel_d_attenuation <- function(att_df) {
  model_levels <- c("Group only", "Group + motor", "Group + motor + mechanisms")
  plot_df <- att_df %>% mutate(model_label = factor(as.character(model), levels = model_levels))
  ggplot(plot_df, aes(x = model_label, y = group_beta)) +
    geom_hline(yintercept = 0, colour = "#B8B8B8", linewidth = 0.4) +
    geom_errorbar(aes(ymin = group_ci_lower, ymax = group_ci_upper), width = 0.12, linewidth = 0.55) +
    geom_point(aes(size = r_squared), fill = "#7E6B8F", colour = "black", shape = 21, stroke = 0.35) +
    geom_text(aes(label = paste0(round(attenuation_pct), "%")), vjust = -1.0, size = 2.8) +
    scale_size_continuous(range = c(2.5, 4.2), guide = "none") +
    labs(title = "Group-effect attenuation", x = NULL, y = "Patient-control beta") +
    theme_classic(base_size = 9.5) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10), axis.text.x = element_text(angle = 25, hjust = 1, size = 8), axis.text.y = element_text(size = 8.5))
}

write_summary <- function(group_stats, shapley_df, att_df) {
  corr_values <- read_csv(file.path(output_dir, "section3_mechanism_correlation_matrix.csv"), show_col_types = FALSE) %>%
    filter(mechanism_1 != mechanism_2) %>%
    mutate(abs_rho = abs(spearman_rho))
  max_abs_corr <- max(corr_values$abs_rho, na.rm = TRUE)
  median_abs_corr <- median(corr_values$abs_rho, na.rm = TRUE)
  top_shap <- shapley_df %>% slice(1)
  full_att <- att_df %>% filter(as.character(model) == "Group + motor + mechanisms") %>% slice(1)

  lines <- c(
    "# Section 3. Cognitive mechanisms",
    "",
    "## Outputs",
    "",
    "- Figure PNG: `paper/outputs/section3/figure4_cognitive_mechanisms.png`",
    "- Figure PDF: `paper/outputs/section3/figure4_cognitive_mechanisms.pdf`",
    "- Mechanism correlations: `paper/outputs/section3/section3_mechanism_correlation_matrix.csv`",
    "- Mechanism group stats: `paper/outputs/section3/section3_mechanism_group_stats.csv`",
    "- Shapley dominance: `paper/outputs/section3/section3_shapley_residualised_uniqueness.csv`",
    "- Group-effect attenuation: `paper/outputs/section3/section3_group_effect_attenuation.csv`",
    "",
    "## Paragraph Draft",
    "",
    paste0(
      "The four mechanism measures were mostly weakly to moderately inter-correlated in patients (median absolute Spearman rho = ", fmt(median_abs_corr, 2),
      "; maximum absolute rho = ", fmt(max_abs_corr, 2), "). ",
      "Across the full complete-case sample, dominance analysis predicting motor-residualised uniqueness showed that the four mechanisms explained R2 = ", fmt(unique(shapley_df$total_r_squared), 3),
      " of the variance (n = ", unique(shapley_df$n), "), with the largest Shapley contribution from ", top_shap$label,
      " (Shapley R2 = ", fmt(top_shap$shapley_r_squared, 3), "). ",
      "Adding motor speed and the four mechanisms attenuated the patient-control uniqueness effect by ", fmt(full_att$attenuation_pct, 1),
      "% (group beta = ", fmt(full_att$group_beta), ", ", format_p(full_att$group_p), ")."
    )
  )
  writeLines(lines, file.path(output_dir, "section3_cognitive_mechanisms_summary.md"))
}

rows <- read_section3_data()
group_stats <- compare_groups(rows)
corr_df <- make_correlation_matrix(rows)
plot_df <- make_raincloud_data(rows)
shapley_df <- shapley_r2(rows, "test_uniqueness_motor_residualised", mechanisms$metric)
write_csv(shapley_df, file.path(output_dir, "section3_shapley_residualised_uniqueness.csv"))
att_df <- make_attenuation(rows)

p_a <- panel_a_corr(corr_df)
p_b <- panel_c_shapley(shapley_df)
p_c <- panel_b_groups(plot_df)
p_d <- panel_d_attenuation(att_df)

combined <- plot_grid(
  p_a, p_b,
  p_c, p_d,
  labels = c("A", "B", "C", "D"),
  ncol = 2,
  rel_heights = c(1.05, 0.95),
  label_size = 16,
  label_fontface = "bold"
)

ggsave(file.path(output_dir, "figure4_cognitive_mechanisms.png"), combined, width = 10.8, height = 7.4, dpi = 300, bg = "white")
ggsave(file.path(output_dir, "figure4_cognitive_mechanisms.pdf"), combined, width = 10.8, height = 7.4, device = cairo_pdf, bg = "white")
write_summary(group_stats, shapley_df, att_df)

message("Wrote section 3 outputs to ", output_dir)
