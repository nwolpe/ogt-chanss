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

output_dir <- file.path("paper", "outputs", "section4")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

section2_path <- file.path("paper", "outputs", "section2", "section2_participant_metrics.csv")
mechanism_path <- file.path("analysis_outputs", "phaseB_exploration_dynamics", "phaseB_exploration_dynamics_participant_metrics.csv")
clinical_path <- "CHANSS_clinical_demog.csv"
medication_path <- file.path("paper", "CHANSS_clinical_demog_filled_APLoad.xlsx")
corrections_path <- file.path("analysis_outputs", "ogt_patients_missing_clinical_data_corrected.csv")

mechanisms <- tibble::tribble(
  ~metric, ~label,
  "roaming_entropy_normalised", "Roaming entropy",
  "submovements_mean_per_path", "Option elaboration",
  "perseveration_ratio", "Perseveration",
  "temporal_prefix_uniqueness_spearman_rho", "Novelty maintenance"
)

clinical_specs <- tibble::tribble(
  ~metric, ~label,
  "test_uniqueness_motor_residualised", "Residualised uniqueness",
  "AMI_behavioural", "AMI behavioural",
  "BACS_VF_t", "BACS verbal fluency",
  "CDS_total", "CDS total",
  "AP_BNFmax", "Antipsychotic burden",
  "AP_Load2026", "Antipsychotic load 2026"
)

cluster_colours <- c("1" = "#6E1F18", "2" = "#E66A22", "3" = "#F2C14E")

to_num <- function(x) suppressWarnings(as.numeric(x))
sum_observed_items <- function(items) {
  items <- as.data.frame(items)
  out <- rowSums(items, na.rm = TRUE)
  out[rowSums(!is.na(items)) == 0] <- NA_real_
  out
}
fmt <- function(x, digits = 2) formatC(x, digits = digits, format = "f", big.mark = ",")
format_p <- function(p) {
  if (!is.finite(p)) return("")
  if (p < 0.001) return("p < .001")
  sub("^0", "", sprintf("p = %.3f", p))
}

bh <- function(p) {
  out <- rep(NA_real_, length(p))
  ok <- which(is.finite(p))
  if (!length(ok)) return(out)
  out[ok] <- p.adjust(p[ok], method = "BH")
  out
}

load_clinical <- function(path) {
  clinical <- read_csv(path, show_col_types = FALSE, name_repair = "unique_quiet")
  tol_cols <- grep("^BACS_ToL_t", names(clinical), value = TRUE)
  tol_values <- if (length(tol_cols)) clinical[[tol_cols[[1]]]] else NA_real_

  clinical %>%
    mutate(
      ID_subject = toupper(trimws(ID_subject)),
      BACS_ToL_t_clean = to_num(tol_values),
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
    transmute(
      ID_subject,
      EXP,
      MAP,
      AMI_behavioural,
      AMI_social,
      AMI_emotional,
      PSP_score = to_num(PSP_score),
      BACS_VM_t = to_num(BACS_VM_t),
      BACS_DS_t = to_num(BACS_DS_t),
      BACS_TMT_t = to_num(BACS_TMT_t),
      BACS_VF_t = to_num(BACS_VF_t),
      BACS_SC_t = to_num(BACS_SC_t),
      BACS_ToL_t = BACS_ToL_t_clean,
      CDS_total,
      PANSS_marder_positive
    ) %>%
    distinct(ID_subject, .keep_all = TRUE)
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

read_section4_data <- function() {
  paper_rows <- read_csv(section2_path, show_col_types = FALSE, guess_max = 10000) %>%
    mutate(group = as.character(group), participant_id = as.character(participant_id), source_id = as.character(source_id))

  mechanism_rows <- read_csv(mechanism_path, show_col_types = FALSE, guess_max = 10000) %>%
    mutate(group = as.character(group), participant_id = as.character(participant_id), source_id = as.character(source_id)) %>%
    select(group, participant_id, source_id, mechanism_residualised_uniqueness = test_uniqueness_residualised, all_of(mechanisms$metric))

  clinical <- load_clinical(clinical_path)
  medication <- load_medication_burden(medication_path)
  corrections <- if (file.exists(corrections_path)) {
    read_csv(corrections_path, show_col_types = FALSE) %>%
      transmute(participant_id_upper = toupper(trimws(participant_id)), clinical_lookup_id = toupper(trimws(participant_id_corrected)))
  } else {
    tibble(participant_id_upper = character(), clinical_lookup_id = character())
  }

  rows <- paper_rows %>%
    left_join(mechanism_rows, by = c("group", "participant_id", "source_id")) %>%
    filter(group == "patient") %>%
    mutate(
      participant_id_upper = toupper(trimws(participant_id)),
      clinical_lookup_id = coalesce(corrections$clinical_lookup_id[match(participant_id_upper, corrections$participant_id_upper)], participant_id_upper)
    ) %>%
    left_join(clinical, by = c("clinical_lookup_id" = "ID_subject")) %>%
    left_join(medication, by = c("clinical_lookup_id" = "ID_subject"))

  write_csv(
    tibble(
      item = c("patients", "mechanism_complete_patients"),
      n = c(nrow(rows), sum(complete.cases(rows[, mechanisms$metric])))
    ),
    file.path(output_dir, "section4_sample_audit.csv")
  )
  rows
}

cluster_rows <- function(rows, selected_k = 3, seed = 42) {
  complete <- rows %>%
    filter(if_all(all_of(mechanisms$metric), ~ is.finite(to_num(.x))))
  X <- complete %>% select(all_of(mechanisms$metric)) %>% mutate(across(everything(), to_num)) %>% as.matrix()
  means <- colMeans(X)
  sds <- apply(X, 2, sd)
  sds[!is.finite(sds) | sds <= 0] <- 1
  Z <- scale(X, center = means, scale = sds)

  model_rows <- bind_rows(lapply(1:6, function(k) {
    set.seed(seed)
    if (k == 1) {
      km <- kmeans(Z, centers = 1, nstart = 100)
      silhouette <- NA_real_
    } else {
      km <- kmeans(Z, centers = k, nstart = 100)
      silhouette <- mean(cluster::silhouette(km$cluster, dist(Z))[, "sil_width"])
    }
    tibble(
      k = k,
      total_withinss = km$tot.withinss,
      between_ss = km$betweenss,
      silhouette = silhouette,
      selected = k == selected_k
    )
  }))
  write_csv(model_rows, file.path(output_dir, "section4_kmeans_model_selection.csv"))

  set.seed(seed)
  km <- kmeans(Z, centers = selected_k, nstart = 500)
  raw_clusters <- km$cluster
  roaming_idx <- match("roaming_entropy_normalised", mechanisms$metric)
  old_order <- tapply(Z[, roaming_idx], raw_clusters, mean) %>% sort(decreasing = TRUE) %>% names() %>% as.integer()
  remap <- setNames(seq_along(old_order), old_order)
  clusters <- unname(remap[as.character(raw_clusters)])
  # Final cluster labels are ordered by mean residualised uniqueness:
  # cluster 1 = highest, cluster 2 = middle, cluster 3 = lowest.
  clusters <- dplyr::recode(as.integer(clusters), `2` = 1L, `1` = 2L, `3` = 3L)

  pca <- prcomp(Z, center = FALSE, scale. = FALSE)
  scores <- as_tibble(pca$x[, 1:2]) %>%
    rename(PC1 = PC1, PC2 = PC2)

  complete <- bind_cols(complete, scores) %>%
    mutate(kmeans_cluster = factor(clusters, levels = 1:selected_k))

  std <- tibble(metric = mechanisms$metric, label = mechanisms$label, mean = as.numeric(means), sd = as.numeric(sds))
  write_csv(std, file.path(output_dir, "section4_kmeans_standardisation.csv"))
  write_csv(complete, file.path(output_dir, "section4_kmeans_memberships.csv"))

  profiles <- complete %>%
    select(kmeans_cluster, all_of(mechanisms$metric)) %>%
    mutate(across(all_of(mechanisms$metric), to_num)) %>%
    mutate(across(all_of(mechanisms$metric), ~ (.x - means[cur_column()]) / sds[cur_column()])) %>%
    group_by(kmeans_cluster) %>%
    summarise(n = n(), across(all_of(mechanisms$metric), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    pivot_longer(all_of(mechanisms$metric), names_to = "metric", values_to = "mean_z") %>%
    left_join(mechanisms, by = "metric")
  write_csv(profiles, file.path(output_dir, "section4_kmeans_cluster_profiles.csv"))

  list(rows = complete, model = model_rows, profiles = profiles, pca = pca)
}

kruskal_table <- function(rows, specs, cluster_col = "kmeans_cluster") {
  out <- bind_rows(lapply(seq_len(nrow(specs)), function(i) {
    metric <- specs$metric[[i]]
    label <- specs$label[[i]]
    vals <- rows %>%
      select(all_of(cluster_col), all_of(metric)) %>%
      rename(cluster = all_of(cluster_col), value = all_of(metric)) %>%
      mutate(value = to_num(value)) %>%
      filter(is.finite(value), !is.na(cluster))
    groups <- split(vals$value, vals$cluster)
    test <- if (length(groups) >= 2) kruskal.test(groups) else NULL
    tibble(
      metric = metric,
      label = label,
      n = nrow(vals),
      kruskal_h = if (is.null(test)) NA_real_ else unname(test$statistic),
      kruskal_p = if (is.null(test)) NA_real_ else test$p.value
    )
  }))
  out %>% mutate(kruskal_q_fdr = bh(kruskal_p))
}

profile_clinical <- function(clustered) {
  rows <- clustered$rows
  stats <- kruskal_table(rows, clinical_specs)
  write_csv(stats, file.path(output_dir, "section4_cluster_clinical_kruskal.csv"))

  profiles <- rows %>%
    select(kmeans_cluster, all_of(clinical_specs$metric)) %>%
    pivot_longer(all_of(clinical_specs$metric), names_to = "metric", values_to = "value") %>%
    mutate(value = to_num(value)) %>%
    filter(is.finite(value)) %>%
    left_join(clinical_specs, by = "metric") %>%
    group_by(metric) %>%
    mutate(value_z = as.numeric(scale(value))) %>%
    ungroup() %>%
    group_by(kmeans_cluster, metric, label) %>%
    summarise(n = n(), mean = mean(value), sd = sd(value), median = median(value), mean_z = mean(value_z), .groups = "drop")
  write_csv(profiles, file.path(output_dir, "section4_cluster_clinical_profiles.csv"))
  list(profiles = profiles, stats = stats)
}

cluster_labels <- function(rows) {
  paste0("Cluster ", levels(rows$kmeans_cluster), " (n=", as.integer(table(rows$kmeans_cluster)), ")")
}

clinical_raincloud_data <- function(clustered) {
  clustered$rows %>%
    select(kmeans_cluster, all_of(clinical_specs$metric)) %>%
    pivot_longer(all_of(clinical_specs$metric), names_to = "metric", values_to = "value") %>%
    mutate(value = to_num(value)) %>%
    filter(is.finite(value)) %>%
    left_join(clinical_specs, by = "metric") %>%
    group_by(metric) %>%
    mutate(value_z = as.numeric(scale(value))) %>%
    ungroup() %>%
    mutate(
      label = factor(label, levels = clinical_specs$label),
      kmeans_cluster = factor(kmeans_cluster, levels = 1:3),
      x_center = as.numeric(kmeans_cluster),
      x_violin = x_center - 0.10,
      x_box = x_center,
      x_point = x_center + 0.13
    )
}

half_density_data <- function(plot_df, value_col = "value_z", group_cols = c("label", "kmeans_cluster")) {
  split_key <- interaction(plot_df[, group_cols], drop = TRUE)
  bind_rows(lapply(split(plot_df, split_key), function(df) {
    vals <- df[[value_col]]
    vals <- vals[is.finite(vals)]
    if (length(vals) < 2 || !is.finite(sd(vals)) || sd(vals) == 0) return(tibble())
    den <- density(vals, n = 256, from = min(vals), to = max(vals), na.rm = TRUE)
    x_anchor <- unique(df$x_violin)[1]
    half_width <- 0.22
    left <- x_anchor - (den$y / max(den$y, na.rm = TRUE)) * half_width
    tibble(
      label = df$label[1],
      kmeans_cluster = df$kmeans_cluster[1],
      x = c(left, rep(x_anchor, length(left))),
      y = c(den$x, rev(den$x))
    )
  }))
}

plot_kmeans_profile <- function(clustered, profiled) {
  model_df <- clustered$model
  rows <- clustered$rows
  profiles <- clustered$profiles %>%
    mutate(label = factor(label, levels = mechanisms$label))
  clinical_df <- clinical_raincloud_data(clustered)
  clinical_density <- half_density_data(clinical_df)
  var_exp <- clustered$pca$sdev^2 / sum(clustered$pca$sdev^2)

  p_a <- ggplot(model_df, aes(x = k)) +
    geom_line(aes(y = silhouette), colour = "#333333", linewidth = 0.65, na.rm = TRUE) +
    geom_point(aes(y = silhouette), colour = "#333333", size = 2, na.rm = TRUE) +
    geom_vline(xintercept = 3, colour = "#B6463A", linewidth = 0.6) +
    scale_x_continuous(breaks = 1:6) +
    labs(title = "Model selection", x = "Number of k-means clusters", y = "Mean silhouette width") +
    theme_classic(base_size = 9.5) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0, size = 11)
    )

  p_b <- ggplot(rows, aes(x = PC1, y = PC2, colour = kmeans_cluster)) +
    geom_point(size = 2.2, alpha = 0.75) +
    scale_colour_manual(values = cluster_colours, labels = cluster_labels(rows), name = NULL) +
    labs(title = "Patient mechanism space", x = paste0("PC1 (", fmt(100 * var_exp[1], 1), "%)"), y = paste0("PC2 (", fmt(100 * var_exp[2], 1), "%)")) +
    theme_classic(base_size = 9.5) +
    theme(plot.title = element_text(face = "bold", hjust = 0, size = 11), legend.position = "top")

  p_c <- ggplot(profiles, aes(x = label, y = mean_z, fill = kmeans_cluster)) +
    geom_hline(yintercept = 0, colour = "#999999", linewidth = 0.35) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    scale_fill_manual(values = cluster_colours, labels = cluster_labels(rows), name = NULL) +
    labs(title = "K-means mechanism profiles", x = NULL, y = "Mean mechanism score (z)") +
    theme_classic(base_size = 9.5) +
    theme(plot.title = element_text(face = "bold", hjust = 0, size = 11), axis.text.x = element_text(angle = 25, hjust = 1, size = 8), legend.position = "none")

  p_d <- ggplot() +
    geom_hline(yintercept = 0, colour = "#B8B8B8", linewidth = 0.3) +
    geom_polygon(data = clinical_density, aes(x = x, y = y, fill = kmeans_cluster, group = interaction(label, kmeans_cluster)), alpha = 0.32, colour = NA) +
    geom_boxplot(data = clinical_df, aes(x = x_box, y = value_z, fill = kmeans_cluster, group = interaction(label, kmeans_cluster)), width = 0.16, outlier.shape = NA, colour = "black", linewidth = 0.35, alpha = 0.82) +
    geom_point(data = clinical_df, aes(x = x_point, y = value_z, colour = kmeans_cluster), position = position_jitter(width = 0.035, height = 0, seed = 11), size = 0.55, alpha = 0.34, stroke = 0) +
    facet_wrap(~ label, nrow = 1) +
    scale_x_continuous(breaks = 1:3, labels = paste0("C", 1:3), limits = c(0.42, 3.42)) +
    scale_fill_manual(values = cluster_colours, labels = cluster_labels(rows), name = NULL) +
    scale_colour_manual(values = cluster_colours, labels = cluster_labels(rows), name = NULL) +
    labs(title = "Clinical profiles across k-means clusters", x = NULL, y = "Clinical measure (z)") +
    theme_classic(base_size = 9.5) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0, size = 11),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 8),
      axis.text.x = element_text(size = 8),
      axis.text.y = element_text(size = 8),
      legend.position = "none",
      panel.spacing.x = unit(1.0, "lines")
    )

  top <- plot_grid(p_a, p_b, labels = c("A", "B"), ncol = 2, label_size = 15, label_fontface = "bold")
  middle <- plot_grid(p_c, labels = "C", ncol = 1, label_size = 15, label_fontface = "bold")
  bottom <- plot_grid(p_d, labels = "D", ncol = 1, label_size = 15, label_fontface = "bold")
  title <- ggdraw() + draw_label("Hard k-means clustering of four patient mechanism measures (K = 3)", fontface = "bold", size = 14)
  combined <- plot_grid(title, top, middle, bottom, ncol = 1, rel_heights = c(0.07, 1, 0.78, 0.9))
  ggsave(file.path(output_dir, "figure5_kmeans_mechanism_profiles.png"), combined, width = 12.0, height = 10.0, dpi = 300, bg = "white")
  ggsave(file.path(output_dir, "figure5_kmeans_mechanism_profiles.pdf"), combined, width = 12.0, height = 10.0, device = cairo_pdf, bg = "white")
}

# plot_cluster_clinical <- function(clustered, profiled) {
#   profiles <- profiled$profiles %>%
#     left_join(profiled$stats %>% select(metric, kruskal_p, kruskal_q_fdr), by = "metric") %>%
#     mutate(
#       label = factor(label, levels = rev(clinical_specs$label)),
#       kmeans_cluster = factor(kmeans_cluster, levels = 1:3)
#     )
# 
#   p <- ggplot(profiles, aes(x = kmeans_cluster, y = label, fill = mean_z)) +
#     geom_tile(colour = "white", linewidth = 0.65) +
#     geom_text(aes(label = paste0(sprintf("%+.2f", mean_z), if_else(is.finite(kruskal_q_fdr) & kruskal_q_fdr < 0.05, "**", if_else(is.finite(kruskal_p) & kruskal_p < 0.05, "*", "")))), size = 2.8) +
#     scale_fill_gradient2(low = "#5A7D9A", mid = "white", high = "#D67C59", midpoint = 0, limits = c(-0.9, 0.9), oob = squish, name = "Cluster mean z") +
#     scale_x_discrete(labels = paste0("Cluster ", 1:3, "\n(n=", as.integer(table(clustered$rows$kmeans_cluster)), ")")) +
#     labs(title = "Residualised uniqueness and clinical profiles across k-means clusters", x = NULL, y = NULL) +
#     theme_minimal(base_size = 10) +
#     theme(
#       plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
#       axis.text.y = element_text(size = 8.8),
#       axis.text.x = element_text(size = 9),
#       panel.grid = element_blank()
#     )
#   ggsave(file.path(output_dir, "figure6_kmeans_cluster_clinical_profiles.png"), p, width = 8.0, height = 3.4, dpi = 300, bg = "white")
#   ggsave(file.path(output_dir, "figure6_kmeans_cluster_clinical_profiles.pdf"), p, width = 8.0, height = 3.4, device = cairo_pdf, bg = "white")
# }

write_summary <- function(clustered, profiled) {
  sizes <- as.integer(table(clustered$rows$kmeans_cluster))
  selected <- clustered$model %>% filter(k == 3) %>% slice(1)
  sig <- profiled$stats %>% filter(is.finite(kruskal_p), kruskal_p < 0.05) %>% arrange(kruskal_p)
  lines <- c(
    "# Section 4. Mechanism profiles",
    "",
    "## Outputs",
    "",
    "- Combined k-means mechanism and clinical profile figure: `paper/outputs/section4/figure5_kmeans_mechanism_profiles.png`",
    "- K-means memberships: `paper/outputs/section4/section4_kmeans_memberships.csv`",
    "- Clinical Kruskal tests: `paper/outputs/section4/section4_cluster_clinical_kruskal.csv`",
    "",
    "## Summary",
    "",
    paste0("K-means clustering was run on ", nrow(clustered$rows), " complete-case patients using four mechanisms. Cluster sizes were ", paste(sizes, collapse = ", "), "."),
    paste0("For K = 3, mean silhouette = ", fmt(selected$silhouette, 3), ".")
  )
  if (nrow(sig)) {
    lines <- c(
      lines,
      "",
      "Nominal cluster differences:",
      paste0(
        "- ", sig$label,
        ": H = ", vapply(sig$kruskal_h, fmt, character(1)),
        ", ", vapply(sig$kruskal_p, format_p, character(1)),
        ", q = ", vapply(sig$kruskal_q_fdr, fmt, character(1), digits = 3)
      )
    )
  } else {
    lines <- c(lines, "", "No cluster-profile variables showed nominal Kruskal p < .05.")
  }
  writeLines(lines, file.path(output_dir, "section4_kmeans_profiles_summary.md"))
}

rows <- read_section4_data()
clustered <- cluster_rows(rows, selected_k = 3)
profiled <- profile_clinical(clustered)
plot_kmeans_profile(clustered, profiled)
write_summary(clustered, profiled)

message("Wrote section 4 outputs to ", output_dir)
