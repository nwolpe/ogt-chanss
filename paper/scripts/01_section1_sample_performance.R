#!/usr/bin/env Rscript

if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a  # compatibility with older R/ggplot2
suppressPackageStartupMessages({
  library(cowplot)
  library(dplyr)
  library(flextable)
  library(ggplot2)
  library(jsonlite)
  library(officer)
  library(readr)
  library(scales)
  library(tidyr)
})

output_dir <- file.path("paper", "outputs", "section1")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

participant_metrics_path <- file.path("analysis_outputs", "ogt_participant_metrics.csv")
control_demographics_path <- "control_demographics_sorted.csv"
clinical_path <- "CHANSS_clinical_demog.csv"
corrections_path <- file.path("analysis_outputs", "ogt_patients_missing_clinical_data_corrected.csv")

control_fill <- "#7FA9C6"
patient_fill <- "#D67C59"
control_point <- "#5A7D9A"
patient_point <- "#C26D52"

parse_jatos_file <- function(path) {
  if (is.na(path) || !nzchar(path) || !file.exists(path)) {
    return(tibble())
  }
  txt <- paste(readLines(path, warn = FALSE), collapse = "")
  if (!nzchar(trimws(txt))) {
    return(tibble())
  }
  txt <- gsub("\\]\\[", ",", txt)
  if (startsWith(txt, "{\"round\"")) {
    txt <- paste0("[", gsub("\\}\\{", "},{", txt), "]")
  }
  payload <- tryCatch(fromJSON(txt, flatten = TRUE), error = function(e) NULL)
  if (!is.data.frame(payload) || !all(c("round", "path", "x", "y", "time_since_round_started", "complete") %in% names(payload))) {
    return(tibble())
  }
  tibble(
    round = as.integer(payload$round),
    route_index = as.integer(payload$path),
    time_ms = as.numeric(payload$time_since_round_started),
    x = as.numeric(payload$x),
    y = as.numeric(payload$y),
    complete = as.integer(payload$complete)
  )
}

control_task_file <- function(source_id) {
  candidates <- list.files(
    file.path("data_controls", source_id),
    pattern = "data[.]txt$",
    recursive = TRUE,
    full.names = TRUE
  )
  best_file <- NA_character_
  best_n <- -Inf
  for (candidate in candidates) {
    pts <- parse_jatos_file(candidate)
    n_test <- if (nrow(pts) && all(c("round", "complete") %in% names(pts))) {
      sum(pts$round == 1 & pts$complete == 1, na.rm = TRUE)
    } else {
      0
    }
    if (nrow(pts) && n_test > best_n) {
      best_file <- candidate
      best_n <- n_test
    }
  }
  best_file
}

route_summary <- function(points) {
  if (!nrow(points)) {
    return(tibble(
      route_count = 0,
      valid_route_count = 0,
      session_duration_ms = NA_real_,
      total_draw_time_ms = NA_real_,
      mean_path_duration_ms = NA_real_,
      mean_interpath_interval_ms = NA_real_,
      total_distance_px = NA_real_,
      mean_path_length_px = NA_real_,
      mean_speed_px_per_s = NA_real_
    ))
  }
  routes <- split(points[order(points$route_index, points$time_ms), ], points$route_index)
  starts <- vapply(routes, function(r) r$time_ms[1], numeric(1))
  ends <- vapply(routes, function(r) r$time_ms[nrow(r)], numeric(1))
  durations <- ends - starts
  interpath <- if (length(routes) > 1) starts[-1] - ends[-length(ends)] else numeric(0)
  path_lengths <- vapply(routes, function(r) {
    if (nrow(r) < 2) return(0)
    sum(sqrt(diff(r$x)^2 + diff(r$y)^2), na.rm = TRUE)
  }, numeric(1))
  total_distance <- sum(path_lengths, na.rm = TRUE)
  total_draw_time <- sum(durations, na.rm = TRUE)
  tibble(
    route_count = length(routes),
    valid_route_count = sum(vapply(routes, nrow, integer(1)) > 1),
    session_duration_ms = max(ends, na.rm = TRUE) - min(starts, na.rm = TRUE),
    total_draw_time_ms = total_draw_time,
    mean_path_duration_ms = mean(durations, na.rm = TRUE),
    mean_interpath_interval_ms = if (length(interpath)) mean(interpath, na.rm = TRUE) else NA_real_,
    total_distance_px = total_distance,
    mean_path_length_px = mean(path_lengths, na.rm = TRUE),
    mean_speed_px_per_s = if (is.finite(total_draw_time) && total_draw_time > 0) total_distance / (total_draw_time / 1000) else NA_real_
  )
}

strip_session_suffix <- function(participant_id) {
  m <- regexec("^(.*?)(?:_(0*1|0*2))$", participant_id)
  hit <- regmatches(participant_id, m)[[1]]
  if (length(hit) == 0) {
    return(list(base = participant_id, session = NA_integer_))
  }
  list(base = hit[[2]], session = as.integer(hit[[3]]))
}

select_controls_include_second_only <- function(control_demo) {
  parsed <- lapply(control_demo$subID, strip_session_suffix)
  control_demo$base_id <- vapply(parsed, `[[`, character(1), "base")
  control_demo$session_number <- vapply(parsed, `[[`, integer(1), "session")

  unsuffixed <- control_demo %>% filter(is.na(session_number))
  suffixed <- control_demo %>%
    filter(!is.na(session_number)) %>%
    arrange(base_id, session_number) %>%
    group_by(base_id) %>%
    slice(if (any(session_number == 1L)) which(session_number == 1L)[1] else which(session_number == 2L)[1]) %>%
    ungroup()

  bind_rows(unsuffixed, suffixed) %>% arrange(study_result_id)
}

build_extra_control_rows <- function(existing_metrics, selected_controls) {
  existing_control_keys <- existing_metrics %>%
    filter(group == "control") %>%
    transmute(study_result_id = as.character(source_id), participant_id = as.character(participant_id))

  needed <- selected_controls %>%
    anti_join(existing_control_keys, by = c("study_result_id", "subID" = "participant_id"))

  rows <- list()
  audit <- list()
  for (i in seq_len(nrow(needed))) {
    row <- needed[i, ]
    task_file <- control_task_file(row$study_result_id)
    pts <- if (!is.na(task_file)) parse_jatos_file(task_file) else tibble()
    pts <- pts %>%
      filter(complete == 1, is.finite(x), is.finite(y), is.finite(time_ms)) %>%
      mutate(condition = case_when(round == 0 ~ "practice", round == 1 ~ "test", TRUE ~ NA_character_)) %>%
      filter(!is.na(condition))

    practice <- route_summary(pts %>% filter(condition == "practice"))
    test <- route_summary(pts %>% filter(condition == "test"))

    out <- tibble(
      group = "control",
      participant_id = row$subID,
      source_id = row$study_result_id,
      source_platform = "JATOS",
      raw_source = "",
      raw_practice_file = "",
      raw_test_file = "",
      raw_task_file = task_file,
      coordinate_transform = "none",
      coordinate_calibration_method = "control_native_jatos",
      calibration_practice_routes = NA_real_,
      calibration_x_center_raw = NA_real_,
      calibration_y_top_raw = NA_real_,
      calibration_y_bottom_raw = NA_real_,
      calibration_raw_width_estimate = NA_real_,
      calibration_raw_height_from_span = NA_real_,
      calibration_x_scale = NA_real_,
      calibration_y_scale = NA_real_,
      has_practice = as.integer(practice$route_count > 0),
      has_test = as.integer(test$route_count > 0)
    )
    for (name in names(practice)) out[[paste0("practice_", name)]] <- practice[[name]]
    for (name in names(test)) out[[paste0("test_", name)]] <- test[[name]]
    out <- out %>%
      mutate(
        test_uniqueness_mean = NA_real_,
        test_log_uniqueness_mean = NA_real_,
        test_mds_hull_area = NA_real_,
        test_uniqueness_route_count = NA_real_,
        test_expected_duration_ms = 120000,
        test_short_threshold_ms = 115000,
        test_duration_gap_ms = test_expected_duration_ms - test_session_duration_ms,
        test_duration_gap_s = test_duration_gap_ms / 1000,
        flag_test_short_under_115s = as.integer(is.finite(test_session_duration_ms) & test_session_duration_ms < 115000),
        study_result_id = row$study_result_id,
        component_result_id = row$component_result_id,
        age = row$age,
        sex = row$sex,
        hand = row$hand,
        timestamp_utc = row$timestamp_utc,
        timestamp_local = row$timestamp_local
      )

    rows[[length(rows) + 1]] <- out
    audit[[length(audit) + 1]] <- tibble(
      participant_id = row$subID,
      source_id = row$study_result_id,
      reason = "included_session_2_only_control_exception",
      raw_task_file = task_file,
      practice_route_count = practice$route_count,
      test_route_count = test$route_count
    )
  }

  list(rows = bind_rows(rows), audit = bind_rows(audit))
}

to_num <- function(x) suppressWarnings(as.numeric(x))
sum_observed_items <- function(items) {
  items <- as.data.frame(items)
  out <- rowSums(items, na.rm = TRUE)
  out[rowSums(!is.na(items)) == 0] <- NA_real_
  out
}
finite_values <- function(df, field) {
  vals <- to_num(df[[field]])
  vals[is.finite(vals)]
}
mean_sd <- function(df, field, denom = nrow(df), digits = 1) {
  vals <- finite_values(df, field)
  if (!length(vals)) return("")
  suffix <- if (length(vals) < denom) paste0(", n = ", length(vals)) else ""
  paste0(sprintf(paste0("%.", digits, "f"), mean(vals)), " (", sprintf(paste0("%.", digits, "f"), sd(vals)), ")", suffix)
}
n_percent <- function(n, denom) {
  if (!denom) return("")
  paste0(n, " (", sprintf("%.1f", 100 * n / denom), "%)")
}
counts_with_n <- function(parts, denom, labels = NULL) {
  value <- paste(parts, collapse = " / ")
  if (!is.null(labels)) value <- paste(value, paste(labels, collapse = " / "))
  observed <- sum(parts)
  if (observed < denom) value <- paste0(value, ", n = ", observed)
  value
}
format_p <- function(p) {
  if (!is.finite(p)) return("")
  if (p < 0.001) return("p < .001")
  sub("^0", "", sprintf("p = %.3f", p))
}
format_t <- function(a, b) {
  a <- a[is.finite(a)]
  b <- b[is.finite(b)]
  if (length(a) < 2 || length(b) < 2) return("")
  tt <- t.test(a, b, var.equal = FALSE)
  paste0("t = ", sprintf("%.2f", unname(tt$statistic)), ", ", format_p(tt$p.value))
}
format_chi2 <- function(tab) {
  if (any(rowSums(tab) == 0) || any(colSums(tab) == 0)) return("")
  out <- suppressWarnings(chisq.test(tab, correct = FALSE))
  paste0("chi2 = ", sprintf("%.2f", unname(out$statistic)), ", ", format_p(out$p.value))
}

load_clinical <- function(path) {
  read_csv(path, show_col_types = FALSE, name_repair = "unique_quiet") %>%
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
      MAP = if_else(is.nan(MAP), NA_real_, as.numeric(MAP)),
      EXP = if_else(is.nan(EXP), NA_real_, as.numeric(EXP)),
      CDS_total = if_else(is.nan(CDS_total), NA_real_, as.numeric(CDS_total)),
      PANSS_marder_positive = if_else(is.nan(PANSS_marder_positive), NA_real_, as.numeric(PANSS_marder_positive))
    ) %>%
    distinct(ID_subject, .keep_all = TRUE)
}

patient_site <- function(platform, subject_id) {
  case_when(
    platform == "OGT_Mandarin" | grepl("^CH", subject_id) ~ "China",
    platform == "OGT_Spanish" | grepl("^C[378]", subject_id) ~ "Spain",  # Barcelona, Bilbao, Santander
    platform %in% c("OGT_English", "OCTAL") | grepl("^C", subject_id) ~ "UK",
    TRUE ~ "Unknown"
  )
}

read_metric_data <- function() {
  metrics <- read_csv(participant_metrics_path, show_col_types = FALSE, guess_max = 10000)
  control_demo <- read_csv(control_demographics_path, show_col_types = FALSE)
  selected_controls <- select_controls_include_second_only(control_demo)
  extras <- build_extra_control_rows(metrics, selected_controls)
  all_metrics <- bind_rows(metrics, extras$rows) %>%
    mutate(group = as.character(group), participant_id = as.character(participant_id), source_id = as.character(source_id))

  inclusion_audit <- tibble(
    item = c("existing_controls", "added_controls", "final_controls", "final_patients"),
    n = c(
      sum(metrics$group == "control"),
      nrow(extras$rows),
      sum(all_metrics$group == "control"),
      sum(all_metrics$group == "patient")
    )
  )
  if (nrow(extras$audit)) {
    write_csv(extras$audit, file.path(output_dir, "section1_added_control_exceptions.csv"))
  }
  write_csv(inclusion_audit, file.path(output_dir, "section1_inclusion_audit.csv"))
  write_csv(all_metrics, file.path(output_dir, "section1_participant_metrics.csv"))
  all_metrics
}

make_table1 <- function(metrics) {
  controls <- metrics %>% filter(group == "control")
  patient_task <- metrics %>% filter(group == "patient")
  clinical <- load_clinical(clinical_path)
  corrections <- if (file.exists(corrections_path)) {
    read_csv(corrections_path, show_col_types = FALSE) %>%
      transmute(participant_id = toupper(trimws(participant_id)), participant_id_corrected = toupper(trimws(participant_id_corrected)))
  } else {
    tibble(participant_id = character(), participant_id_corrected = character())
  }
  patients <- patient_task %>%
    mutate(
      participant_id_upper = toupper(trimws(participant_id)),
      clinical_lookup_id = coalesce(corrections$participant_id_corrected[match(participant_id_upper, corrections$participant_id)], participant_id_upper)
    ) %>%
    left_join(clinical, by = c("clinical_lookup_id" = "ID_subject")) %>%
    mutate(
      illness_duration = to_num(Age) - to_num(Age_FEP),
      is_fep_5y = if_else(is.finite(illness_duration) & illness_duration >= 0 & illness_duration <= 5, 1, 0, missing = NA_real_),
      is_trs = case_when(is.na(AP_1) | trimws(as.character(AP_1)) == "" ~ NA_real_, grepl("cloz", AP_1, ignore.case = TRUE) | grepl("cloz", AP_2, ignore.case = TRUE) ~ 1, TRUE ~ 0),
      site = patient_site(source_platform, clinical_lookup_id)
    )

  patient_matched <- patients %>% filter(!is.na(Age) | !is.na(Gender))
  n_patient <- nrow(patient_task)
  n_control <- nrow(controls)
  n_patient_clinical <- n_distinct(patient_matched$participant_id)
  patient_gender <- table(factor(if_else(as.character(patient_matched$Gender) == "1", "Male", "Female"), levels = c("Female", "Male")), useNA = "no")
  control_gender <- table(factor(tools::toTitleCase(trimws(as.character(controls$sex))), levels = c("Female", "Male")), useNA = "no")
  patient_sites <- table(factor(patient_matched$site, levels = c("UK", "Spain", "China")), useNA = "no")
  control_hand <- table(factor(tools::toTitleCase(trimws(as.character(controls$hand))), levels = c("Right", "Left")), useNA = "no")
  fep_known <- patients %>% filter(is.finite(is_fep_5y))
  trs_known <- patients %>% filter(is.finite(is_trs))

  rows <- tibble(section = character(), measure = character(), patients = character(), controls = character(), inferential_statistic = character())
  add <- function(section, measure, patients_value, controls_value = "", stat = "") {
    rows <<- bind_rows(rows, tibble(section = section, measure = measure, patients = patients_value, controls = controls_value, inferential_statistic = stat))
  }

  add("Sample", "OGT task sample", as.character(n_patient), as.character(n_control))
  add("Sample", "Matched patient clinical data", paste0(n_patient_clinical, "/", n_patient))
  if (n_patient_clinical < n_patient) add("Sample", "Unmatched patient clinical data", as.character(n_patient - n_patient_clinical))
  add("Demographic", "Age, years", mean_sd(patient_matched, "Age", n_patient), mean_sd(controls, "age", n_control), format_t(finite_values(patient_matched, "Age"), finite_values(controls, "age")))
  add("Demographic", "Sex, female / male", counts_with_n(as.integer(patient_gender), n_patient), counts_with_n(as.integer(control_gender), n_control), format_chi2(rbind(as.integer(patient_gender), as.integer(control_gender))))
  add("Demographic", "Patient site, UK / Spain / China", counts_with_n(as.integer(patient_sites), n_patient))
  add("Demographic", "Control handedness, right / left", "", counts_with_n(as.integer(control_hand), n_control))
  add("Demographic", "Education, years", mean_sd(patient_matched, "pEducation", n_patient))
  add("Illness", "Age at first episode, years", mean_sd(patient_matched, "Age_FEP", n_patient))
  add("Illness", "Illness duration, years", mean_sd(patients, "illness_duration", n_patient))
  add("Illness", "FEP within 5 years", paste0(n_percent(sum(fep_known$is_fep_5y == 1, na.rm = TRUE), nrow(fep_known)), ", n = ", nrow(fep_known)))
  add("Illness", "TRS / current clozapine", paste0(n_percent(sum(trs_known$is_trs == 1, na.rm = TRUE), nrow(trs_known)), ", n = ", nrow(trs_known)))
  add("Cognitive", "BACS verbal fluency T-score", mean_sd(patient_matched, "BACS_VF_t", n_patient))
  add("Cognitive", "BACS composite T-score", mean_sd(patient_matched, "BACS_comp_t", n_patient))
  add("Cognitive", "BACS digit sequencing T-score", mean_sd(patient_matched, "BACS_DS_t", n_patient))
  add("Cognitive", "BACS trail-making T-score", mean_sd(patient_matched, "BACS_TMT_t", n_patient))
  add("Clinical", "PANSS Marder positive", mean_sd(patient_matched, "PANSS_marder_positive", n_patient))
  add("Clinical", "BNSS MAP", mean_sd(patient_matched, "MAP", n_patient))
  add("Clinical", "BNSS EXP", mean_sd(patient_matched, "EXP", n_patient))
  add("Clinical", "AMI behavioural", mean_sd(patient_matched, "AMI_behavioural", n_patient))
  add("Clinical", "AMI social", mean_sd(patient_matched, "AMI_social", n_patient))
  add("Clinical", "AMI emotional", mean_sd(patient_matched, "AMI_emotional", n_patient))
  add("Clinical", "CDSS total", mean_sd(patient_matched, "CDS_total", n_patient))
  add("Clinical", "PSP score", mean_sd(patient_matched, "PSP_score", n_patient))
  add("Task", "Practice route count", mean_sd(patient_task, "practice_route_count", n_patient), mean_sd(controls, "practice_route_count", n_control), format_t(finite_values(patient_task, "practice_route_count"), finite_values(controls, "practice_route_count")))
  add("Task", "Practice mean speed, px/s", mean_sd(patient_task, "practice_mean_speed_px_per_s", n_patient), mean_sd(controls, "practice_mean_speed_px_per_s", n_control), format_t(finite_values(patient_task, "practice_mean_speed_px_per_s"), finite_values(controls, "practice_mean_speed_px_per_s")))
  add("Task", "Test route count", mean_sd(patient_task, "test_route_count", n_patient), mean_sd(controls, "test_route_count", n_control), format_t(finite_values(patient_task, "test_route_count"), finite_values(controls, "test_route_count")))
  add("Task", "Test mean speed, px/s", mean_sd(patient_task, "test_mean_speed_px_per_s", n_patient), mean_sd(controls, "test_mean_speed_px_per_s", n_control), format_t(finite_values(patient_task, "test_mean_speed_px_per_s"), finite_values(controls, "test_mean_speed_px_per_s")))

  write_csv(rows, file.path(output_dir, "table1_sample_characteristics.csv"))

  md <- c(
    "# Table 1. Demographic, clinical, and raw task-performance measures",
    "",
    paste0("| Measure | Patients (n = ", n_patient, ") | Controls (n = ", n_control, ") | Inferential statistic |"),
    "|---|---:|---:|---:|"
  )
  current <- ""
  for (i in seq_len(nrow(rows))) {
    if (rows$section[[i]] != current) {
      current <- rows$section[[i]]
      md <- c(md, paste0("| **", current, "** |  |  |  |"))
    }
    md <- c(md, paste0("| ", rows$measure[[i]], " | ", rows$patients[[i]], " | ", rows$controls[[i]], " | ", rows$inferential_statistic[[i]], " |"))
  }
  md <- c(
    md,
    "",
    "Values are mean (SD) or n (%). The OGT task sample uses harmonised coordinate-frame participant metrics and includes one control with only a second-session record.",
    "Inferential tests are Welch t-tests for continuous measures and chi-square tests for categorical sex."
  )
  writeLines(md, file.path(output_dir, "table1_sample_characteristics.md"))

  doc_rows <- rows %>%
    select(section, measure, patients, controls, inferential_statistic) %>%
    rename(Section = section, Measure = measure, Patients = patients, Controls = controls, `Inferential statistic` = inferential_statistic)
  ft <- flextable(doc_rows) %>%
    theme_booktabs() %>%
    merge_v(j = "Section") %>%
    valign(j = "Section", valign = "top") %>%
    bold(i = ~ !duplicated(Section), j = "Section", bold = TRUE) %>%
    autofit()
  doc <- read_docx() %>%
    body_add_par("Table 1. Demographic, clinical, and raw task-performance measures", style = "heading 1") %>%
    body_add_flextable(ft) %>%
    body_add_par("Values are mean (SD) or n (%). The OGT task sample uses harmonised coordinate-frame participant metrics and includes one control with only a second-session record.", style = "Normal")
  print(doc, target = file.path(output_dir, "table1_sample_characteristics.docx"))
  rows
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

summarise_figure_metric <- function(metrics, metric, label, panel) {
  control_vals <- finite_values(metrics %>% filter(group == "control"), metric)
  patient_vals <- finite_values(metrics %>% filter(group == "patient"), metric)
  tt <- t.test(control_vals, patient_vals, var.equal = FALSE)
  wt <- suppressWarnings(wilcox.test(control_vals, patient_vals, exact = FALSE))
  tibble(
    panel = panel,
    metric = metric,
    label = label,
    n_control = length(control_vals),
    mean_control = mean(control_vals),
    sd_control = sd(control_vals),
    n_patient = length(patient_vals),
    mean_patient = mean(patient_vals),
    sd_patient = sd(patient_vals),
    welch_t = unname(tt$statistic),
    welch_df = unname(tt$parameter),
    welch_p = tt$p.value,
    wilcoxon_p = wt$p.value
  )
}

panel_plot <- function(metrics, metric, label) {
  plot_df <- metrics %>%
    transmute(group = factor(group, levels = c("control", "patient")), value = to_num(.data[[metric]])) %>%
    filter(is.finite(value)) %>%
    mutate(
      x_center = if_else(group == "control", 1, 2),
      x_violin = x_center - 0.10,
      x_box = x_center,
      x_point = x_center + 0.14
    )
  density_df <- bind_rows(lapply(split(plot_df, plot_df$group), function(df) {
    vals <- df$value[is.finite(df$value)]
    if (length(vals) < 2 || sd(vals) == 0) return(tibble())
    den <- density(vals, n = 256, from = min(vals), to = max(vals), na.rm = TRUE)
    x_anchor <- unique(df$x_violin)[1]
    half_width <- 0.30
    left <- x_anchor - (den$y / max(den$y, na.rm = TRUE)) * half_width
    tibble(
      group = df$group[1],
      x = c(left, rep(x_anchor, length(left))),
      y = c(den$x, rev(den$x))
    )
  }))
  y_label <- if (grepl("speed", metric)) "Mean speed (pixels/sec)" else "Route count"
  ggplot() +
    geom_polygon(data = density_df, aes(x = x, y = y, fill = group, group = group), alpha = 0.28, colour = NA) +
    geom_boxplot(data = plot_df, aes(x = x_box, y = value, fill = group, group = interaction(group, x_box)), width = 0.22, outlier.shape = NA, colour = "black", linewidth = 0.7, fatten = 1.1) +
    geom_point(data = plot_df, aes(x = x_point, y = value, colour = group), position = position_jitter(width = 0.055, height = 0, seed = 42), size = 1.8, alpha = 0.45, stroke = 0) +
    scale_x_continuous(breaks = c(1, 2), labels = c("Controls", "Patients"), limits = c(0.60, 2.42)) +
    scale_fill_manual(values = c(control = control_fill, patient = patient_fill), guide = "none") +
    scale_colour_manual(values = c(control = control_point, patient = patient_point), guide = "none") +
    labs(title = label, x = NULL, y = y_label) +
    theme_classic(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11, hjust = 0.5), axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), axis.title.y = element_text(size = 10), plot.margin = margin(5, 7, 5, 7)) +
    {if (grepl("speed", metric)) scale_y_continuous(labels = label_number(big.mark = ",")) else NULL}
}

make_figure2 <- function(metrics) {
  specs <- tibble::tribble(
    ~panel, ~metric, ~label,
    "A", "practice_route_count", "Control route count",
    "B", "practice_mean_speed_px_per_s", "Control mean speed",
    "C", "test_route_count", "Test route count",
    "D", "test_mean_speed_px_per_s", "Test mean speed"
  )
  stats_df <- bind_rows(lapply(seq_len(nrow(specs)), function(i) summarise_figure_metric(metrics, specs$metric[[i]], specs$label[[i]], specs$panel[[i]])))
  write_csv(stats_df, file.path(output_dir, "figure2_raw_task_performance_stats.csv"))

  plots <- lapply(seq_len(nrow(specs)), function(i) panel_plot(metrics, specs$metric[[i]], specs$label[[i]]))
  combined <- cowplot::plot_grid(plotlist = plots, labels = specs$panel, ncol = 2, label_size = 14, label_fontface = "bold", hjust = -0.5, vjust = 1.2)
  ggsave(file.path(output_dir, "figure2_raw_task_performance.png"), combined, width = 9.3, height = 6.8, dpi = 300, bg = "white")
  ggsave(file.path(output_dir, "figure2_raw_task_performance.pdf"), combined, width = 9.3, height = 6.8, device = cairo_pdf, bg = "white")

  fmt_num <- function(x) formatC(x, digits = 2, format = "f", big.mark = ",")
  stat_lines <- stats_df %>%
    mutate(
      controls = paste0(fmt_num(mean_control), " (", fmt_num(sd_control), "), n = ", n_control),
      patients = paste0(fmt_num(mean_patient), " (", fmt_num(sd_patient), "), n = ", n_patient),
      inferential = paste0("t = ", fmt_num(welch_t), ", df = ", fmt_num(welch_df), ", ", vapply(welch_p, format_p, character(1)), "; Wilcoxon ", vapply(wilcoxon_p, format_p, character(1)))
    )
  md <- c(
    "# Figure 2. Raw Task Performance",
    "",
    "- PNG: `paper/outputs/section1/figure2_raw_task_performance.png`",
    "- PDF: `paper/outputs/section1/figure2_raw_task_performance.pdf`",
    "",
    "| Panel | Measure | Controls mean (SD) | Patients mean (SD) | Inferential statistic |",
    "|---|---|---:|---:|---|",
    paste0("| ", stat_lines$panel, " | ", stat_lines$label, " | ", stat_lines$controls, " | ", stat_lines$patients, " | ", stat_lines$inferential, " |"),
    "",
    "## Paragraph Draft",
    "",
    paste0(
      "During practice, controls generated more routes than patients (controls: M = ", fmt_num(stats_df$mean_control[1]), ", SD = ", fmt_num(stats_df$sd_control[1]), ", n = ", stats_df$n_control[1],
      "; patients: M = ", fmt_num(stats_df$mean_patient[1]), ", SD = ", fmt_num(stats_df$sd_patient[1]), ", n = ", stats_df$n_patient[1],
      "; Welch t = ", fmt_num(stats_df$welch_t[1]), ", ", format_p(stats_df$welch_p[1]), "). ",
      "Controls were also faster during practice (controls: M = ", fmt_num(stats_df$mean_control[2]), ", SD = ", fmt_num(stats_df$sd_control[2]), ", n = ", stats_df$n_control[2],
      "; patients: M = ", fmt_num(stats_df$mean_patient[2]), ", SD = ", fmt_num(stats_df$sd_patient[2]), ", n = ", stats_df$n_patient[2],
      "; Welch t = ", fmt_num(stats_df$welch_t[2]), ", ", format_p(stats_df$welch_p[2]), "). ",
      "At test, route count did not differ reliably between groups (controls: M = ", fmt_num(stats_df$mean_control[3]), ", SD = ", fmt_num(stats_df$sd_control[3]), ", n = ", stats_df$n_control[3],
      "; patients: M = ", fmt_num(stats_df$mean_patient[3]), ", SD = ", fmt_num(stats_df$sd_patient[3]), ", n = ", stats_df$n_patient[3],
      "; Welch t = ", fmt_num(stats_df$welch_t[3]), ", ", format_p(stats_df$welch_p[3]), "). ",
      "Test mean speed did not differ reliably between groups (controls: M = ", fmt_num(stats_df$mean_control[4]), ", SD = ", fmt_num(stats_df$sd_control[4]), ", n = ", stats_df$n_control[4],
      "; patients: M = ", fmt_num(stats_df$mean_patient[4]), ", SD = ", fmt_num(stats_df$sd_patient[4]), ", n = ", stats_df$n_patient[4],
      "; Welch t = ", fmt_num(stats_df$welch_t[4]), ", ", format_p(stats_df$welch_p[4]), ")."
    )
  )
  writeLines(md, file.path(output_dir, "figure2_raw_task_performance_summary.md"))
  stats_df
}

metrics <- read_metric_data()
table1 <- make_table1(metrics)
figure2 <- make_figure2(metrics)

message("Wrote section 1 outputs to ", output_dir)
message("Final task sample: ", sum(metrics$group == "control"), " controls; ", sum(metrics$group == "patient"), " patients.")
