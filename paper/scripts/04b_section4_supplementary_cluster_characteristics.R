#!/usr/bin/env Rscript

if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a  # compatibility with older R/ggplot2
suppressPackageStartupMessages({
  library(dplyr)
  library(flextable)
  library(officer)
  library(readr)
  library(tibble)
})

output_dir <- file.path("paper", "outputs", "section4")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

memberships_path <- file.path(output_dir, "section4_kmeans_memberships.csv")
clinical_path <- "CHANSS_clinical_demog.csv"

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

format_p <- function(p) {
  if (!is.finite(p)) return("")
  if (p < 0.001) return("p < .001")
  sub("^0", "", sprintf("p = %.3f", p))
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

counts_with_n <- function(parts, denom) {
  value <- paste(parts, collapse = " / ")
  observed <- sum(parts)
  if (observed < denom) value <- paste0(value, ", n = ", observed)
  value
}

format_kruskal <- function(rows, field) {
  vals <- rows %>%
    transmute(cluster = kmeans_cluster, value = to_num(.data[[field]])) %>%
    filter(is.finite(value), !is.na(cluster))
  groups <- split(vals$value, vals$cluster)
  groups <- groups[vapply(groups, length, integer(1)) > 0]
  if (length(groups) < 2) return("")
  kt <- kruskal.test(groups)
  paste0("H = ", sprintf("%.2f", unname(kt$statistic)), ", ", format_p(kt$p.value))
}

format_fisher <- function(tab) {
  if (!length(tab) || any(rowSums(tab) == 0) || any(colSums(tab) == 0)) return("")
  ft <- fisher.test(tab)
  paste0("Fisher exact, ", format_p(ft$p.value))
}

patient_site <- function(platform, subject_id) {
  case_when(
    platform == "OGT_Mandarin" | grepl("^CH", subject_id) ~ "China",
    platform == "OGT_Spanish" ~ "Spain",
    platform %in% c("OGT_English", "OCTAL") | grepl("^C", subject_id) ~ "UK",
    TRUE ~ "Unknown"
  )
}

load_clinical_extra <- function(path) {
  clinical <- read_csv(path, show_col_types = FALSE, name_repair = "unique_quiet")
  tol_cols <- grep("^BACS_ToL_t", names(clinical), value = TRUE)
  tol_values <- if (length(tol_cols)) clinical[[tol_cols[[1]]]] else NA_real_

  clinical %>%
    mutate(
      ID_subject = toupper(trimws(ID_subject)),
      BACS_ToL_t = to_num(tol_values),
      MAP_check = sum_observed_items(across(c(BNSS_1, BNSS_2, BNSS_3, BNSS_5, BNSS_6, BNSS_7, BNSS_8), to_num)),
      EXP_check = sum_observed_items(across(c(BNSS_9, BNSS_10, BNSS_11, BNSS_12, BNSS_13), to_num)),
      CDS_total_check = sum_observed_items(across(paste0("CDS_", 1:9), to_num)),
      PANSS_marder_positive_check = sum_observed_items(across(c(PANSS_p1, PANSS_p3, PANSS_pg9, PANSS_pg10), to_num))
    ) %>%
    rowwise() %>%
    mutate(
      AMI_behavioural_check = mean(c_across(c(AMI_5, AMI_9, AMI_10, AMI_11, AMI_12, AMI_15)), na.rm = TRUE),
      AMI_social_check = mean(c_across(c(AMI_2, AMI_3, AMI_4, AMI_8, AMI_14, AMI_17)), na.rm = TRUE),
      AMI_emotional_check = mean(c_across(c(AMI_1, AMI_6, AMI_7, AMI_13, AMI_16, AMI_18)), na.rm = TRUE)
    ) %>%
    ungroup() %>%
    transmute(
      ID_subject,
      Gender,
      Age = to_num(Age),
      pEducation = to_num(pEducation),
      Age_FEP = to_num(Age_FEP),
      AP_1,
      AP_2,
      BACS_comp_t = to_num(BACS_comp_t),
      BACS_Comp_z = to_num(BACS_Comp_z),
      BACS_DS_z = to_num(BACS_DS_z),
      BACS_TMT_z = to_num(BACS_TMT_z),
      BACS_VF_z = to_num(BACS_VF_z),
      BACS_ToL_t = to_num(BACS_ToL_t),
      MAP_check = to_num(MAP_check),
      EXP_check = to_num(EXP_check),
      CDS_total_check = to_num(CDS_total_check),
      PANSS_marder_positive_check = to_num(PANSS_marder_positive_check),
      AMI_behavioural_check = if_else(is.nan(AMI_behavioural_check), NA_real_, as.numeric(AMI_behavioural_check)),
      AMI_social_check = if_else(is.nan(AMI_social_check), NA_real_, as.numeric(AMI_social_check)),
      AMI_emotional_check = if_else(is.nan(AMI_emotional_check), NA_real_, as.numeric(AMI_emotional_check))
    ) %>%
    distinct(ID_subject, .keep_all = TRUE)
}

read_cluster_rows <- function() {
  memberships <- read_csv(memberships_path, show_col_types = FALSE, guess_max = 10000) %>%
    mutate(
      kmeans_cluster = factor(kmeans_cluster, levels = 1:3),
      clinical_lookup_id = toupper(trimws(clinical_lookup_id))
    )
  clinical_extra <- load_clinical_extra(clinical_path)

  memberships %>%
    left_join(clinical_extra, by = c("clinical_lookup_id" = "ID_subject")) %>%
    mutate(
      illness_duration = Age - Age_FEP,
      is_fep_5y = if_else(is.finite(illness_duration) & illness_duration >= 0 & illness_duration <= 5, 1, 0, missing = NA_real_),
      is_trs = case_when(
        is.na(AP_1) | trimws(as.character(AP_1)) == "" ~ NA_real_,
        grepl("cloz", AP_1, ignore.case = TRUE) | grepl("cloz", AP_2, ignore.case = TRUE) ~ 1,
        TRUE ~ 0
      ),
      sex_label = if_else(as.character(Gender) == "1", "Male", "Female", missing = NA_character_),
      site = patient_site(source_platform, clinical_lookup_id)
    )
}

make_table <- function(rows) {
  sizes <- as.integer(table(rows$kmeans_cluster))
  cluster_rows <- lapply(levels(rows$kmeans_cluster), function(k) rows %>% filter(kmeans_cluster == k))
  names(cluster_rows) <- levels(rows$kmeans_cluster)

  out <- tibble(section = character(), measure = character(), cluster_1 = character(), cluster_2 = character(), cluster_3 = character(), omnibus_statistic = character())

  add <- function(section, measure, values, omnibus = "") {
    out <<- bind_rows(
      out,
      tibble(
        section = section,
        measure = measure,
        cluster_1 = values[[1]],
        cluster_2 = values[[2]],
        cluster_3 = values[[3]],
        omnibus_statistic = omnibus
      )
    )
  }

  continuous <- function(section, measure, field) {
    vals <- lapply(cluster_rows, mean_sd, field = field)
    add(section, measure, vals, format_kruskal(rows, field))
  }

  add("Sample", "K-means cluster sample", as.list(as.character(sizes)))
  add(
    "Sample",
    "Matched patient clinical data",
    lapply(cluster_rows, function(x) as.character(sum(!is.na(x$Age) | !is.na(x$Gender))))
  )

  continuous("Demographic", "Age, years", "Age")

  sex_vals <- lapply(cluster_rows, function(x) {
    sex_tab <- table(factor(x$sex_label, levels = c("Female", "Male")), useNA = "no")
    counts_with_n(as.integer(sex_tab), nrow(x))
  })
  sex_tab <- table(rows$kmeans_cluster, factor(rows$sex_label, levels = c("Female", "Male")), useNA = "no")
  add("Demographic", "Sex, female / male", sex_vals, format_fisher(sex_tab))

  site_vals <- lapply(cluster_rows, function(x) {
    site_tab <- table(factor(x$site, levels = c("UK", "Spain", "China")), useNA = "no")
    counts_with_n(as.integer(site_tab), nrow(x))
  })
  site_tab <- table(rows$kmeans_cluster, factor(rows$site, levels = c("UK", "Spain", "China")), useNA = "no")
  add("Demographic", "Patient site, UK / Spain / China", site_vals, format_fisher(site_tab))

  continuous("Demographic", "Education, years", "pEducation")
  continuous("Illness", "Age at first episode, years", "Age_FEP")
  continuous("Illness", "Illness duration, years", "illness_duration")
  continuous("Medication", "Antipsychotic burden (% maximum dose)", "AP_BNFmax")
  continuous("Medication", "Antipsychotic load 2026", "AP_Load2026")

  fep_vals <- lapply(cluster_rows, function(x) {
    known <- x %>% filter(is.finite(is_fep_5y))
    paste0(n_percent(sum(known$is_fep_5y == 1), nrow(known)), ", n = ", nrow(known))
  })
  fep_tab <- table(rows$kmeans_cluster, factor(rows$is_fep_5y, levels = c(0, 1)), useNA = "no")
  add("Illness", "FEP within 5 years", fep_vals, format_fisher(fep_tab))

  trs_vals <- lapply(cluster_rows, function(x) {
    known <- x %>% filter(is.finite(is_trs))
    paste0(n_percent(sum(known$is_trs == 1), nrow(known)), ", n = ", nrow(known))
  })
  trs_tab <- table(rows$kmeans_cluster, factor(rows$is_trs, levels = c(0, 1)), useNA = "no")
  add("Illness", "TRS / current clozapine", trs_vals, format_fisher(trs_tab))

  continuous("Cognitive", "BACS verbal fluency z-score", "BACS_VF_z")
  continuous("Cognitive", "BACS composite z-score", "BACS_Comp_z")
  continuous("Cognitive", "BACS digit sequencing z-score", "BACS_DS_z")
  continuous("Cognitive", "BACS trail-making z-score", "BACS_TMT_z")

  continuous("Clinical", "PANSS Marder positive", "PANSS_marder_positive")
  continuous("Clinical", "BNSS MAP", "MAP")
  continuous("Clinical", "BNSS EXP", "EXP")
  continuous("Clinical", "AMI behavioural", "AMI_behavioural")
  continuous("Clinical", "AMI social", "AMI_social")
  continuous("Clinical", "AMI emotional", "AMI_emotional")
  continuous("Clinical", "CDSS total", "CDS_total")
  continuous("Clinical", "PSP score", "PSP_score")

  attr(out, "sizes") <- sizes
  out
}

write_markdown <- function(rows, path) {
  sizes <- attr(rows, "sizes")
  lines <- c(
    "# Supplementary Table. Patient demographic, cognitive, and clinical characteristics across Figure 5 k-means clusters",
    "",
    paste0("| Measure | Cluster 1 (n = ", sizes[[1]], ") | Cluster 2 (n = ", sizes[[2]], ") | Cluster 3 (n = ", sizes[[3]], ") | Omnibus statistic |"),
    "|---|---:|---:|---:|---:|"
  )
  current <- ""
  for (i in seq_len(nrow(rows))) {
    if (rows$section[[i]] != current) {
      current <- rows$section[[i]]
      lines <- c(lines, paste0("| **", current, "** |  |  |  |  |"))
    }
    lines <- c(lines, paste0("| ", rows$measure[[i]], " | ", rows$cluster_1[[i]], " | ", rows$cluster_2[[i]], " | ", rows$cluster_3[[i]], " | ", rows$omnibus_statistic[[i]], " |"))
  }
  lines <- c(
    lines,
    "",
    "Values are mean (SD) or n (%). Cells include measure-specific n where data are missing.",
    "Omnibus tests are Kruskal-Wallis tests for continuous measures and Fisher exact tests for categorical measures.",
    "Cluster assignments are taken directly from `paper/outputs/section4/section4_kmeans_memberships.csv`, the membership file used for Figure 5."
  )
  writeLines(lines, path)
}

write_docx <- function(rows, path) {
  doc_rows <- rows %>%
    rename(
      Section = section,
      Measure = measure,
      `Cluster 1` = cluster_1,
      `Cluster 2` = cluster_2,
      `Cluster 3` = cluster_3,
      `Omnibus statistic` = omnibus_statistic
    )

  ft <- flextable(doc_rows) %>%
    theme_booktabs() %>%
    merge_v(j = "Section") %>%
    valign(j = "Section", valign = "top") %>%
    bold(i = ~ !duplicated(Section), j = "Section", bold = TRUE) %>%
    autofit()

  read_docx() %>%
    body_add_par("Supplementary Table. Patient demographic, cognitive, and clinical characteristics across Figure 5 k-means clusters", style = "heading 1") %>%
    body_add_flextable(ft) %>%
    body_add_par("Values are mean (SD) or n (%). Cells include measure-specific n where data are missing. Omnibus tests are Kruskal-Wallis tests for continuous measures and Fisher exact tests for categorical measures.", style = "Normal") %>%
    print(target = path)
}

rows <- read_cluster_rows()
table_rows <- make_table(rows)

write_csv(rows, file.path(output_dir, "section4_cluster_characteristics_rows.csv"))
write_csv(table_rows, file.path(output_dir, "supplementary_cluster_characteristics.csv"))
write_markdown(table_rows, file.path(output_dir, "supplementary_cluster_characteristics.md"))
write_docx(table_rows, file.path(output_dir, "supplementary_cluster_characteristics.docx"))

message("Wrote supplementary cluster characteristics table to ", output_dir)
