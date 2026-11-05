pkgs <- c("dplyr", "readr", "tidyr", "tibble", "ggplot2", "cowplot", "scales", "readxl",
          "cluster", "flextable", "officer", "jsonlite")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
