#!/usr/bin/env Rscript
#
# Emit the canonical survival validation datasets from R's survival::lung.
#
# One job: write the EXACT rows (after the documented complete-case NA drop)
# that both pystatistics and R will fit, so a coefficient-for-coefficient
# comparison is meaningful. The CSVs are committed under data/ and are the
# single source of truth for the lung designs.
#
#   Usage:  Rscript prep_lung.R <data_dir>
#
# survival::lung (NCCTG advanced lung cancer, Loprinzi et al. 1994):
#   time     survival time in days
#   status   1 = censored, 2 = dead
#   age      age in years
#   sex      1 = male, 2 = female
#   ph.ecog  ECOG performance score (0 best .. 5 dead), a few NAs
#
# event = (status == 2). Two designs, each complete-cased over only the
# columns it uses (so KM/log-rank keep more rows than Cox, exactly as the
# archived suite-1 reference did):
#   lung_km.csv     time, event, sex                 (KM overall + log-rank by sex)
#   lung_coxph.csv  time, event, age, sex, ph.ecog   (Cox PH + discrete-time)

suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: prep_lung.R <data_dir>")
data_dir <- args[[1]]
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

data(lung, package = "survival")

# KM / log-rank design: complete cases on (time, status, sex).
km <- lung[complete.cases(lung[, c("time", "status", "sex")]), ]
km$event <- as.numeric(km$status == 2)
write.csv(km[, c("time", "event", "sex")],
          file.path(data_dir, "lung_km.csv"), row.names = FALSE)

# Cox / discrete-time design: complete cases on (time, status, age, sex, ph.ecog).
cox <- lung[complete.cases(lung[, c("time", "status", "age", "sex", "ph.ecog")]), ]
cox$event <- as.numeric(cox$status == 2)
write.csv(cox[, c("time", "event", "age", "sex", "ph.ecog")],
          file.path(data_dir, "lung_coxph.csv"), row.names = FALSE)

cat(sprintf("wrote lung_km.csv (n=%d, events=%d)\n", nrow(km), sum(km$event)))
cat(sprintf("wrote lung_coxph.csv (n=%d, events=%d)\n", nrow(cox), sum(cox$event)))
