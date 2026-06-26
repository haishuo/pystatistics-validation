#!/usr/bin/env Rscript
#
# One job: emit the two R-native regression datasets as CSV so Python fits the
# exact same rows / model matrix R does. Deterministic; run once, CSVs committed.
#
#   Usage:  Rscript prep_datasets.R <out_dir>
#
#   airquality.csv  — Ozone ~ Solar.R + Temp + Wind, complete cases (Gamma target)
#   quine_mm.csv    — Days + model.matrix(Days ~ Eth + Sex + Age + Lrn)[, -1]
#                     (negative-binomial; R's exact factor coding)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: prep_datasets.R <out_dir>")
out_dir <- args[[1]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(MASS))

# airquality: complete cases over the modelled columns only.
aq <- airquality[complete.cases(airquality[, c("Ozone", "Solar.R", "Temp", "Wind")]), ]
write.csv(aq[, c("Ozone", "Solar.R", "Temp", "Wind")],
          file.path(out_dir, "airquality.csv"), row.names = FALSE)
cat(sprintf("airquality.csv: %d complete cases\n", nrow(aq)))

# quine: hand Python the numeric model matrix R built (drop the intercept col).
X <- model.matrix(Days ~ Eth + Sex + Age + Lrn, data = quine)
quine_mm <- data.frame(Days = quine$Days, X[, -1, drop = FALSE], check.names = TRUE)
colnames(quine_mm)[-1] <- colnames(X)[-1]
write.csv(quine_mm, file.path(out_dir, "quine_mm.csv"), row.names = FALSE)
cat(sprintf("quine_mm.csv: %d rows, %d model-matrix columns\n",
            nrow(quine_mm), ncol(X) - 1))
