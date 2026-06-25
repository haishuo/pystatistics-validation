#!/usr/bin/env Rscript
# =============================================================================
# R `mice` worker for the head-to-head fidelity study (Study C).
#
# Reads an incomplete dataset (CSV with a header; NA/NaN mark missing), runs R
# mice with the requested settings, and writes the flattened imputed values per
# incomplete column plus the elapsed time to JSON. The Python driver
# (bench_r_fidelity.py) then compares these to pystatistics' imputations
# distributionally — same defaults, same algorithm — and contrasts wall-clock.
#
# Pinned to mice 3.19.0 (matches the committed reference fixtures). Args:
#   mice_run.R <csv_in> <json_out> <m> <maxit> <method> <seed>
# =============================================================================

suppressPackageStartupMessages({
  library(mice)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: mice_run.R <csv_in> <json_out> <m> <maxit> <method> <seed> [ordered_cols] [factor_cols]")
}
csv_in   <- args[1]
json_out <- args[2]
m        <- as.integer(args[3])
maxit    <- as.integer(args[4])
method   <- args[5]
seed     <- as.integer(args[6])
# Optional: comma-separated column NAMES to coerce to (ordered) factors so mice
# selects polr / logreg / polyreg. "none"/absent => all-numeric (use `method`).
ordered_cols <- if (length(args) >= 7 && args[7] != "none") strsplit(args[7], ",")[[1]] else character(0)
factor_cols  <- if (length(args) >= 8 && args[8] != "none") strsplit(args[8], ",")[[1]] else character(0)

incomplete <- read.csv(csv_in, na.strings = c("NA", "NaN", "nan", ""))
for (nm in factor_cols)  incomplete[[nm]] <- factor(incomplete[[nm]])
for (nm in ordered_cols) incomplete[[nm]] <- factor(incomplete[[nm]], ordered = TRUE)
incomplete_cols <- names(incomplete)[sapply(incomplete, function(v) any(is.na(v)))]

is_mixed <- length(ordered_cols) + length(factor_cols) > 0
elapsed <- system.time({
  if (is_mixed) {
    # Let mice pick per-column defaults (pmm/logreg/polyreg/polr) by column type.
    mids <- mice(incomplete, m = m, maxit = maxit, seed = seed, printFlag = FALSE)
  } else {
    mids <- mice(incomplete, m = m, maxit = maxit, method = method,
                 seed = seed, printFlag = FALSE)
  }
})["elapsed"]

per_col <- list()
for (col in incomplete_cols) {
  vals <- as.numeric(as.matrix(mids$imp[[col]]))   # (n_missing x m) -> flat
  per_col[[col]] <- vals[is.finite(vals)]
}

result <- list(
  mice_version = as.character(packageVersion("mice")),
  method       = method,
  m            = m,
  maxit        = maxit,
  seed         = seed,
  elapsed_s    = as.numeric(elapsed),
  incomplete_columns = incomplete_cols,
  per_col      = per_col
)

write_json(result, json_out, auto_unbox = TRUE, digits = 10)
cat("wrote", json_out, "(mice", as.character(packageVersion("mice")),
    "elapsed", round(as.numeric(elapsed), 3), "s)\n")
