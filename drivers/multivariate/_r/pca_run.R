#!/usr/bin/env Rscript
#
# R reference for one PCA fit -> JSON.
#
# One job: run stats::prcomp() on a numeric matrix that Python prepared (the
# EXACT float64 values, dumped at full precision so R and pystatistics analyse
# identical numbers), timing the decomposition inside R, and emit the comparable
# quantities (sdev, rotation/loadings, scores, center, scale) as JSON.
#
#   Usage: Rscript pca_run.R <x_csv> <center 0|1> <scale 0|1> <out_json> [reps]
#
# <x_csv> is the data matrix WITH a header row of variable names, no row names,
# numeric. prcomp adds no intercept (PCA centers internally). Eigenvector signs
# are arbitrary (LAPACK-dependent); the caller sign-aligns before comparing.

options(digits = 22)
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: pca_run.R <x_csv> <center> <scale> <out_json> [reps]")
x_csv <- args[[1]]
do_center <- as.integer(args[[2]]) == 1L
do_scale <- as.integer(args[[3]]) == 1L
out_json <- args[[4]]
reps <- if (length(args) >= 5) as.integer(args[[5]]) else 5L

X <- as.matrix(read.csv(x_csv, check.names = FALSE))
storage.mode(X) <- "double"

fit_once <- function() prcomp(X, center = do_center, scale. = do_scale)

fit <- fit_once()
times <- numeric(reps)
for (i in seq_len(reps)) times[i] <- system.time(fit_once())[["elapsed"]]

rot <- fit$rotation                       # (p x k) loadings
scores <- fit$x                           # (n x k)
sdev <- as.numeric(fit$sdev)
# center / scale: prcomp returns FALSE when not applied; normalize to vectors.
ctr <- if (isFALSE(fit$center)) rep(0, ncol(X)) else as.numeric(fit$center)
scl <- if (isFALSE(fit$scale)) rep(1, ncol(X)) else as.numeric(fit$scale)

out <- list(
  method = "prcomp",
  elapsed_s = as.numeric(median(times)),
  elapsed_times_s = as.numeric(times),
  reps = reps,
  n = as.integer(nrow(X)),
  p = as.integer(ncol(X)),
  var_names = colnames(X),
  sdev = sdev,
  explained_variance = sdev^2,
  explained_variance_ratio = sdev^2 / sum(sdev^2),
  rotation = unname(as.matrix(rot)),      # JSON: list of rows (p rows, k cols)
  scores = unname(as.matrix(scores)),     # (n rows, k cols)
  center = ctr,
  scale = scl,
  r_version = as.character(getRversion())
)

write_json(out, out_json, auto_unbox = TRUE, digits = 16, matrix = "rowmajor",
           pretty = TRUE)
cat(sprintf("R prcomp: n=%d p=%d center=%d scale=%d elapsed=%.5fs\n",
            nrow(X), ncol(X), do_center, do_scale, out$elapsed_s))
