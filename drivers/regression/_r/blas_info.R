#!/usr/bin/env Rscript
#
# Report the BLAS / LAPACK R is linked against on this host → JSON.
#
# One job: record the numerical backend behind R's lm()/glm() so the CPU-vs-R
# speed gap is interpretable (RIGOR R11). A reference-BLAS host and an
# OpenBLAS/MKL host give very different R baselines; an unstated BLAS makes the
# gap meaningless.
#
#   Usage:  Rscript blas_info.R <host> <out_json>

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: blas_info.R <host> <out_json>")
host <- args[[1]]; out_json <- args[[2]]

ext <- tryCatch(extSoftVersion(), error = function(e) c(BLAS = NA))
si  <- sessionInfo()

out <- list(
  host        = host,
  r_version   = as.character(getRversion()),
  blas        = tryCatch(as.character(ext[["BLAS"]]), error = function(e) NA_character_),
  lapack      = tryCatch(as.character(La_library()), error = function(e) NA_character_),
  lapack_ver  = tryCatch(as.character(La_version()), error = function(e) NA_character_),
  blas_si     = if (!is.null(si$BLAS)) si$BLAS else NA_character_,
  lapack_si   = if (!is.null(si$LAPACK)) si$LAPACK else NA_character_,
  platform    = as.character(si$platform)
)
write_json(out, out_json, auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("BLAS on %s: %s\n", host, out$blas))
