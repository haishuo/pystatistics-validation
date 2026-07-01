#!/usr/bin/env Rscript
# Reference GRM / GBLUP fit for the pystatistics grm_lmm validation.
#
# One job: fit the EXACT model grm_lmm fits — y = Xβ + g + e with Cov(g) = σ_g²·K,
# K = W Wᵀ / M — using rrBLUP::mixed.solve (the canonical GBLUP/GRM reference), on
# the identical numbers (K and y/X handed in via CSVs the driver dumped), and emit
# JSON comparable field-for-field with the pystatistics record.
#
# mixed.solve(y, Z=I (default), K, X, method) fits u ~ N(0, K·Vu), e ~ N(0, Ve);
# with Z=I the u are per-individual genetic values — matching grm_lmm's K=WWᵀ/M.
#
# Args: <y_csv> <X_csv> <K_csv> <reml:0|1> <out_json>

suppressMessages({ library(rrBLUP); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
y <- as.numeric(read.csv(args[[1]])[[1]])
X <- as.matrix(read.csv(args[[2]], check.names = FALSE))
K <- as.matrix(read.csv(args[[3]], check.names = FALSE))
reml <- as.integer(args[[4]]) == 1L
out_json <- args[[5]]
method <- if (reml) "REML" else "ML"

t0 <- proc.time()[["elapsed"]]
fit <- mixed.solve(y = y, K = K, X = X, method = method, SE = TRUE)
elapsed <- proc.time()[["elapsed"]] - t0

Vu <- as.numeric(fit$Vu); Ve <- as.numeric(fit$Ve)
out <- list(
  coefficients    = as.numeric(fit$beta),
  standard_errors = as.numeric(fit$beta.SE),
  var_genetic     = Vu,
  var_residual    = Ve,
  heritability    = Vu / (Vu + Ve),
  variance_ratio  = Vu / Ve,
  genetic_values  = as.numeric(fit$u),
  log_likelihood  = as.numeric(fit$LL),
  method          = method,
  elapsed_s       = elapsed,
  r_version       = paste(R.version$major, R.version$minor, sep = "."),
  rrBLUP_version  = as.character(packageVersion("rrBLUP"))
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), out_json)
