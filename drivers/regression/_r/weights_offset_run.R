#!/usr/bin/env Rscript
#
# R reference for the weights= / offset= correctness cases (RIGOR R10).
#
# As of pystatistics 4.3.0, fit() supports prior weights and an offset term; these
# are no longer a fail-loud gap but a correctness claim. This worker fits R's
# weighted lm() / offset glm() on a Python-prepared design and returns the
# coefficients and standard errors for a coefficient-for-coefficient comparison.
#
#   Usage: Rscript weights_offset_run.R <mode> <x_noint_csv> <y_csv> <extra_csv> <out_json>
#     mode = weighted_lm    : lm(y ~ X, weights = extra)
#     mode = offset_poisson : glm(y ~ X, family = poisson, offset = extra)

options(digits = 17)
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) stop("usage: weights_offset_run.R <mode> <x_csv> <y_csv> <extra_csv> <out_json>")
mode <- args[[1]]; x_csv <- args[[2]]; y_csv <- args[[3]]
extra_csv <- args[[4]]; out_json <- args[[5]]

X <- as.matrix(read.csv(x_csv, check.names = FALSE))
y <- as.numeric(readLines(y_csv))
extra <- as.numeric(readLines(extra_csv))
df <- data.frame(.y = y, X, check.names = FALSE)
predictors <- colnames(X)
fml <- as.formula(paste0("`.y` ~ ", paste(sprintf("`%s`", predictors), collapse = " + ")))

if (mode == "weighted_lm") {
  fit <- lm(fml, data = df, weights = extra)
} else if (mode == "offset_poisson") {
  fit <- glm(fml, data = df, family = poisson, offset = extra)
} else {
  stop(sprintf("unknown mode: %s", mode))
}

s <- summary(fit)
cm <- s$coefficients
out <- list(
  mode            = mode,
  coef_names      = rownames(cm),
  coefficients    = as.numeric(cm[, 1]),
  standard_errors = as.numeric(cm[, 2]),
  n               = as.integer(nrow(df)),
  r_version       = as.character(getRversion())
)
write_json(out, out_json, auto_unbox = TRUE, digits = 16)
cat(sprintf("R %s: n=%d coefs=%d\n", mode, nrow(df), length(out$coef_names)))
