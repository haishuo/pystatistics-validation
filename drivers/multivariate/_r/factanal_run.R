#!/usr/bin/env Rscript
#
# R reference for one maximum-likelihood factor analysis -> JSON.
#
# One job: run stats::factanal() on a numeric matrix Python prepared (identical
# float64 values), timing the fit inside R, and emit the comparable quantities
# (loadings up to rotation/sign, uniquenesses, the ML objective, and the
# chi-squared goodness-of-fit test) as JSON.
#
#   Usage: Rscript factanal_run.R <x_csv> <n_factors> <rotation> <out_json> [reps]
#
# rotation in { none, varimax, promax }. factanal works on the correlation matrix
# by default (matching pystatistics). Loadings are identified only up to rotation
# and column sign/order; uniquenesses and the objective/chi-sq are invariant and
# are the robust comparison.

options(digits = 22)
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: factanal_run.R <x_csv> <n_factors> <rotation> <out_json> [reps]")
x_csv <- args[[1]]
n_factors <- as.integer(args[[2]])
rotation <- args[[3]]
out_json <- args[[4]]
reps <- if (length(args) >= 5) as.integer(args[[5]]) else 5L

X <- as.matrix(read.csv(x_csv, check.names = FALSE))
storage.mode(X) <- "double"

# Capture any warning factanal emits (e.g. a Heywood case) so the caller can
# match R's BEHAVIOUR, not only its numbers (R10).
warn_msg <- ""
fit_once <- function() {
  withCallingHandlers(
    factanal(x = X, factors = n_factors, rotation = rotation, scores = "none"),
    warning = function(w) {
      warn_msg <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    })
}

fit <- fit_once()
times <- numeric(reps)
for (i in seq_len(reps)) times[i] <- system.time(fit_once())[["elapsed"]]

load_mat <- matrix(as.numeric(fit$loadings), nrow = nrow(fit$loadings))

out <- list(
  method = "factanal",
  elapsed_s = as.numeric(median(times)),
  elapsed_times_s = as.numeric(times),
  reps = reps,
  n = as.integer(nrow(X)),
  p = as.integer(ncol(X)),
  n_factors = n_factors,
  rotation = rotation,
  var_names = colnames(X),
  loadings = unname(load_mat),            # (p rows, m cols)
  uniquenesses = as.numeric(fit$uniquenesses),
  communalities = as.numeric(1 - fit$uniquenesses),
  objective = as.numeric(fit$criteria[["objective"]]),
  chi_sq = if (is.null(fit$STATISTIC)) NA else as.numeric(fit$STATISTIC),
  dof = as.integer(fit$dof),
  p_value = if (is.null(fit$PVAL)) NA else as.numeric(fit$PVAL),
  converged = as.integer(fit$converged),
  warning = warn_msg,
  r_version = as.character(getRversion())
)

write_json(out, out_json, auto_unbox = TRUE, digits = 16, matrix = "rowmajor",
           pretty = TRUE, na = "null")
cat(sprintf("R factanal: n=%d p=%d m=%d obj=%.6f dof=%d%s\n",
            nrow(X), ncol(X), n_factors, out$objective, out$dof,
            if (nzchar(warn_msg)) paste0(" WARN: ", warn_msg) else ""))
