#!/usr/bin/env Rscript
#
# Batch-timed R reference fit for the CPU-vs-R speed study.
#
# One job: time R's lm()/glm()/glm.nb() fit on a Python-prepared numeric design,
# averaged over a LOOP of `reps` fits under a single system.time() — so that
# sub-millisecond fits at small n are resolved instead of reading as 0 against
# system.time's ~1 ms granularity. Reports the mean elapsed PER FIT.
#
#   Usage: Rscript cpu_speed_run.R <x_noint_csv> <y_csv> <family> <reps> <out_json>
#
# To match what an R user actually pays (and pystatistics' fit(X, y), which builds
# its design internally), the timed region includes formula evaluation /
# model.matrix construction — the data frame is built once, outside the loop.
#
#   family in { lm, binomial, poisson, Gamma, negbin }

options(digits = 17)
suppressPackageStartupMessages({
  library(jsonlite)
  library(MASS)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) stop("usage: cpu_speed_run.R <x_csv> <y_csv> <family> <reps> <out_json>")
x_csv <- args[[1]]; y_csv <- args[[2]]; family <- args[[3]]
reps <- as.integer(args[[4]]); out_json <- args[[5]]

X <- as.matrix(read.csv(x_csv, check.names = FALSE))
y <- as.numeric(readLines(y_csv))
df <- data.frame(.y = y, X, check.names = FALSE)
predictors <- colnames(X)
fml <- as.formula(paste0("`.y` ~ ", paste(sprintf("`%s`", predictors), collapse = " + ")))

fit_once <- function() {
  if (family == "lm")            lm(fml, data = df)
  else if (family == "binomial") glm(fml, data = df, family = binomial)
  else if (family == "poisson")  glm(fml, data = df, family = poisson)
  else if (family == "Gamma")    glm(fml, data = df, family = Gamma(link = "log"))
  else if (family == "negbin")   glm.nb(fml, data = df)
  else stop(sprintf("unknown family: %s", family))
}

invisible(fit_once())                                   # warm up
elapsed <- system.time(for (i in seq_len(reps)) fit_once())[["elapsed"]]
per_fit <- elapsed / reps

out <- list(
  family = family, reps = reps, n = as.integer(nrow(df)),
  elapsed_total_s = as.numeric(elapsed),
  elapsed_per_fit_s = as.numeric(per_fit),
  r_version = as.character(getRversion())
)
write_json(out, out_json, auto_unbox = TRUE, digits = 12)
cat(sprintf("R %s n=%d reps=%d: %.6g s/fit\n", family, nrow(df), reps, per_fit))
