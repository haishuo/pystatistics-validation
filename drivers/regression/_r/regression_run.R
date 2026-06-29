#!/usr/bin/env Rscript
#
# Generic R reference for one regression fit → JSON.
#
# One job: fit lm() / glm() / glm.nb() on a numeric design that Python prepared,
# timing the fit inside R (so R interpreter startup is excluded), and emit the
# comparable quantities (coefficients, SEs, test statistics, p-values, and the
# model-level fit summary) as JSON.
#
#   Usage:  Rscript regression_run.R <x_noint_csv> <y_csv> <family> <out_json> [reps]
#
# Convention: <x_noint_csv> is the design WITHOUT the intercept column (headered,
# numeric — factors already coded by the caller). R adds its own intercept, so the
# returned coefficient vector is [(Intercept), betas...], matching pystatistics'
# intercept-first ordering and giving the correctly centered R^2 for lm.
#
#   family ∈ { lm, binomial, poisson, Gamma, negbin }

options(digits = 22)
suppressPackageStartupMessages({
  library(jsonlite)
  library(MASS)   # glm.nb
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: regression_run.R <x_noint_csv> <y_csv> <family> <out_json>")
x_csv <- args[[1]]; y_csv <- args[[2]]; family <- args[[3]]; out_json <- args[[4]]
reps <- if (length(args) >= 5) as.integer(args[[5]]) else 5L

X <- as.matrix(read.csv(x_csv, check.names = FALSE))
y <- as.numeric(readLines(y_csv))
df <- data.frame(.y = y, X, check.names = FALSE)
predictors <- colnames(X)
fml <- as.formula(paste0("`.y` ~ ", paste(sprintf("`%s`", predictors), collapse = " + ")))

# Fit once for the estimates, then time `reps` more fits and report the median
# elapsed — a single system.time() is too noisy to compare against pystatistics.
fit_once <- function() {
  if (family == "lm")            lm(fml, data = df)
  else if (family == "binomial") glm(fml, data = df, family = binomial)
  else if (family == "poisson")  glm(fml, data = df, family = poisson)
  else if (family == "Gamma")    glm(fml, data = df, family = Gamma(link = "log"))
  else if (family == "negbin")   glm.nb(fml, data = df)
  else stop(sprintf("unknown family: %s", family))
}

fit <- fit_once()
times <- numeric(reps)
for (i in seq_len(reps)) times[i] <- system.time(fit_once())[["elapsed"]]
elapsed_median <- as.numeric(median(times))
r <- list(elapsed = elapsed_median)
s <- summary(fit)
cm <- s$coefficients   # cols: Estimate, Std. Error, (t|z) value, Pr(>|.|)

out <- list(
  family          = family,
  elapsed_s       = r$elapsed,
  elapsed_times_s = as.numeric(times),
  reps            = reps,
  coef_names      = rownames(cm),
  coefficients    = as.numeric(cm[, 1]),
  standard_errors = as.numeric(cm[, 2]),
  test_statistic  = as.numeric(cm[, 3]),
  p_values        = as.numeric(cm[, 4]),
  stat_name       = colnames(cm)[3],
  n               = as.integer(nrow(df)),
  df_residual     = as.integer(if (is.null(fit$df.residual)) fit$df.residual else fit$df.residual),
  r_version       = as.character(getRversion())
)

if (family == "lm") {
  out$r_squared          <- as.numeric(s$r.squared)
  out$adjusted_r_squared <- as.numeric(s$adj.r.squared)
  out$residual_std_error <- as.numeric(s$sigma)
  out$fitted_first10     <- as.numeric(fitted(fit)[1:10])
  out$residuals_first10  <- as.numeric(residuals(fit)[1:10])
} else {
  out$deviance      <- as.numeric(fit$deviance)
  out$null_deviance <- as.numeric(fit$null.deviance)
  out$aic           <- as.numeric(fit$aic)
  out$bic           <- as.numeric(BIC(fit))
  out$dispersion    <- as.numeric(s$dispersion)
  if (family == "negbin") out$theta <- as.numeric(fit$theta)
}

write_json(out, out_json, auto_unbox = TRUE, digits = 16, pretty = TRUE)
cat(sprintf("R %s: n=%d  elapsed=%.4fs\n", family, nrow(df), r$elapsed))
