#!/usr/bin/env Rscript
#
# R reference for one survival procedure -> JSON.
#
# One job: fit the R survival reference for a single procedure on the SAME rows
# pystatistics fits (the committed lung CSVs, or — for discrete-time — a
# person-period design Python reconstructs and hands over), time the fit inside
# R (interpreter startup excluded), and emit the comparable quantities as JSON.
#
#   Usage:
#     Rscript survival_run.R km       <lung_km_csv>    <out_json> [reps]
#     Rscript survival_run.R logrank  <lung_km_csv>    <out_json> [reps]
#     Rscript survival_run.R coxph    <lung_coxph_csv> <out_json> [reps]
#     Rscript survival_run.R glmfit   <x_csv> <y_csv>  <out_json> [reps]
#
# - km/logrank read lung_km.csv (columns time, event, sex).
# - coxph reads lung_coxph.csv (columns time, event, age, sex, ph.ecog).
# - glmfit fits glm(y ~ 0 + ., binomial) on a headered design WITHOUT an
#   intercept (the person-period interval dummies ARE the intercept). It is the
#   reference for the discrete-time model; Python selects the covariate columns.

options(digits = 22)
suppressPackageStartupMessages({
  library(jsonlite)
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: survival_run.R <mode> <csv...> <out_json> [reps]")
mode <- args[[1]]

# Time `reps` fits and return a per-fit elapsed vector. `thunk` MUST be a
# zero-arg function (NOT a bare expression): a lazy promise would be forced only
# once and every later rep would measure the cached value (0s).
#
# We block-time the whole batch and divide by reps rather than timing each fit
# individually: a single fast fit (sub-millisecond at small n) rounds to 0 under
# system.time's resolution, which would zero out the median. Block timing
# accumulates enough wall-clock to measure reliably; the returned vector repeats
# the per-fit mean so the downstream min/median/max all reflect it.
timed_median <- function(thunk, reps) {
  total <- system.time(for (i in seq_len(reps)) thunk())[["elapsed"]]
  rep(total / reps, reps)
}

r_version <- as.character(getRversion())

if (mode == "km") {
  csv <- args[[2]]; out_json <- args[[3]]
  reps <- if (length(args) >= 4) as.integer(args[[4]]) else 5L
  df <- read.csv(csv)
  fit <- survfit(Surv(time, event) ~ 1, data = df)
  times <- timed_median(function() survfit(Surv(time, event) ~ 1, data = df), reps)
  s <- summary(fit)
  med <- as.numeric(quantile(fit, probs = 0.5)$quantile)
  out <- list(
    procedure       = "kaplan_meier",
    elapsed_s       = as.numeric(median(times)),
    elapsed_times_s = as.numeric(times),
    reps            = reps,
    time            = as.numeric(s$time),
    survival        = as.numeric(s$surv),
    n_risk          = as.numeric(s$n.risk),
    n_events        = as.numeric(s$n.event),
    std_err         = as.numeric(s$std.err),
    ci_lower        = as.numeric(s$lower),
    ci_upper        = as.numeric(s$upper),
    median_survival = med,
    n               = as.integer(nrow(df)),
    n_events_total  = as.integer(sum(df$event)),
    r_version       = r_version
  )

} else if (mode == "logrank") {
  csv <- args[[2]]; out_json <- args[[3]]
  reps <- if (length(args) >= 4) as.integer(args[[4]]) else 5L
  df <- read.csv(csv)
  fit <- survdiff(Surv(time, event) ~ sex, data = df)
  times <- timed_median(function() survdiff(Surv(time, event) ~ sex, data = df), reps)
  df_dof <- length(fit$n) - 1L
  out <- list(
    procedure       = "survdiff",
    elapsed_s       = as.numeric(median(times)),
    elapsed_times_s = as.numeric(times),
    reps            = reps,
    statistic       = as.numeric(fit$chisq),
    df              = as.integer(df_dof),
    p_value         = as.numeric(pchisq(fit$chisq, df = df_dof, lower.tail = FALSE)),
    observed        = as.numeric(fit$obs),
    expected        = as.numeric(fit$exp),
    group_labels    = names(fit$n),
    n               = as.integer(nrow(df)),
    r_version       = r_version
  )

} else if (mode == "coxph") {
  csv <- args[[2]]; out_json <- args[[3]]
  reps <- if (length(args) >= 4) as.integer(args[[4]]) else 5L
  df <- read.csv(csv, check.names = FALSE)
  fit <- coxph(Surv(time, event) ~ age + sex + `ph.ecog`, data = df, ties = "efron")
  times <- timed_median(
    function() coxph(Surv(time, event) ~ age + sex + `ph.ecog`, data = df, ties = "efron"),
    reps)
  s <- summary(fit)
  cm <- s$coefficients   # cols: coef, exp(coef), se(coef), z, Pr(>|z|)
  out <- list(
    procedure       = "coxph",
    elapsed_s       = as.numeric(median(times)),
    elapsed_times_s = as.numeric(times),
    reps            = reps,
    coef_names      = rownames(cm),
    coefficients    = as.numeric(cm[, "coef"]),
    hazard_ratios   = as.numeric(cm[, "exp(coef)"]),
    standard_errors = as.numeric(cm[, "se(coef)"]),
    z_values        = as.numeric(cm[, "z"]),
    p_values        = as.numeric(cm[, "Pr(>|z|)"]),
    concordance     = as.numeric(s$concordance[["C"]]),
    loglik_null     = as.numeric(fit$loglik[1]),
    loglik_model    = as.numeric(fit$loglik[2]),
    n               = as.integer(fit$n),
    n_events        = as.integer(fit$nevent),
    ties            = "efron",
    r_version       = r_version
  )

} else if (mode == "coxfit") {
  # Generic Cox fit on a synthetic CSV (columns: time, event, then covariates).
  # Used by the scaling study; covariate names are whatever the CSV carries.
  csv <- args[[2]]; out_json <- args[[3]]
  reps <- if (length(args) >= 4) as.integer(args[[4]]) else 5L
  df <- read.csv(csv, check.names = FALSE)
  covs <- setdiff(colnames(df), c("time", "event"))
  fml <- as.formula(paste0("Surv(time, event) ~ ",
                           paste(sprintf("`%s`", covs), collapse = " + ")))
  fit <- coxph(fml, data = df, ties = "efron")
  times <- timed_median(function() coxph(fml, data = df, ties = "efron"), reps)
  cm <- summary(fit)$coefficients
  out <- list(
    procedure       = "coxph",
    elapsed_s       = as.numeric(median(times)),
    elapsed_times_s = as.numeric(times),
    reps            = reps,
    coefficients    = as.numeric(cm[, "coef"]),
    n               = as.integer(fit$n),
    n_events        = as.integer(fit$nevent),
    r_version       = r_version
  )

} else if (mode == "glmfit") {
  x_csv <- args[[2]]; y_csv <- args[[3]]; out_json <- args[[4]]
  reps <- if (length(args) >= 5) as.integer(args[[5]]) else 5L
  X <- as.matrix(read.csv(x_csv, check.names = FALSE))
  y <- as.numeric(readLines(y_csv))
  df <- data.frame(.y = y, X, check.names = FALSE)
  cols <- colnames(X)
  fml <- as.formula(paste0("`.y` ~ 0 + ", paste(sprintf("`%s`", cols), collapse = " + ")))
  fit <- glm(fml, data = df, family = binomial())
  times <- timed_median(function() glm(fml, data = df, family = binomial()), reps)
  s <- summary(fit)
  cm <- s$coefficients   # cols: Estimate, Std. Error, z value, Pr(>|z|)
  out <- list(
    procedure       = "discrete_time",
    elapsed_s       = as.numeric(median(times)),
    elapsed_times_s = as.numeric(times),
    reps            = reps,
    coef_names      = rownames(cm),
    coefficients    = as.numeric(cm[, 1]),
    standard_errors = as.numeric(cm[, 2]),
    z_values        = as.numeric(cm[, 3]),
    p_values        = as.numeric(cm[, 4]),
    deviance        = as.numeric(fit$deviance),
    aic             = as.numeric(fit$aic),
    n_rows          = as.integer(nrow(df)),
    r_version       = r_version
  )

} else {
  stop(sprintf("unknown mode: %s", mode))
}

write_json(out, out_json, auto_unbox = TRUE, digits = 16, pretty = TRUE)
cat(sprintf("R %s: n=%s elapsed=%.4fs\n", mode,
            if (!is.null(out$n)) out$n else out$n_rows, out$elapsed_s))
