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
#     Rscript survival_run.R glmdiag  <x_csv> <y_csv>  <out_json> [reps]
#
# - km/logrank read lung_km.csv (columns time, event, sex).
# - coxph reads lung_coxph.csv (columns time, event, age, sex, ph.ecog).
# - glmfit fits glm(y ~ 0 + ., binomial) on a headered design WITHOUT an
#   intercept (the person-period interval dummies ARE the intercept). It is the
#   reference for the discrete-time model; Python selects the covariate columns.
# - glmdiag is glmfit PLUS the reference's BEHAVIOUR diagnostics — R's own
#   convergence flag, iteration count, any glm warnings (e.g. "fitted
#   probabilities numerically 0 or 1 occurred"), and the maximum |coefficient|.
#   It is the reference for the RIGOR R15 default-degeneracy behaviour-match
#   study (does pystatistics' default match R's behaviour, including R's own
#   warnings?). Untimed (a single fit); the standard glmfit carries timing.

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

} else if (mode == "glmdiag") {
  # glmfit + behaviour diagnostics for the R15 default-degeneracy study. Captures
  # R's OWN convergence flag, iteration count, and any glm warnings (so the
  # head-to-head can claim pystatistics matches R's BEHAVIOUR, not just numbers),
  # plus the maximum |coefficient| (the separated-baseline blow-up, if any).
  x_csv <- args[[2]]; y_csv <- args[[3]]; out_json <- args[[4]]
  X <- as.matrix(read.csv(x_csv, check.names = FALSE))
  y <- as.numeric(readLines(y_csv))
  df <- data.frame(.y = y, X, check.names = FALSE)
  cols <- colnames(X)
  fml <- as.formula(paste0("`.y` ~ 0 + ", paste(sprintf("`%s`", cols), collapse = " + ")))
  warn_msgs <- character(0)
  fit <- withCallingHandlers(
    glm(fml, data = df, family = binomial()),
    warning = function(w) {
      warn_msgs <<- c(warn_msgs, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  s <- summary(fit)
  cm <- s$coefficients   # cols: Estimate, Std. Error, z value, Pr(>|z|)
  cf <- coef(fit)
  out <- list(
    procedure       = "discrete_time_default",
    coef_names      = rownames(cm),
    coefficients    = as.numeric(cm[, 1]),
    standard_errors = as.numeric(cm[, 2]),
    z_values        = as.numeric(cm[, 3]),
    p_values        = as.numeric(cm[, 4]),
    deviance        = as.numeric(fit$deviance),
    aic             = as.numeric(fit$aic),
    converged       = isTRUE(fit$converged),
    n_iter          = as.integer(fit$iter),
    n_warnings      = length(unique(warn_msgs)),
    warnings        = unique(warn_msgs),
    max_abs_coef    = as.numeric(max(abs(cf), na.rm = TRUE)),
    n_rows          = as.integer(nrow(df)),
    r_version       = r_version
  )

} else if (mode == "coxfeat") {
  # Generic Cox for the A1+VA-8 feature cluster. CSV columns: time, event,
  # then any of the RESERVED feature columns (.start, .strata, .cluster);
  # every remaining column is a covariate. Extra args after reps:
  #   ties ("efron"/"breslow"), robust ("0"/"1"), zph transform ("" = skip).
  csv <- args[[2]]; out_json <- args[[3]]
  reps <- if (length(args) >= 4) as.integer(args[[4]]) else 5L
  tie_m <- if (length(args) >= 5) args[[5]] else "efron"
  want_robust <- length(args) >= 6 && args[[6]] == "1"
  zph_tr <- if (length(args) >= 7) args[[7]] else ""

  df <- read.csv(csv, check.names = FALSE)
  feat <- intersect(c(".start", ".strata", ".cluster"), colnames(df))
  covs <- setdiff(colnames(df), c("time", "event", feat))
  rhs <- paste(sprintf("`%s`", covs), collapse = " + ")
  if (".strata" %in% feat) rhs <- paste0(rhs, " + strata(`.strata`)")
  if (".cluster" %in% feat) rhs <- paste0(rhs, " + cluster(`.cluster`)")
  lhs <- if (".start" %in% feat) "Surv(`.start`, time, event)" else "Surv(time, event)"
  fml <- as.formula(paste(lhs, "~", rhs))
  # Pass robust= only when explicitly requested: an explicit robust=FALSE would
  # SUPPRESS the robust variance that a cluster() term otherwise triggers.
  fit_once <- if (want_robust) {
    function() coxph(fml, data = df, ties = tie_m, robust = TRUE)
  } else {
    function() coxph(fml, data = df, ties = tie_m)
  }
  warn_msgs <- character(0)
  fit <- withCallingHandlers(fit_once(), warning = function(w) {
    warn_msgs <<- c(warn_msgs, conditionMessage(w)); invokeRestart("muffleWarning")
  })
  times <- timed_median(function() suppressWarnings(fit_once()), reps)
  s <- summary(fit)
  cm <- s$coefficients
  out <- list(
    procedure       = "coxfeat",
    elapsed_s       = as.numeric(median(times)),
    elapsed_times_s = as.numeric(times),
    reps            = reps,
    coef_names      = covs,
    coefficients    = as.numeric(coef(fit)),
    standard_errors = as.numeric(sqrt(diag(vcov(fit)))),
    naive_se        = if (!is.null(fit$naive.var))
                        as.numeric(sqrt(diag(fit$naive.var))) else NULL,
    z_values        = as.numeric(cm[, "z"]),
    p_values        = as.numeric(cm[, ncol(cm)]),
    concordance     = as.numeric(fit$concordance[["concordance"]]),
    loglik_null     = as.numeric(fit$loglik[1]),
    loglik_model    = as.numeric(fit$loglik[2]),
    n               = as.integer(fit$n),
    n_events        = as.integer(fit$nevent),
    n_iter          = as.integer(fit$iter),
    ties            = tie_m,
    robust          = want_robust || (".cluster" %in% feat),
    warnings        = unique(warn_msgs),
    r_version       = r_version
  )
  if (nzchar(zph_tr)) {
    z <- cox.zph(fit, transform = zph_tr)
    out$zph_rows  <- rownames(z$table)
    out$zph_chisq <- as.numeric(z$table[, "chisq"])
    out$zph_df    <- as.numeric(z$table[, "df"])
    out$zph_p     <- as.numeric(z$table[, "p"])
  }

} else if (mode == "kmfeat") {
  # Generic KM for the feature cluster. CSV columns: time, event, optional
  # .entry (left truncation), optional .strata. Extra arg: conf.type.
  csv <- args[[2]]; out_json <- args[[3]]
  reps <- if (length(args) >= 4) as.integer(args[[4]]) else 5L
  conf_type <- if (length(args) >= 5) args[[5]] else "log"
  df <- read.csv(csv, check.names = FALSE)
  lhs <- if (".entry" %in% colnames(df)) "Surv(`.entry`, time, event)" else "Surv(time, event)"
  rhs <- if (".strata" %in% colnames(df)) "`.strata`" else "1"
  fml <- as.formula(paste(lhs, "~", rhs))
  fit_once <- function() survfit(fml, data = df, conf.type = conf_type)
  fit <- fit_once()
  times <- timed_median(fit_once, reps)
  sm <- summary(fit)   # one row per event time; std.err on the survival scale
  strata_lab <- if (is.null(sm$strata)) rep("", length(sm$time))
                else sub("^[^=]*=", "", as.character(sm$strata))
  out <- list(
    procedure       = "kmfeat",
    elapsed_s       = as.numeric(median(times)),
    elapsed_times_s = as.numeric(times),
    reps            = reps,
    conf_type       = conf_type,
    strata          = strata_lab,
    time            = as.numeric(sm$time),
    survival        = as.numeric(sm$surv),
    n_risk          = as.numeric(sm$n.risk),
    n_events        = as.numeric(sm$n.event),
    se              = as.numeric(sm$std.err),
    # undefined CI bounds (S at 0/1, or an undefined transform) -> -1 sentinel
    # so the JSON stays numeric; the Python reducer masks ci < 0.
    ci_lower        = if (is.null(sm$lower)) NULL
                      else ifelse(is.na(sm$lower), -1, as.numeric(sm$lower)),
    ci_upper        = if (is.null(sm$upper)) NULL
                      else ifelse(is.na(sm$upper), -1, as.numeric(sm$upper)),
    n               = as.integer(sum(fit$n)),
    r_version       = r_version
  )

} else {
  stop(sprintf("unknown mode: %s", mode))
}

write_json(out, out_json, auto_unbox = TRUE, digits = 16, pretty = TRUE)
cat(sprintf("R %s: n=%s elapsed=%.4fs\n", mode,
            if (!is.null(out$n)) out$n else out$n_rows, out$elapsed_s))
