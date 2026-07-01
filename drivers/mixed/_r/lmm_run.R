#!/usr/bin/env Rscript
# Reference LMM fit for the pystatistics `mixed` validation.
#
# One job: fit the EXACT model pystatistics fit -- same data (read from the CSV the
# driver dumped at full float64 precision), same formula -- with lme4 via lmerTest
# (lmerTest::lmer adds Satterthwaite df/p on top of lme4's fit; the point estimates,
# SEs, variance components and logLik are lme4's), and emit JSON comparable
# field-for-field with the pystatistics record.
#
# Args: <data_csv> <formula> <reml:0|1> <factor_cols_csv> <out_json> <reps>
#   factor_cols_csv: comma-separated column names to coerce with as.factor()
#                    (the integer-coded grouping factors). "" if none.

suppressMessages({ library(lme4); library(lmerTest); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
data_csv     <- args[[1]]
formula_str  <- args[[2]]
reml         <- as.integer(args[[3]]) == 1L
factor_cols  <- if (nzchar(args[[4]])) strsplit(args[[4]], ",")[[1]] else character(0)
out_json     <- args[[5]]
reps         <- as.integer(args[[6]])

df <- read.csv(data_csv, check.names = FALSE)
for (fc in factor_cols) df[[fc]] <- as.factor(df[[fc]])
form <- as.formula(formula_str)

# Capture singular/convergence diagnostics emitted during the fit (R10: we
# validate that pystatistics matches R's BEHAVIOUR, including what it raises).
# lme4 emits "boundary (singular) fit" as a *message*, convergence issues as
# *warnings* -- capture both, tagged, and strip trailing newlines.
diags <- character(0)
fit_once <- function() {
  withCallingHandlers(
    lmerTest::lmer(form, data = df, REML = reml),
    warning = function(w) { diags <<- c(diags, paste0("warning: ", trimws(conditionMessage(w)))); invokeRestart("muffleWarning") },
    message = function(m) { diags <<- c(diags, paste0("message: ", trimws(conditionMessage(m)))); invokeRestart("muffleMessage") })
}
m <- fit_once()
warns <- diags

# Timing: median of `reps` refits (untimed warmup already done above).
times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  suppressWarnings(lmerTest::lmer(form, data = df, REML = reml))
  times[i] <- proc.time()[["elapsed"]] - t0
}

co     <- summary(m)$coefficients           # Estimate, Std. Error, df, t value, Pr(>|t|)
fe     <- fixef(m)
vc_df  <- as.data.frame(VarCorr(m))         # grp, var1, var2, vcov, sdcor
re     <- ranef(m)                          # named list of (n_groups x q) frames
blups  <- lapply(re, function(d) as.matrix(d))

out <- list(
  coef_names      = names(fe),
  coefficients    = as.numeric(fe),
  standard_errors = as.numeric(co[, "Std. Error"]),
  df_satterthwaite= as.numeric(co[, "df"]),
  t_values        = as.numeric(co[, "t value"]),
  p_values        = as.numeric(co[, "Pr(>|t|)"]),
  # Variance components as parallel arrays (NA-safe: var2/sdcor may be NA).
  vc_grp   = as.character(vc_df$grp),
  vc_var1  = ifelse(is.na(vc_df$var1), "", as.character(vc_df$var1)),
  vc_var2  = ifelse(is.na(vc_df$var2), "", as.character(vc_df$var2)),
  vc_vcov  = as.numeric(vc_df$vcov),
  vc_sdcor = as.numeric(vc_df$sdcor),
  blups        = blups,
  log_likelihood = as.numeric(logLik(m)),
  reml_criterion = as.numeric(-2 * logLik(m)),
  aic          = as.numeric(AIC(m)),
  bic          = as.numeric(BIC(m)),
  is_singular  = isSingular(m),
  reml         = reml,
  warnings     = warns,
  elapsed_s    = median(times),
  elapsed_times_s = times,
  r_version    = paste(R.version$major, R.version$minor, sep = "."),
  lme4_version = as.character(packageVersion("lme4")),
  lmerTest_version = as.character(packageVersion("lmerTest"))
)

writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), out_json)
