#!/usr/bin/env Rscript
# Reference GLMM fit for the pystatistics `mixed` validation.
#
# One job: fit the EXACT model pystatistics.mixed.glmm fit -- same data (read from
# the CSV the driver dumped at full float64 precision), same formula, same family
# and link -- with lme4::glmer via the LAPLACE approximation (nAGQ = 1, matching
# glmm's approximation order), and emit JSON comparable field-for-field with the
# pystatistics record.
#
# Args: <data_csv> <formula> <family> <link> <factor_cols_csv> <out_json> <reps>
#   family: binomial | poisson
#   link:   logit | probit | log
#   factor_cols_csv: comma-separated column names to coerce with as.factor()
#                    (grouping + fixed factors). "" if none.

suppressMessages({ library(lme4); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
data_csv    <- args[[1]]
formula_str <- args[[2]]
family_str  <- args[[3]]
link_str    <- args[[4]]
factor_cols <- if (nzchar(args[[5]])) strsplit(args[[5]], ",")[[1]] else character(0)
out_json    <- args[[6]]
reps        <- as.integer(args[[7]])

df <- read.csv(data_csv, check.names = FALSE)
for (fc in factor_cols) df[[fc]] <- as.factor(df[[fc]])
form <- as.formula(formula_str)

fam <- switch(family_str,
  binomial = binomial(link = link_str),
  poisson  = poisson(link = link_str),
  stop(paste("unsupported family:", family_str)))

# Capture singular / convergence diagnostics emitted during the fit (R10).
diags <- character(0)
fit_once <- function() {
  withCallingHandlers(
    glmer(form, data = df, family = fam, nAGQ = 1),
    warning = function(w) { diags <<- c(diags, paste0("warning: ", trimws(conditionMessage(w)))); invokeRestart("muffleWarning") },
    message = function(m) { diags <<- c(diags, paste0("message: ", trimws(conditionMessage(m)))); invokeRestart("muffleMessage") })
}
m <- fit_once()
warns <- diags

# Timing: median of `reps` refits (untimed warmup already done above).
times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  suppressWarnings(suppressMessages(glmer(form, data = df, family = fam, nAGQ = 1)))
  times[i] <- proc.time()[["elapsed"]] - t0
}

co    <- summary(m)$coefficients          # Estimate, Std. Error, z value, Pr(>|z|)
fe    <- fixef(m)
vc_df <- as.data.frame(VarCorr(m))        # grp, var1, var2, vcov, sdcor
re    <- ranef(m)                         # named list of (n_groups x q) frames
blups <- lapply(re, function(d) as.matrix(d))

# Conditional deviance = sum of squared deviance residuals at the modes; this is
# what pystatistics reports (family.deviance at mu_hat), NOT lme4's -2logLik.
dev_cond <- sum(residuals(m, type = "deviance")^2)

out <- list(
  coef_names      = names(fe),
  coefficients    = as.numeric(fe),
  standard_errors = as.numeric(co[, "Std. Error"]),
  z_values        = as.numeric(co[, "z value"]),
  p_values        = as.numeric(co[, "Pr(>|z|)"]),
  vc_grp   = as.character(vc_df$grp),
  vc_var1  = ifelse(is.na(vc_df$var1), "", as.character(vc_df$var1)),
  vc_var2  = ifelse(is.na(vc_df$var2), "", as.character(vc_df$var2)),
  vc_vcov  = as.numeric(vc_df$vcov),
  vc_sdcor = as.numeric(vc_df$sdcor),
  blups          = blups,
  log_likelihood = as.numeric(logLik(m)),
  deviance       = as.numeric(dev_cond),
  aic            = as.numeric(AIC(m)),
  bic            = as.numeric(BIC(m)),
  is_singular    = isSingular(m),
  family         = family_str,
  link           = link_str,
  warnings       = warns,
  elapsed_s      = median(times),
  elapsed_times_s = times,
  r_version    = paste(R.version$major, R.version$minor, sep = "."),
  lme4_version = as.character(packageVersion("lme4"))
)

writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), out_json)
