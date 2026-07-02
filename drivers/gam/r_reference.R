#!/usr/bin/env Rscript
# mgcv::gam reference for one GAM case.
#
# Usage: Rscript r_reference.R <spec.json> <out.json>
#   spec.json: {data_csv, response, formula, family, method, sp (optional)}
# Writes out.json with every quantity the validation compares: parametric
# coefficients + SEs, per-smooth & total EDF, selected sp, GCV/REML score,
# deviance, AIC, scale, fitted values, and the summary smooth table
# (edf / Ref.df / statistic / p-value). Fails loud (non-zero exit) on any
# mgcv error so the Python driver records the case as an R-side failure.

suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
spec <- fromJSON(args[[1]])
out_path <- args[[2]]

d <- read.csv(spec$data_csv)
fam <- switch(spec$family,
              gaussian = gaussian(),
              poisson  = poisson(),
              binomial = binomial(),
              Gamma    = Gamma(link = "inverse"),
              `Gamma-log` = Gamma(link = "log"),
              stop(paste("unknown family", spec$family)))

form <- as.formula(spec$formula)
ctrl <- gam.control()

fit_args <- list(formula = form, data = d, family = fam, method = spec$method)
if (!is.null(spec$sp)) fit_args$sp <- as.numeric(spec$sp)

m <- do.call(gam, fit_args)
sm <- summary(m)

# Smooth-term table (order matches the smooths in the formula).
stab <- sm$s.table
smooth_edf   <- as.numeric(stab[, "edf"])
smooth_refdf <- as.numeric(stab[, "Ref.df"])
stat_col     <- if ("F" %in% colnames(stab)) "F" else "Chi.sq"
smooth_stat  <- as.numeric(stab[, stat_col])
smooth_p     <- as.numeric(stab[, "p-value"])

out <- list(
  family = spec$family, method = spec$method,
  coef = as.numeric(coef(m)),
  coef_names = names(coef(m)),
  # parametric table (intercept + any parametric covariates)
  p_coef = as.numeric(sm$p.coeff),
  p_se   = as.numeric(sm$se[seq_along(sm$p.coeff)]),
  sp = as.numeric(m$sp),
  smooth_edf = smooth_edf,
  smooth_refdf = smooth_refdf,
  smooth_stat = smooth_stat,
  smooth_stat_kind = stat_col,
  smooth_p = smooth_p,
  total_edf = sum(m$edf),
  gcv_ubre = m$gcv.ubre,               # GCV, UBRE, or -REML per method/family
  deviance = deviance(m),
  null_deviance = m$null.deviance,
  aic = AIC(m),
  scale = m$sig2,
  fitted = as.numeric(fitted(m)),
  n = nrow(d),
  n_coef = length(coef(m)),
  mgcv_version = as.character(packageVersion("mgcv"))
)

write_json(out, out_path, auto_unbox = TRUE, digits = 14, na = "null")
