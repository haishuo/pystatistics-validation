#!/usr/bin/env Rscript
#
# R reference for the ADVERSARIAL (hard-case) regression grid → JSON.
#
# One job: fit the R reference for one hard case and report not just the
# estimates but R's *behaviour* — the warnings it emits and whether the IRLS
# converged — so the validator can check that pystatistics matches R's failures
# and warnings, not only its coefficients (RIGOR R10).
#
# Two modes:
#   numeric  Rscript hardcase_run.R numeric <x_noint_csv> <y_csv> <family> <out_json>
#            Fit lm()/glm() on a numeric design Python built (intercept added by
#            R). Aliased columns come back as NA coefficients (R's drop-to-NA),
#            serialized as JSON null. Captures warnings + glm convergence.
#   frame    Rscript hardcase_run.R frame <frame_csv> <formula> <family> <out_json> <mm_out_csv>
#            Read a raw frame with string factor columns, fit with an R formula
#            so R builds the model matrix with its DEFAULT treatment contrasts,
#            and emit both the coefficients and the realized model matrix (to
#            <mm_out_csv>) so Python can confirm its own contrast coding matches.
#
#   family ∈ { lm, binomial, poisson }   (negbin is deliberately excluded — the
#            factor/contrast claim is isolated from the NB theta gap, per R10.)

options(digits = 22)
suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: hardcase_run.R <mode> ...")
mode <- args[[1]]

# Capture every warning glm/lm raises while still returning the fitted object.
fit_capturing <- function(expr) {
  warns <- character(0)
  val <- withCallingHandlers(
    expr,
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(fit = val, warnings = warns)
}

na_to_null <- function(x) lapply(x, function(v) if (is.na(v)) NULL else as.numeric(v))

fit_family <- function(fml, df, family) {
  if (family == "lm")            lm(fml, data = df)
  else if (family == "binomial") glm(fml, data = df, family = binomial)
  else if (family == "poisson")  glm(fml, data = df, family = poisson)
  else stop(sprintf("unknown family: %s", family))
}

if (mode == "numeric") {
  x_csv <- args[[2]]; y_csv <- args[[3]]; family <- args[[4]]; out_json <- args[[5]]
  X <- as.matrix(read.csv(x_csv, check.names = FALSE))
  y <- as.numeric(readLines(y_csv))
  df <- data.frame(.y = y, X, check.names = FALSE)
  predictors <- colnames(X)
  fml <- as.formula(paste0("`.y` ~ ",
                           paste(sprintf("`%s`", predictors), collapse = " + ")))

  cap <- fit_capturing(fit_family(fml, df, family))
  fit <- cap$fit
  s <- summary(fit)
  cm <- s$coefficients          # only NON-NA rows appear here
  full_coef <- coef(fit)        # includes NA for aliased columns, named

  # Align SE/stat/p (from cm, keyed by name) back onto the full coef vector so
  # NA (aliased) positions survive as null in all four vectors.
  coef_names <- names(full_coef)
  pick <- function(col) {
    out <- rep(NA_real_, length(coef_names))
    idx <- match(rownames(cm), coef_names)
    out[idx] <- cm[, col]
    out
  }
  out <- list(
    mode            = "numeric",
    family          = family,
    coef_names      = coef_names,
    coefficients    = na_to_null(as.numeric(full_coef)),
    standard_errors = na_to_null(pick(2)),
    test_statistic  = na_to_null(pick(3)),
    p_values        = na_to_null(pick(4)),
    n_aliased       = as.integer(sum(is.na(full_coef))),
    warnings        = as.list(cap$warnings),
    converged       = if (is.null(fit$converged)) TRUE else as.logical(fit$converged),
    n_iter          = if (is.null(fit$iter)) NA_integer_ else as.integer(fit$iter),
    n               = as.integer(nrow(df)),
    r_version       = as.character(getRversion())
  )
  write_json(out, out_json, auto_unbox = TRUE, digits = 16, null = "null", pretty = TRUE)
  cat(sprintf("R numeric %s: n=%d aliased=%d converged=%s warns=%d\n",
              family, nrow(df), out$n_aliased, out$converged, length(cap$warnings)))

} else if (mode == "frame") {
  frame_csv <- args[[2]]; formula_str <- args[[3]]; family <- args[[4]]
  out_json <- args[[5]]; mm_out_csv <- args[[6]]
  df <- read.csv(frame_csv, check.names = FALSE, stringsAsFactors = TRUE)
  fml <- as.formula(formula_str)

  cap <- fit_capturing(fit_family(fml, df, family))
  fit <- cap$fit
  s <- summary(fit)
  cm <- s$coefficients

  # The realized model matrix with R's DEFAULT treatment contrasts — written out
  # so Python can confirm its own term builder produced the identical columns.
  mm <- model.matrix(fit)
  write.csv(as.data.frame(mm), mm_out_csv, row.names = FALSE)

  out <- list(
    mode            = "frame",
    family          = family,
    formula         = formula_str,
    coef_names      = rownames(cm),
    coefficients    = as.numeric(cm[, 1]),
    standard_errors = as.numeric(cm[, 2]),
    test_statistic  = as.numeric(cm[, 3]),
    p_values        = as.numeric(cm[, 4]),
    mm_colnames     = colnames(mm),
    warnings        = as.list(cap$warnings),
    converged       = if (is.null(fit$converged)) TRUE else as.logical(fit$converged),
    n               = as.integer(nrow(df)),
    r_version       = as.character(getRversion())
  )
  write_json(out, out_json, auto_unbox = TRUE, digits = 16, pretty = TRUE)
  cat(sprintf("R frame %s: n=%d coefs=%d warns=%d\n",
              family, nrow(df), length(out$coef_names), length(cap$warnings)))

} else {
  stop(sprintf("unknown mode: %s", mode))
}
