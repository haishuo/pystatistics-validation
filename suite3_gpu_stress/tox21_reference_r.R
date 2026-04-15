#!/usr/bin/env Rscript
# ==========================================================================
# Tox21 HTS dose-response reference fits using R drc package.
#
# Fits 4-parameter log-logistic (LL.4) to all complete compounds from
# PubChem AID 743083 (Tox21 aromatase inhibitor qHTS screen).
#
# This is the gold-standard R computation that pystatsbio's GPU batch
# fitter must agree with.  Prints progress every 100 compounds.
#
# Prerequisites:
#   install.packages("drc")
#
# Usage:
#   Rscript tests/tox21_reference_r.R
#
# Output:
#   tests/data/tox21_r_fits.json  — per-compound fit results
#
# Expected runtime: 30-90 minutes for ~8,000 compounds
# ==========================================================================

suppressPackageStartupMessages(library(drc))

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}
data_dir <- file.path(script_dir, "data")

cat("=== Tox21 R Reference Fits ===\n")
cat("Loading dataset...\n")

# ---- Load data ----
# Header is row 1; rows 2-5 are metadata (RESULT_TYPE, RESULT_DESCR, etc.)
raw <- read.csv(file.path(data_dir, "tox21_aid743083.csv"),
                header = TRUE, stringsAsFactors = FALSE,
                check.names = FALSE)
raw <- raw[-(1:4), ]  # Drop 4 metadata rows
cat("  Data rows:", nrow(raw), "\n")

# Strip leading/trailing whitespace from column names
names(raw) <- trimws(names(raw))

# ---- Extract 8 well-spaced doses ----
dose_strs <- c("0.00614", "0.068", "0.341", "1.702", "3.794", "8.468", "32.48", "70.35")
doses <- as.numeric(dose_strs)
dose_cols <- paste0("Activity at ", dose_strs, " uM-Replicate_1")

# Verify columns exist
missing <- dose_cols[!dose_cols %in% names(raw)]
if (length(missing) > 0) stop("Missing columns: ", paste(missing, collapse = ", "))

# ---- Filter to complete cases ----
resp <- as.matrix(raw[, dose_cols])
mode(resp) <- "numeric"
complete <- apply(resp, 1, function(r) all(is.finite(r)))
cat("  Complete compounds:", sum(complete), "\n")

raw_clean <- raw[complete, ]
resp_clean <- resp[complete, ]
K <- nrow(resp_clean)

n_active <- sum(raw_clean$PUBCHEM_ACTIVITY_OUTCOME == "Active", na.rm = TRUE)
cat("  Active:", n_active, ", Inactive:", K - n_active, "\n")
cat("  Doses:", paste(doses, collapse = ", "), "uM\n\n")

# ---- Fit LL.4 to each compound ----
cat("Fitting LL.4 to", K, "compounds (this will take a while)...\n")
cat("Progress printed every 100 compounds.\n\n")

results <- vector("list", K)
n_converged <- 0
n_failed <- 0
t_start <- proc.time()[3]

for (i in seq_len(K)) {
  response_i <- as.numeric(resp_clean[i, ])

  # Build data frame for drc
  df_i <- data.frame(dose = doses, response = response_i)

  tryCatch({
    m <- drm(response ~ dose, data = df_i, fct = LL.4(),
             control = drmc(maxIt = 200, warnVal = -1, noMessage = TRUE))

    coefs <- coef(m)
    # drc LL.4 parameter names: b, c, d, e
    # b = hill (slope), c = lower limit, d = upper limit, e = EC50
    se_vals <- tryCatch(summary(m)$coefficients[, "Std. Error"],
                        error = function(e) rep(NA, 4))

    results[[i]] <- list(
      index     = i,
      converged = TRUE,
      hill      = as.numeric(coefs["b:(Intercept)"]),
      bottom    = as.numeric(coefs["c:(Intercept)"]),
      top       = as.numeric(coefs["d:(Intercept)"]),
      ec50      = as.numeric(coefs["e:(Intercept)"]),
      hill_se   = as.numeric(se_vals[1]),
      bottom_se = as.numeric(se_vals[2]),
      top_se    = as.numeric(se_vals[3]),
      ec50_se   = as.numeric(se_vals[4]),
      rss       = sum(residuals(m)^2),
      aic       = AIC(m),
      sid       = raw_clean$PUBCHEM_SID[i],
      outcome   = raw_clean$PUBCHEM_ACTIVITY_OUTCOME[i]
    )
    n_converged <- n_converged + 1
  }, error = function(e) {
    results[[i]] <<- list(
      index     = i,
      converged = FALSE,
      hill      = NA_real_, bottom = NA_real_, top = NA_real_, ec50 = NA_real_,
      hill_se   = NA_real_, bottom_se = NA_real_, top_se = NA_real_, ec50_se = NA_real_,
      rss       = NA_real_, aic = NA_real_,
      sid       = raw_clean$PUBCHEM_SID[i],
      outcome   = raw_clean$PUBCHEM_ACTIVITY_OUTCOME[i],
      error     = conditionMessage(e)
    )
    n_failed <<- n_failed + 1
  })

  # Progress report every 100 compounds
  if (i %% 100 == 0 || i == K) {
    elapsed <- proc.time()[3] - t_start
    rate <- i / elapsed
    eta <- (K - i) / rate
    cat(sprintf("  [%5d/%d] %d converged, %d failed | %.1f cmpd/s | ETA %.0fs\n",
                i, K, n_converged, n_failed, rate, eta))
  }
}

elapsed_total <- proc.time()[3] - t_start
cat(sprintf("\nDone. %d/%d converged in %.1f seconds (%.1f cmpd/s)\n",
            n_converged, K, elapsed_total, K / elapsed_total))

# ---- Write JSON ----
to_json <- function(x, indent = 0) {
  pad  <- paste(rep("  ", indent), collapse = "")
  pad1 <- paste(rep("  ", indent + 1), collapse = "")

  if (is.null(x)) return("null")
  if (is.logical(x) && length(x) == 1) return(tolower(as.character(x)))
  if (is.atomic(x) && length(x) == 1) {
    if (is.character(x)) return(paste0('"', gsub('"', '\\"', x), '"'))
    if (is.na(x)) return("null")
    if (is.numeric(x)) {
      if (is.infinite(x)) return("null")
      return(format(x, digits = 15, scientific = FALSE))
    }
    return(as.character(x))
  }
  if (is.atomic(x) && length(x) > 1) {
    items <- vapply(x, function(v) to_json(v, indent + 1), character(1))
    return(paste0("[", paste(items, collapse = ", "), "]"))
  }
  if (is.list(x)) {
    nms <- names(x)
    if (!is.null(nms)) {
      items <- mapply(function(k, v) {
        paste0(pad1, '"', k, '": ', to_json(v, indent + 1))
      }, nms, x, SIMPLIFY = FALSE, USE.NAMES = FALSE)
      return(paste0("{\n", paste(items, collapse = ",\n"), "\n", pad, "}"))
    } else {
      items <- lapply(x, function(v) paste0(pad1, to_json(v, indent + 1)))
      return(paste0("[\n", paste(items, collapse = ",\n"), "\n", pad, "]"))
    }
  }
  return(paste0('"', as.character(x), '"'))
}

output <- list(
  metadata = list(
    aid         = 743083,
    description = "Tox21 aromatase inhibitor qHTS (PubChem AID 743083)",
    n_compounds = K,
    n_converged = n_converged,
    n_failed    = n_failed,
    n_active    = n_active,
    doses       = doses,
    model       = "LL.4",
    r_version   = paste(R.version$major, R.version$minor, sep = "."),
    drc_version = as.character(packageVersion("drc")),
    elapsed_sec = round(elapsed_total, 1)
  ),
  fits = results
)

json_path <- file.path(data_dir, "tox21_r_fits.json")
cat("Writing results to:", json_path, "\n")
writeLines(to_json(output), json_path)
cat("Done.\n")
