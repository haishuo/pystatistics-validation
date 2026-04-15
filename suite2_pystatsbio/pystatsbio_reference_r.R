#!/usr/bin/env Rscript
# ==========================================================================
# PyStatsBio R reference script
#
# Generates reference results for cross-validating pystatsbio against R.
#
# Prerequisites:
#   install.packages(c("drc", "NonCompart", "pROC", "pwr", "PowerTOST"))
#
# Usage:
#   Rscript tests/pystatsbio_reference_r.R
#
# Outputs:
#   tests/data/theoph.csv               (Theophylline PK data)
#   tests/data/ryegrass.csv             (Dose-response data)
#   tests/data/asah.csv                 (Diagnostic data)
#   tests/data/pystatsbio_r_results.json (All reference results)
# ==========================================================================

suppressPackageStartupMessages({
  library(drc)
  library(NonCompart)
  library(pROC)
  library(pwr)
  library(PowerTOST)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}
data_dir <- file.path(script_dir, "data")
dir.create(data_dir, showWarnings = FALSE)

results <- list()

# --------------------------------------------------------------------------
# 0. Export datasets as CSV
# --------------------------------------------------------------------------
cat("Exporting datasets...\n")

# Theoph (built-in)
data(Theoph)
write.csv(Theoph, file.path(data_dir, "theoph.csv"), row.names = FALSE)
cat("  theoph.csv:", nrow(Theoph), "rows\n")

# ryegrass (from drc)
data(ryegrass)
write.csv(ryegrass, file.path(data_dir, "ryegrass.csv"), row.names = FALSE)
cat("  ryegrass.csv:", nrow(ryegrass), "rows\n")

# aSAH (from pROC)
data(aSAH)
# Convert factors to strings for CSV portability
asah_export <- aSAH
asah_export$outcome <- as.character(aSAH$outcome)
asah_export$gender  <- as.character(aSAH$gender)
write.csv(asah_export, file.path(data_dir, "asah.csv"), row.names = FALSE)
cat("  asah.csv:", nrow(aSAH), "rows\n")

# --------------------------------------------------------------------------
# 1. PK — NCA on Theoph Subject 1
# --------------------------------------------------------------------------
cat("\n=== PK: NCA (Theoph Subject 1) ===\n")

s1 <- Theoph[Theoph$Subject == 1, ]
dose_s1 <- s1$Dose[1]  # mg/kg

# NonCompart sNCA: extravascular, linear-up/log-down
nca_r <- sNCA(s1$Time, s1$conc, dose = dose_s1, adm = "Extravascular", down = "Log")

results$nca_subject1 <- list(
  dose      = dose_s1,
  cmax      = as.numeric(nca_r["CMAX"]),
  tmax      = as.numeric(nca_r["TMAX"]),
  auc_last  = as.numeric(nca_r["AUCLST"]),
  auc_inf   = as.numeric(nca_r["AUCIFO"]),
  pct_extrap = as.numeric(nca_r["AUCPEO"]),
  lambda_z  = as.numeric(nca_r["LAMZ"]),
  half_life = as.numeric(nca_r["LAMZHL"]),
  r_squared = as.numeric(nca_r["R2ADJ"]),
  n_terminal = as.numeric(nca_r["LAMZNPT"]),
  cl_f      = as.numeric(nca_r["CLFO"]),
  vz_f      = as.numeric(nca_r["VZFO"])
)

cat("  Cmax:", nca_r["CMAX"], "\n")
cat("  Tmax:", nca_r["TMAX"], "\n")
cat("  AUC(0-last):", nca_r["AUCLST"], "\n")
cat("  AUC(0-inf):", nca_r["AUCIFO"], "\n")
cat("  % extrap:", nca_r["AUCPEO"], "\n")
cat("  lambda_z:", nca_r["LAMZ"], "\n")
cat("  t1/2:", nca_r["LAMZHL"], "\n")
cat("  CL/F:", nca_r["CLFO"], "\n")
cat("  Vz/F:", nca_r["VZFO"], "\n")

# --------------------------------------------------------------------------
# 2. PK — NCA on multiple subjects (batch)
# --------------------------------------------------------------------------
cat("\n=== PK: NCA (all 12 subjects) ===\n")

nca_all <- list()
for (subj in unique(Theoph$Subject)) {
  si <- Theoph[Theoph$Subject == subj, ]
  ri <- sNCA(si$Time, si$conc, dose = si$Dose[1], adm = "Extravascular", down = "Log")
  nca_all[[as.character(subj)]] <- list(
    cmax     = as.numeric(ri["CMAX"]),
    tmax     = as.numeric(ri["TMAX"]),
    auc_last = as.numeric(ri["AUCLST"]),
    auc_inf  = as.numeric(ri["AUCIFO"]),
    half_life = as.numeric(ri["LAMZHL"]),
    lambda_z = as.numeric(ri["LAMZ"])
  )
  cat("  Subject", subj, ": Cmax=", ri["CMAX"], ", AUC=", ri["AUCLST"], "\n")
}
results$nca_all_subjects <- nca_all

# --------------------------------------------------------------------------
# 3. Dose-response — LL.4 fit (ryegrass)
# --------------------------------------------------------------------------
cat("\n=== Dose-Response: LL.4 fit (ryegrass) ===\n")

m <- drm(rootl ~ conc, data = ryegrass, fct = LL.4())
s <- summary(m)
coefs <- s$coefficients

# drc convention: b=hill (slope), c=lower, d=upper, e=ED50
# Map to pystatsbio: hill=b, bottom=c, top=d, ec50=e
results$drm_ll4 <- list(
  hill     = as.numeric(coefs["b:(Intercept)", "Estimate"]),
  bottom   = as.numeric(coefs["c:(Intercept)", "Estimate"]),
  top      = as.numeric(coefs["d:(Intercept)", "Estimate"]),
  ec50     = as.numeric(coefs["e:(Intercept)", "Estimate"]),
  hill_se  = as.numeric(coefs["b:(Intercept)", "Std. Error"]),
  bottom_se = as.numeric(coefs["c:(Intercept)", "Std. Error"]),
  top_se   = as.numeric(coefs["d:(Intercept)", "Std. Error"]),
  ec50_se  = as.numeric(coefs["e:(Intercept)", "Std. Error"]),
  rss      = sum(residuals(m)^2),
  aic      = AIC(m)
)

cat("  hill (b):", coefs["b:(Intercept)", "Estimate"], "\n")
cat("  bottom (c):", coefs["c:(Intercept)", "Estimate"], "\n")
cat("  top (d):", coefs["d:(Intercept)", "Estimate"], "\n")
cat("  ec50 (e):", coefs["e:(Intercept)", "Estimate"], "\n")
cat("  RSS:", sum(residuals(m)^2), "\n")
cat("  AIC:", AIC(m), "\n")

# ED50 with delta-method CI
ed50 <- ED(m, 50, interval = "delta", display = FALSE)
results$drm_ec50 <- list(
  estimate = as.numeric(ed50[1, "Estimate"]),
  se       = as.numeric(ed50[1, "Std. Error"]),
  ci_lower = as.numeric(ed50[1, "Lower"]),
  ci_upper = as.numeric(ed50[1, "Upper"])
)

cat("  ED50:", ed50[1, "Estimate"], " SE:", ed50[1, "Std. Error"], "\n")
cat("  ED50 CI: [", ed50[1, "Lower"], ",", ed50[1, "Upper"], "]\n")

# Predictions at original doses
pred_doses <- sort(unique(ryegrass$conc))
pred_vals  <- predict(m, newdata = data.frame(conc = pred_doses))
results$drm_predictions <- list(
  doses = pred_doses,
  predicted = as.numeric(pred_vals)
)
cat("  Predictions at unique doses:", paste(round(pred_vals, 4), collapse = ", "), "\n")

# --------------------------------------------------------------------------
# 4. Diagnostic — ROC on aSAH s100b
# --------------------------------------------------------------------------
cat("\n=== Diagnostic: ROC (aSAH s100b) ===\n")

# Binary: Poor=1, Good=0
response <- ifelse(aSAH$outcome == "Poor", 1, 0)

roc1 <- roc(response, aSAH$s100b, direction = "<", levels = c(0, 1), quiet = TRUE)
ci1  <- ci.auc(roc1, method = "delong")

results$roc_s100b <- list(
  auc        = as.numeric(auc(roc1)),
  auc_se     = sqrt(var(roc1)),
  ci_lower   = as.numeric(ci1[1]),
  ci_upper   = as.numeric(ci1[3]),
  direction  = roc1$direction,
  n_positive = sum(response == 1),
  n_negative = sum(response == 0)
)

cat("  AUC:", auc(roc1), "\n")
cat("  95% CI: [", ci1[1], ",", ci1[3], "]\n")
cat("  Direction:", roc1$direction, "\n")
cat("  n+:", sum(response == 1), ", n-:", sum(response == 0), "\n")

# Optimal cutoff (Youden)
co <- coords(roc1, "best", ret = c("threshold", "sensitivity", "specificity"),
             best.method = "youden")
results$roc_youden <- list(
  cutoff      = as.numeric(co$threshold),
  sensitivity = as.numeric(co$sensitivity),
  specificity = as.numeric(co$specificity)
)
cat("  Youden cutoff:", co$threshold, "\n")
cat("  Sensitivity:", co$sensitivity, ", Specificity:", co$specificity, "\n")

# --------------------------------------------------------------------------
# 5. Diagnostic — ROC on aSAH ndka + DeLong test
# --------------------------------------------------------------------------
cat("\n=== Diagnostic: ROC (aSAH ndka) + DeLong test ===\n")

roc2 <- roc(response, aSAH$ndka, direction = "<", levels = c(0, 1), quiet = TRUE)

results$roc_ndka <- list(
  auc = as.numeric(auc(roc2))
)

# DeLong test comparing s100b vs ndka
delong <- roc.test(roc1, roc2, method = "delong")

results$delong_test <- list(
  statistic = as.numeric(delong$statistic),
  p_value   = delong$p.value,
  auc1      = as.numeric(auc(roc1)),
  auc2      = as.numeric(auc(roc2)),
  auc_diff  = as.numeric(auc(roc1)) - as.numeric(auc(roc2))
)

cat("  AUC (ndka):", auc(roc2), "\n")
cat("  DeLong Z:", delong$statistic, "\n")
cat("  DeLong p:", delong$p.value, "\n")

# --------------------------------------------------------------------------
# 6. Diagnostic — Diagnostic accuracy at cutoff = 0.205
# --------------------------------------------------------------------------
cat("\n=== Diagnostic: Accuracy at cutoff = 0.205 ===\n")

cutoff_val <- 0.205
predicted_pos <- as.numeric(aSAH$s100b >= cutoff_val)
tp <- sum(predicted_pos == 1 & response == 1)
fp <- sum(predicted_pos == 1 & response == 0)
tn <- sum(predicted_pos == 0 & response == 0)
fn <- sum(predicted_pos == 0 & response == 1)

sens <- tp / (tp + fn)
spec <- tn / (tn + fp)
ppv  <- tp / (tp + fp)
npv  <- tn / (tn + fn)
lr_pos <- sens / (1 - spec)
lr_neg <- (1 - sens) / spec
dor    <- (tp * tn) / (fp * fn)

results$diag_accuracy <- list(
  cutoff      = cutoff_val,
  tp = tp, fp = fp, tn = tn, fn = fn,
  sensitivity = sens,
  specificity = spec,
  ppv         = ppv,
  npv         = npv,
  lr_positive = lr_pos,
  lr_negative = lr_neg,
  dor         = dor
)

cat("  TP:", tp, " FP:", fp, " TN:", tn, " FN:", fn, "\n")
cat("  Sensitivity:", sens, "\n")
cat("  Specificity:", spec, "\n")
cat("  PPV:", ppv, ", NPV:", npv, "\n")
cat("  LR+:", lr_pos, ", LR-:", lr_neg, "\n")
cat("  DOR:", dor, "\n")

# --------------------------------------------------------------------------
# 7. Power — t-test
# --------------------------------------------------------------------------
cat("\n=== Power: t-test ===\n")

# Solve for n
pw_t_n <- pwr.t.test(d = 0.5, power = 0.80, sig.level = 0.05,
                      type = "two.sample", alternative = "two.sided")
results$power_ttest_n <- list(
  n_exact = pw_t_n$n,
  n_ceil  = ceiling(pw_t_n$n),
  d       = pw_t_n$d,
  power   = pw_t_n$power,
  alpha   = pw_t_n$sig.level
)
cat("  n (exact):", pw_t_n$n, " -> ceil:", ceiling(pw_t_n$n), "\n")

# Solve for power
pw_t_p <- pwr.t.test(n = 64, d = 0.5, sig.level = 0.05,
                      type = "two.sample", alternative = "two.sided")
results$power_ttest_power <- list(
  n     = pw_t_p$n,
  d     = pw_t_p$d,
  power = pw_t_p$power,
  alpha = pw_t_p$sig.level
)
cat("  power (n=64, d=0.5):", pw_t_p$power, "\n")

# Solve for effect size
pw_t_d <- pwr.t.test(n = 100, power = 0.80, sig.level = 0.05,
                      type = "two.sample", alternative = "two.sided")
results$power_ttest_effect <- list(
  n     = pw_t_d$n,
  d     = pw_t_d$d,
  power = pw_t_d$power,
  alpha = pw_t_d$sig.level
)
cat("  d (n=100, power=0.80):", pw_t_d$d, "\n")

# One-sided
pw_t_1 <- pwr.t.test(d = 0.5, power = 0.80, sig.level = 0.05,
                      type = "two.sample", alternative = "greater")
results$power_ttest_onesided <- list(
  n_exact = pw_t_1$n,
  n_ceil  = ceiling(pw_t_1$n)
)
cat("  n (one-sided d=0.5):", pw_t_1$n, " -> ceil:", ceiling(pw_t_1$n), "\n")

# One-sample
pw_t_1s <- pwr.t.test(d = 0.5, power = 0.80, sig.level = 0.05,
                       type = "one.sample", alternative = "two.sided")
results$power_ttest_onesample <- list(
  n_exact = pw_t_1s$n,
  n_ceil  = ceiling(pw_t_1s$n)
)
cat("  n (one-sample d=0.5):", pw_t_1s$n, " -> ceil:", ceiling(pw_t_1s$n), "\n")

# Paired
pw_t_paired <- pwr.t.test(d = 0.5, power = 0.80, sig.level = 0.05,
                            type = "paired", alternative = "two.sided")
results$power_ttest_paired <- list(
  n_exact = pw_t_paired$n,
  n_ceil  = ceiling(pw_t_paired$n)
)
cat("  n (paired d=0.5):", pw_t_paired$n, " -> ceil:", ceiling(pw_t_paired$n), "\n")

# --------------------------------------------------------------------------
# 8. Power — ANOVA
# --------------------------------------------------------------------------
cat("\n=== Power: ANOVA ===\n")

pw_a <- pwr.anova.test(f = 0.25, k = 3, power = 0.80, sig.level = 0.05)
results$power_anova <- list(
  n_exact = pw_a$n,
  n_ceil  = ceiling(pw_a$n),
  f       = pw_a$f,
  k       = pw_a$k,
  power   = pw_a$power
)
cat("  ANOVA n (f=0.25, k=3):", pw_a$n, " -> ceil:", ceiling(pw_a$n), "\n")

# Solve for power
pw_a_p <- pwr.anova.test(f = 0.25, k = 3, n = 53, sig.level = 0.05)
results$power_anova_power <- list(
  n     = pw_a_p$n,
  power = pw_a_p$power
)
cat("  ANOVA power (n=53, f=0.25, k=3):", pw_a_p$power, "\n")

# --------------------------------------------------------------------------
# 9. Power — proportions
# --------------------------------------------------------------------------
cat("\n=== Power: proportions ===\n")

pw_p <- pwr.2p.test(h = 0.5, power = 0.80, sig.level = 0.05,
                     alternative = "two.sided")
results$power_prop <- list(
  n_exact = pw_p$n,
  n_ceil  = ceiling(pw_p$n),
  h       = pw_p$h,
  power   = pw_p$power
)
cat("  prop n (h=0.5):", pw_p$n, " -> ceil:", ceiling(pw_p$n), "\n")

# --------------------------------------------------------------------------
# 10. Power — crossover bioequivalence (PowerTOST)
# --------------------------------------------------------------------------
cat("\n=== Power: crossover bioequivalence ===\n")

pw_be <- sampleN.TOST(CV = 0.25, theta0 = 1, theta1 = 0.80, theta2 = 1.25,
                       targetpower = 0.80, alpha = 0.05, design = "2x2",
                       print = FALSE)

results$power_be <- list(
  n          = pw_be[["Sample size"]],
  power      = pw_be[["Achieved power"]],
  cv         = 0.25,
  theta1     = 0.80,
  theta2     = 1.25
)
cat("  BE n:", pw_be[["Sample size"]], ", power:", pw_be[["Achieved power"]], "\n")

# ==========================================================================
# Write JSON
# ==========================================================================

# Simple recursive JSON serializer (no external dependencies)
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

json_path <- file.path(data_dir, "pystatsbio_r_results.json")
writeLines(to_json(results), json_path)
cat("\n=== Results written to:", json_path, "===\n")
cat("Done.\n")
