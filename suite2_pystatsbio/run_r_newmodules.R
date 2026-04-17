#!/usr/bin/env Rscript
#
# R reference results for NEW pystatsbio modules (v1.5.0).
#
# DATA POLICY: use canonical real-world R datasets (or published real data)
# where one exists for the module. The R script writes the exact inputs
# Python will read to CSVs/JSONs in fixtures/newmodules/, so both languages
# operate on byte-identical data.
#
# Dataset mapping:
#   epi_2by2         — Physicians' Health Study aspirin MI data (Steering
#                      Committee 1989), a published real 2x2.
#   mantel_haenszel  — datasets::UCBAdmissions, the canonical stratified
#                      (Simpson's paradox) dataset.
#   rate_standardize — simulated (no compact real dataset with both a
#                      standard population and age-stratified counts is
#                      canonical; documented as synthetic).
#   meta-analysis    — metafor::dat.bcg, the 13 BCG tuberculosis vaccine
#                      trials. THE canonical metafor example.
#   GEE              — geepack::dietox, 72 pigs weighed for 12 weeks.
#                      THE canonical geepack example.

library(jsonlite)
options(digits = 22)

FIX <- "suite2_pystatsbio/fixtures/newmodules"
dir.create(FIX, recursive = TRUE, showWarnings = FALSE)

results <- list()
timing  <- list()

time_it <- function(expr) {
    t <- system.time(val <- eval.parent(substitute(expr)))[["elapsed"]]
    list(value = val, elapsed = as.numeric(t))
}

# ─────────────────────────────────────────────────────────────────────
# 2x2 — Physicians' Health Study (NEJM 1989; doi:10.1056/NEJM198907203210301).
# Aspirin group: 11037 subjects, 104 MIs.
# Placebo group: 11034 subjects, 189 MIs.
# a/b/c/d layout: a=aspirin+MI, b=aspirin+no MI, c=placebo+MI, d=placebo+no MI
# ─────────────────────────────────────────────────────────────────────
cat("\n=== epi 2x2 (Physicians' Health Study aspirin MI) ===\n")
phs <- list(a = 104, b = 11037 - 104, c = 189, d = 11034 - 189,
            citation = "NEJM 1989; 321:129-135; Physicians' Health Study")
writeLines(toJSON(phs, auto_unbox = TRUE, pretty = TRUE),
           file.path(FIX, "epi_2by2.json"))
a <- phs$a; b <- phs$b; c <- phs$c; d <- phs$d
t0 <- proc.time()[["elapsed"]]
r1 <- a / (a + b); r0 <- c / (c + d)
rr <- r1 / r0
se_log_rr <- sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
or_val <- (a * d) / (b * c)
se_log_or <- sqrt(1/a + 1/b + 1/c + 1/d)
rd <- r1 - r0
se_rd <- sqrt(r1*(1-r1)/(a+b) + r0*(1-r0)/(c+d))
elapsed <- proc.time()[["elapsed"]] - t0
results$epi_2by2 <- list(
    risk_ratio = rr,
    rr_ci = c(exp(log(rr) - 1.96*se_log_rr), exp(log(rr) + 1.96*se_log_rr)),
    odds_ratio = or_val,
    or_ci = c(exp(log(or_val) - 1.96*se_log_or), exp(log(or_val) + 1.96*se_log_or)),
    risk_difference = rd, rd_se = se_rd,
    dataset = "Physicians' Health Study aspirin vs placebo MI (1989)"
)
timing$epi_2by2 <- elapsed
cat("  OR:", or_val, " RR:", rr, "  (aspirin halves MI risk — as reported)\n")

# ─────────────────────────────────────────────────────────────────────
# Mantel-Haenszel — UCBAdmissions (Bickel et al. 1975).
# 6 Berkeley graduate departments x {Admit, Reject} x {Male, Female}.
# Reshape to (2x2xK) strata where K = 6 departments.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Mantel-Haenszel (UCBAdmissions) ===\n")
ucb <- UCBAdmissions   # table[Admit, Gender, Dept]
# For MH we want strata = Dept and 2x2 = Admit × Gender.
# Convert to (2, 2, K) array with layout [[a=M+Admit, b=F+Admit],
#                                          [c=M+Reject, d=F+Reject]].
K <- dim(ucb)[3]
arr <- array(0, dim = c(2, 2, K))
for (k in seq_len(K)) {
    arr[1, 1, k] <- ucb["Admitted", "Male", k]
    arr[1, 2, k] <- ucb["Admitted", "Female", k]
    arr[2, 1, k] <- ucb["Rejected", "Male", k]
    arr[2, 2, k] <- ucb["Rejected", "Female", k]
}
# Save as JSON for Python.
mh_list <- list()
for (k in seq_len(K)) {
    mh_list[[k]] <- list(a = as.integer(arr[1,1,k]), b = as.integer(arr[1,2,k]),
                         c = as.integer(arr[2,1,k]), d = as.integer(arr[2,2,k]),
                         dept = dimnames(ucb)[[3]][k])
}
writeLines(toJSON(mh_list, auto_unbox = TRUE, pretty = TRUE),
           file.path(FIX, "mh_tables.json"))

r <- time_it(mantelhaen.test(arr, correct = FALSE, exact = FALSE))
mh <- r$value
# Manual MH OR
num <- 0; den <- 0
for (k in seq_len(K)) {
    t <- sum(arr[, , k])
    num <- num + arr[1,1,k] * arr[2,2,k] / t
    den <- den + arr[1,2,k] * arr[2,1,k] / t
}
mh_or <- num / den
results$mantel_haenszel <- list(
    pooled_or       = as.numeric(mh_or),
    r_mh_estimate   = as.numeric(mh$estimate),
    cmh_statistic   = as.numeric(mh$statistic),
    cmh_p_value     = as.numeric(mh$p.value),
    n_strata        = K,
    dataset         = "datasets::UCBAdmissions (Bickel 1975)"
)
timing$mantel_haenszel <- r$elapsed
cat("  MH OR:", mh_or, "CMH:", mh$statistic, "  (Simpson's paradox example)\n")

# ─────────────────────────────────────────────────────────────────────
# Rate standardization — simulated (no canonical R dataset).
# Copied from the previous synthetic version.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Rate standardization (synthetic) ===\n")
library(epitools)
rs <- list(
    counts        = c(8, 15, 22, 30, 40),
    person_time   = c(1000, 1500, 2000, 2500, 3000),
    standard_pop  = c(5000, 4000, 3000, 2000, 1000),
    standard_rates = c(0.005, 0.008, 0.012, 0.015, 0.020)
)
writeLines(toJSON(rs, auto_unbox = TRUE, pretty = TRUE),
           file.path(FIX, "rate_std.json"))
r <- time_it(ageadjust.direct(count = rs$counts, pop = rs$person_time,
                               stdpop = rs$standard_pop, conf.level = 0.95))
direct <- r$value
results$rate_standardize_direct <- list(
    crude_rate    = as.numeric(direct["crude.rate"]),
    adjusted_rate = as.numeric(direct["adj.rate"]),
    lci95         = as.numeric(direct["lci"]),
    uci95         = as.numeric(direct["uci"]),
    dataset       = "synthetic (no canonical compact real dataset)"
)
timing$rate_standardize_direct <- r$elapsed

# ─────────────────────────────────────────────────────────────────────
# Meta-analysis — metafor::dat.bcg (BCG tuberculosis vaccine trials).
# 13 trials, each reporting tpos/tneg/cpos/cneg. Compute log odds ratios
# and their variances with escalc(), then fit REML random-effects model.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== rma (metafor::dat.bcg) ===\n")
library(metafor)
data(dat.bcg, package = "metafor")
escres <- escalc(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg,
                 data = dat.bcg)
# yi and vi are now columns of escres (Hedges log OR and its sampling var).
meta_df <- data.frame(yi = escres$yi, vi = escres$vi,
                      trial = escres$trial, author = escres$author,
                      year = escres$year)
write.csv(meta_df, file.path(FIX, "meta_yi.csv"), row.names = FALSE)

r <- time_it(rma(yi = meta_df$yi, vi = meta_df$vi, method = "REML"))
fit <- r$value
results$rma <- list(
    estimate = as.numeric(fit$beta),
    se       = as.numeric(fit$se),
    ci_lb    = as.numeric(fit$ci.lb),
    ci_ub    = as.numeric(fit$ci.ub),
    tau2     = as.numeric(fit$tau2),
    q        = as.numeric(fit$QE),
    q_p      = as.numeric(fit$QEp),
    i2       = as.numeric(fit$I2),
    k        = as.numeric(fit$k),
    dataset  = "metafor::dat.bcg (13 BCG vaccine trials)"
)
timing$rma <- r$elapsed
cat("  estimate:", fit$beta, "tau2:", fit$tau2, "I2:", fit$I2, "\n")

# ─────────────────────────────────────────────────────────────────────
# GEE — geepack::dietox. 72 pigs, weight ~ time + feed protein level.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== gee (geepack::dietox) ===\n")
library(geepack)
data(dietox, package = "geepack")
# Use Weight ~ Time + Cu, clustering by Pig.
gd <- dietox[, c("Pig", "Weight", "Time", "Cu")]
gd <- gd[order(gd$Pig, gd$Time), ]
# Cu is ordered factor; convert to integer for pystatsbio.
gd$Cu <- as.integer(gd$Cu)
gd$Pig <- as.integer(as.character(gd$Pig))
write.csv(gd, file.path(FIX, "gee_long.csv"), row.names = FALSE)

r <- time_it(geeglm(Weight ~ Time + Cu, data = gd, id = Pig,
                     family = gaussian(), corstr = "exchangeable"))
fit <- r$value; s <- summary(fit)
results$gee <- list(
    coefficients = as.numeric(coef(fit)),
    robust_se    = as.numeric(s$coefficients[, "Std.err"]),
    alpha        = as.numeric(s$corr[1, 1]),
    n_clusters   = length(unique(gd$Pig)),
    dataset      = "geepack::dietox (72 pigs, weekly weights)"
)
timing$gee <- r$elapsed
cat("  coefs:", coef(fit), "  clusters:", length(unique(gd$Pig)), "\n")

# ─────────────────────────────────────────────────────────────────────
out <- list(results = results, timing = timing)
write_json(out, file.path(FIX, "r_results.json"),
           digits = 17, auto_unbox = TRUE, pretty = TRUE)
cat("\nWrote", file.path(FIX, "r_results.json"), "\n")
