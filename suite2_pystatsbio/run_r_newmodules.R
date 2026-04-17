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
# Rate standardization — Fleiss (1981, p.249) Down syndrome birth
# prevalence by maternal age and birth order. This is the example on
# the epitools::ageadjust.direct help page and recreates Table 1 of
# Fay & Feuer (1997, Stat Med 16:791-801).
#
# population[age, order] = live births for maternal-age × birth-order cell.
# count[age, order]      = Down syndrome cases for the same cell.
# We standardize birth-order group 1 to the average-population standard
# (the average across birth-order columns 1..5).
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Rate standardization (Fleiss 1981, Down syndrome) ===\n")
library(epitools)
# Copy the R help-page data verbatim (column-major fill, matching the
# printed table in ?ageadjust.direct).
population <- matrix(c(
    230061, 329449, 114920, 39487, 14208, 3052,
     72202, 326701, 208667, 83228, 28466, 5375,
     15050, 175702, 207081, 117300, 45026, 8660,
      2293,  68800, 132424, 98301, 46075, 9834,
       327,  30666, 123419, 149919, 104088, 34392,
    319933, 931318, 786511, 488235, 237863, 61313
), 6, 6,
   dimnames = list(
     age = c("Under 20","20-24","25-29","30-34","35-39","40 and over"),
     birth_order = c("1","2","3","4","5+","Total")))
count <- matrix(c(
    107, 141,  60,  40,  39,  25,
     25, 150, 110,  84,  82,  39,
      3,  71, 114, 103, 108,  75,
      1,  26,  64,  89, 137,  96,
      0,   8,  63, 112, 262, 295,
    136, 396, 411, 428, 628, 530
), 6, 6, dimnames = dimnames(population))

# Standard = average across birth-order columns 1..5 (as in help example).
standard <- rowMeans(population[, 1:5])

rs <- list(
    counts       = as.integer(count[, 1]),         # birth-order 1
    person_time  = as.integer(population[, 1]),    # births in that group
    standard_pop = as.numeric(standard),
    age_groups   = rownames(population),
    citation     = paste0(
        "Fleiss (1981) Statistical Methods for Rates and Proportions ",
        "p.249; example from epitools::ageadjust.direct; replicates ",
        "Table 1 of Fay & Feuer (1997) Stat Med 16:791-801"
    )
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
    dataset       = "Fleiss 1981 Down syndrome by maternal age, birth-order 1"
)
timing$rate_standardize_direct <- r$elapsed
cat("  adj rate per 100k:", 1e5 * direct["adj.rate"],
    "  (help page reports 92.3)\n")

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
