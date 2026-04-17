#!/usr/bin/env Rscript
#
# R Reference Validation for Suite 1 (PyStatistics on California Housing)
#
# Generates reference results for: regression, descriptive, hypothesis,
# ANOVA, survival, mixed models, and Monte Carlo.
#
# Run from /mnt/projects/test:
#   Rscript suite1_pystatistics/run_r_validation.R
#

library(jsonlite)

options(digits = 22)

fixtures_dir <- "suite1_pystatistics/fixtures"
data_file <- file.path(fixtures_dir, "california_prepared.csv")

if (!file.exists(data_file)) {
    stop("Run generate_data.py first to create california_prepared.csv")
}

cat("Loading data...\n")
df <- read.csv(data_file, stringsAsFactors = TRUE)
cat(sprintf("  %d rows, %d columns\n", nrow(df), ncol(df)))

results <- list()
results$timing <- list()

time_elapsed <- function(expr) {
    # Evaluate expr in caller's environment and return elapsed seconds.
    t <- system.time(invisible(eval.parent(substitute(expr))))[["elapsed"]]
    as.numeric(t)
}

# ──────────────────────────────────────────────────────────────────────
# Regression
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Regression ===\n")

# OLS: MedHouseVal ~ all numeric predictors
numeric_cols <- c("MedInc", "HouseAge", "AveRooms", "AveBedrms",
                  "Population", "AveOccup", "Latitude", "Longitude")
fml <- as.formula(paste("MedHouseVal ~", paste(numeric_cols, collapse = " + ")))
results$timing$ols <- time_elapsed(ols_fit <- lm(fml, data = df))
s <- summary(ols_fit)
cat("  OLS R-squared:", s$r.squared, "\n")

results$ols <- list(
    coefficients = as.numeric(coef(ols_fit)),
    standard_errors = as.numeric(s$coefficients[, "Std. Error"]),
    t_statistics = as.numeric(s$coefficients[, "t value"]),
    p_values = as.numeric(s$coefficients[, "Pr(>|t|)"]),
    r_squared = s$r.squared,
    adj_r_squared = s$adj.r.squared,
    residual_std_error = s$sigma,
    df_residual = ols_fit$df.residual,
    fitted_first10 = as.numeric(fitted(ols_fit)[1:10]),
    residuals_first10 = as.numeric(residuals(ols_fit)[1:10])
)

# GLM Binomial: high_value ~ numeric predictors (subset)
glm_bin_fml <- high_value ~ MedInc + HouseAge + AveRooms + Population
results$timing$glm_binomial <- time_elapsed(
    glm_bin <- glm(glm_bin_fml, data = df, family = binomial)
)
s_bin <- summary(glm_bin)
cat("  GLM Binomial deviance:", glm_bin$deviance, "\n")

results$glm_binomial <- list(
    coefficients = as.numeric(coef(glm_bin)),
    standard_errors = as.numeric(s_bin$coefficients[, "Std. Error"]),
    deviance = glm_bin$deviance,
    null_deviance = glm_bin$null.deviance,
    aic = glm_bin$aic
)

# GLM Poisson: round(Population/100) ~ predictors
df$pop_count <- round(df$Population / 100)
glm_pois_fml <- pop_count ~ MedInc + HouseAge + AveOccup
results$timing$glm_poisson <- time_elapsed(
    glm_pois <- glm(glm_pois_fml, data = df, family = poisson)
)
s_pois <- summary(glm_pois)
cat("  GLM Poisson deviance:", glm_pois$deviance, "\n")

results$glm_poisson <- list(
    coefficients = as.numeric(coef(glm_pois)),
    standard_errors = as.numeric(s_pois$coefficients[, "Std. Error"]),
    deviance = glm_pois$deviance,
    null_deviance = glm_pois$null.deviance,
    aic = glm_pois$aic
)

# ──────────────────────────────────────────────────────────────────────
# Descriptive Statistics
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Descriptive Statistics ===\n")

library(moments)

num_data <- df[, numeric_cols]

results$descriptive <- list(
    means = as.numeric(colMeans(num_data)),
    sds = as.numeric(apply(num_data, 2, sd)),
    skewness = as.numeric(apply(num_data, 2, skewness)),
    kurtosis = as.numeric(apply(num_data, 2, kurtosis)),
    quantiles_25 = as.numeric(apply(num_data, 2, quantile, probs = 0.25, type = 7)),
    quantiles_50 = as.numeric(apply(num_data, 2, quantile, probs = 0.50, type = 7)),
    quantiles_75 = as.numeric(apply(num_data, 2, quantile, probs = 0.75, type = 7)),
    cor_pearson = as.matrix(cor(num_data, method = "pearson")),
    cor_spearman = as.matrix(cor(num_data, method = "spearman")),
    column_names = numeric_cols
)
cat("  Computed means, sds, skewness, kurtosis, quantiles, correlations\n")

# ──────────────────────────────────────────────────────────────────────
# Hypothesis Tests
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Hypothesis Tests ===\n")

# Two-sample t-test: MedInc by high_value
hv0 <- df$MedInc[df$high_value == 0]
hv1 <- df$MedInc[df$high_value == 1]
tt_welch <- t.test(hv0, hv1, var.equal = FALSE)
tt_equal <- t.test(hv0, hv1, var.equal = TRUE)

results$t_test_welch <- list(
    statistic = as.numeric(tt_welch$statistic),
    p_value = tt_welch$p.value,
    df = as.numeric(tt_welch$parameter),
    conf_int = as.numeric(tt_welch$conf.int),
    mean_x = as.numeric(tt_welch$estimate[1]),
    mean_y = as.numeric(tt_welch$estimate[2])
)

results$t_test_equal <- list(
    statistic = as.numeric(tt_equal$statistic),
    p_value = tt_equal$p.value,
    df = as.numeric(tt_equal$parameter),
    conf_int = as.numeric(tt_equal$conf.int)
)

# Paired t-test: standardized MedInc vs AveRooms
set.seed(42)
n_paired <- min(1000, nrow(df))
idx <- sample(nrow(df), n_paired)
x_paired <- scale(df$MedInc[idx])
y_paired <- scale(df$AveRooms[idx])
tt_paired <- t.test(x_paired, y_paired, paired = TRUE)

results$t_test_paired <- list(
    statistic = as.numeric(tt_paired$statistic),
    p_value = tt_paired$p.value,
    df = as.numeric(tt_paired$parameter),
    conf_int = as.numeric(tt_paired$conf.int),
    paired_indices = idx
)

# Chi-squared test: region x high_value
ct <- table(df$region, df$high_value)
chisq_result <- chisq.test(ct, correct = FALSE)

results$chisq <- list(
    statistic = as.numeric(chisq_result$statistic),
    p_value = chisq_result$p.value,
    df = as.numeric(chisq_result$parameter),
    observed = matrix(as.numeric(chisq_result$observed), nrow=nrow(chisq_result$observed)),
    expected = matrix(as.numeric(chisq_result$expected), nrow=nrow(chisq_result$expected))
)
cat("  Chi-squared statistic:", chisq_result$statistic, "\n")

# Wilcoxon rank-sum test
south <- df$MedHouseVal[df$region == "South"]
north <- df$MedHouseVal[df$region == "North"]
# Use subsets to keep it manageable
set.seed(43)
south_sub <- sample(south, min(500, length(south)))
north_sub <- sample(north, min(500, length(north)))
wilcox_result <- wilcox.test(south_sub, north_sub, exact = FALSE, correct = TRUE)

results$wilcoxon <- list(
    statistic = as.numeric(wilcox_result$statistic),
    p_value = wilcox_result$p.value,
    south_n = length(south_sub),
    north_n = length(north_sub),
    south_values = as.numeric(south_sub),
    north_values = as.numeric(north_sub)
)

# KS test: MedInc vs normal
ks_result <- ks.test(df$MedInc, "pnorm", mean(df$MedInc), sd(df$MedInc))

results$ks_test <- list(
    statistic = as.numeric(ks_result$statistic),
    p_value = ks_result$p.value
)

# Proportion test
prop_ct <- table(df$region, df$high_value)
# Two-group proportion: South vs North high_value rate
south_hv <- sum(df$high_value[df$region == "South"])
south_n <- sum(df$region == "South")
north_hv <- sum(df$high_value[df$region == "North"])
north_n <- sum(df$region == "North")
prop_result <- prop.test(c(south_hv, north_hv), c(south_n, north_n), correct = TRUE)

results$prop_test <- list(
    statistic = as.numeric(prop_result$statistic),
    p_value = prop_result$p.value,
    conf_int = as.numeric(prop_result$conf.int),
    estimate = as.numeric(prop_result$estimate),
    x = c(south_hv, north_hv),
    n = c(south_n, north_n)
)

# ──────────────────────────────────────────────────────────────────────
# ANOVA
# ──────────────────────────────────────────────────────────────────────
cat("\n=== ANOVA ===\n")

library(car)

# One-way ANOVA: MedHouseVal ~ region
results$timing$anova_oneway <- time_elapsed(
    aov1 <- aov(MedHouseVal ~ region, data = df)
)
aov1_s <- summary(aov1)[[1]]

results$anova_oneway <- list(
    f_value = as.numeric(aov1_s[["F value"]][1]),
    p_value = as.numeric(aov1_s[["Pr(>F)"]][1]),
    df_between = as.numeric(aov1_s[["Df"]][1]),
    df_within = as.numeric(aov1_s[["Df"]][2]),
    ss_between = as.numeric(aov1_s[["Sum Sq"]][1]),
    ss_within = as.numeric(aov1_s[["Sum Sq"]][2]),
    ms_between = as.numeric(aov1_s[["Mean Sq"]][1]),
    ms_within = as.numeric(aov1_s[["Mean Sq"]][2])
)
cat("  One-way F:", aov1_s[["F value"]][1], "\n")

# Factorial ANOVA Type II: MedHouseVal ~ region * old_house
results$timing$anova_factorial <- time_elapsed({
    aov2 <- lm(MedHouseVal ~ region * factor(old_house), data = df)
    aov2_typeII <- Anova(aov2, type = 2)
})

results$anova_factorial <- list(
    ss = as.numeric(aov2_typeII[["Sum Sq"]]),
    df = as.numeric(aov2_typeII[["Df"]]),
    f_value = as.numeric(aov2_typeII[["F value"]]),
    p_value = as.numeric(aov2_typeII[["Pr(>F)"]]),
    terms = rownames(aov2_typeII)
)

# Tukey HSD
tukey <- TukeyHSD(aov1)$region
results$tukey_hsd <- list(
    comparisons = rownames(tukey),
    diff = as.numeric(tukey[, "diff"]),
    lwr = as.numeric(tukey[, "lwr"]),
    upr = as.numeric(tukey[, "upr"]),
    p_adj = as.numeric(tukey[, "p adj"])
)

# Levene's test
lev <- leveneTest(MedHouseVal ~ region, data = df)
results$levene <- list(
    f_value = as.numeric(lev[["F value"]][1]),
    p_value = as.numeric(lev[["Pr(>F)"]][1]),
    df1 = as.numeric(lev[["Df"]][1]),
    df2 = as.numeric(lev[["Df"]][2])
)

# ──────────────────────────────────────────────────────────────────────
# Survival Analysis
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Survival Analysis ===\n")

library(survival)

# KM and log-rank on REAL survival data (survival::lung) — same dataset
# used by Cox PH below. California Housing's HouseAge/high_value is not a
# real survival process.
data(lung, package = "survival")
lung_km <- lung[complete.cases(lung[, c("time", "status", "sex")]), ]
lung_km$event <- as.numeric(lung_km$status == 2)

# Kaplan-Meier (overall curve)
km_time_start <- proc.time()[["elapsed"]]
km_fit <- survfit(Surv(time, event) ~ 1, data = lung_km)
km_elapsed <- proc.time()[["elapsed"]] - km_time_start
km_summary <- summary(km_fit)

results$kaplan_meier <- list(
    time = as.numeric(km_summary$time),
    survival = as.numeric(km_summary$surv),
    n_risk = as.numeric(km_summary$n.risk),
    n_events = as.numeric(km_summary$n.event),
    std_err = as.numeric(km_summary$std.err),
    lower = as.numeric(km_summary$lower),
    upper = as.numeric(km_summary$upper),
    median_survival = as.numeric(quantile(km_fit, probs = 0.5)$quantile),
    dataset = "survival::lung (overall KM curve)"
)

# Log-rank test: lung survival by sex (1=M, 2=F) — canonical lung example
lr_start <- proc.time()[["elapsed"]]
logrank <- survdiff(Surv(time, event) ~ sex, data = lung_km)
lr_elapsed <- proc.time()[["elapsed"]] - lr_start

results$logrank <- list(
    statistic = as.numeric(logrank$chisq),
    df = length(logrank$n) - 1,
    p_value = as.numeric(1 - pchisq(logrank$chisq, df = length(logrank$n) - 1)),
    observed = as.numeric(logrank$obs),
    expected = as.numeric(logrank$exp),
    dataset = "survival::lung (log-rank by sex)"
)
# Save the lung_km CSV so Python can load the same rows (same NA-drop)
write.csv(lung_km[, c("time", "event", "sex")],
          file.path(fixtures_dir, "lung_km.csv"), row.names = FALSE)
cat("  KM n:", sum(km_summary$n.risk[1:1]), " log-rank chi-sq:", logrank$chisq, "\n")

# Record timings
if (is.null(results$timing)) results$timing <- list()
results$timing$kaplan_meier <- as.numeric(km_elapsed)
results$timing$logrank <- as.numeric(lr_elapsed)

# Cox PH on the NCCTG lung cancer dataset (survival::lung).
# This is the canonical real-world Cox PH dataset — Loprinzi et al. 1994,
# advanced lung cancer survival from the North Central Cancer Treatment
# Group. Pystatistics should match R exactly on this; it's the dataset
# every R survival tutorial uses.
#
# Columns:
#   time     survival time in days
#   status   1 = censored, 2 = dead
#   age      age in years
#   sex      1 = male, 2 = female
#   ph.ecog  ECOG performance score (0 = best, 5 = dead), a few NAs
#
# Drop rows with any NA in the columns we use, then fit
# Surv(time, event) ~ age + sex + ph.ecog.
data(lung, package = "survival")
cox_df <- lung[, c("time", "status", "age", "sex", "ph.ecog")]
cox_df <- cox_df[complete.cases(cox_df), ]
cox_df$event <- as.numeric(cox_df$status == 2)

results$timing$coxph <- time_elapsed(
    cox_fit <- coxph(Surv(time, event) ~ age + sex + ph.ecog, data = cox_df)
)
cox_s <- summary(cox_fit)

# Save the prepared dataset as a CSV so Python reads the EXACT same rows
# (after NA drop) that R fit. The CSV is the source of truth.
cox_csv <- file.path(fixtures_dir, "lung_coxph.csv")
write.csv(cox_df[, c("time", "event", "age", "sex", "ph.ecog")],
          cox_csv, row.names = FALSE)

results$coxph <- list(
    coefficients = as.numeric(coef(cox_fit)),
    se = as.numeric(cox_s$coefficients[, "se(coef)"]),
    hazard_ratios = as.numeric(cox_s$coefficients[, "exp(coef)"]),
    p_values = as.numeric(cox_s$coefficients[, "Pr(>|z|)"]),
    concordance = as.numeric(cox_s$concordance[1]),
    n = nrow(cox_df),
    n_events = sum(cox_df$event),
    covariates = c("age", "sex", "ph.ecog"),
    dataset = "survival::lung (NCCTG advanced lung cancer, complete cases)"
)
cat("  Cox PH coefs (lung, age+sex+ph.ecog):", coef(cox_fit), "\n")
cat("  n =", nrow(cox_df), "events =", sum(cox_df$event), "\n")
cat("  Concordance:", cox_s$concordance[1], "\n")

# ──────────────────────────────────────────────────────────────────────
# Mixed Models
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Mixed Models ===\n")

library(lme4)
library(lmerTest)

# Use subset for speed (mixed models are slow on 20K rows)
set.seed(45)
mm_idx <- sample(nrow(df), 3000)
mm_df <- df[mm_idx, ]

# LMM: MedHouseVal ~ MedInc + HouseAge + (1 | block_id)
results$timing$lmm <- time_elapsed(
    lmm_fit <- lmer(MedHouseVal ~ MedInc + HouseAge + (1 | block_id), data = mm_df)
)
lmm_s <- summary(lmm_fit)

vc <- as.data.frame(VarCorr(lmm_fit))
results$lmm <- list(
    fixed_effects = as.numeric(fixef(lmm_fit)),
    fixed_names = names(fixef(lmm_fit)),
    se = as.numeric(lmm_s$coefficients[, "Std. Error"]),
    t_values = as.numeric(lmm_s$coefficients[, "t value"]),
    var_block = as.numeric(vc$vcov[vc$grp == "block_id"]),
    var_residual = as.numeric(vc$vcov[vc$grp == "Residual"]),
    loglik = as.numeric(logLik(lmm_fit)),
    aic = as.numeric(AIC(lmm_fit)),
    bic = as.numeric(BIC(lmm_fit)),
    mm_indices = mm_idx
)
cat("  LMM fixed effects:", fixef(lmm_fit), "\n")

# ──────────────────────────────────────────────────────────────────────
# Monte Carlo (Bootstrap)
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Monte Carlo ===\n")

library(boot)

# Bootstrap mean of MedInc
set.seed(46)
boot_data <- df$MedInc[1:1000]
mean_fn <- function(data, indices) { mean(data[indices]) }
results$timing$bootstrap <- time_elapsed(
    boot_result <- boot(boot_data, mean_fn, R = 2000)
)

boot_ci_result <- boot.ci(boot_result, type = "perc")

results$bootstrap <- list(
    t0 = as.numeric(boot_result$t0),
    se = sd(boot_result$t),
    bias = mean(boot_result$t) - boot_result$t0,
    ci_perc_lower = as.numeric(boot_ci_result$percent[4]),
    ci_perc_upper = as.numeric(boot_ci_result$percent[5]),
    data = as.numeric(boot_data)
)
cat("  Bootstrap mean:", boot_result$t0, "SE:", sd(boot_result$t), "\n")

# ──────────────────────────────────────────────────────────────────────
# Save all results
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Saving results ===\n")
out_path <- file.path(fixtures_dir, "r_results.json")
write_json(results, out_path, digits = 17, auto_unbox = TRUE, pretty = TRUE)
cat("Saved to:", out_path, "\n")
cat("Done.\n")
