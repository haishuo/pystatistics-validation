#!/usr/bin/env Rscript
#
# R Reference Validation for Suite 2 (PyStatsBio on NHANES + synthetic data)
#
# Generates reference results for: power, dose-response, diagnostic, PK
#
# Run from /mnt/projects/test:
#   Rscript suite2_pystatsbio/run_r_validation.R
#

library(jsonlite)
options(digits = 22)

fixtures_dir <- "suite2_pystatsbio/fixtures"
data_dir <- "data"

results <- list()

# ──────────────────────────────────────────────────────────────────────
# Power / Sample Size (pwr package)
# ──────────────────────────────────────────────────────────────────────
cat("=== Power Analysis ===\n")

library(pwr)

# Load NHANES to compute observed effect sizes
nhanes <- read.csv(file.path(data_dir, "nhanes_biomarker.csv"))
cat(sprintf("  NHANES: %d rows\n", nrow(nhanes)))

# Observed effect size: creatinine by diabetic status
if ("LBXSC3SI" %in% names(nhanes) && "diabetic" %in% names(nhanes)) {
    d0 <- nhanes$LBXSC3SI[nhanes$diabetic == 0]
    d1 <- nhanes$LBXSC3SI[nhanes$diabetic == 1]
    d0 <- d0[!is.na(d0)]
    d1 <- d1[!is.na(d1)]
    pooled_sd <- sqrt(((length(d0)-1)*var(d0) + (length(d1)-1)*var(d1)) / (length(d0)+length(d1)-2))
    obs_d <- abs(mean(d1) - mean(d0)) / pooled_sd
    cat(sprintf("  Observed Cohen's d (creatinine): %.4f\n", obs_d))
} else {
    obs_d <- 0.5  # fallback
}

# Two-sample t-test: solve for n
pwr_n <- pwr.t.test(d = obs_d, sig.level = 0.05, power = 0.80, type = "two.sample")
results$power_t_n <- list(
    n = ceiling(pwr_n$n),
    n_exact = pwr_n$n,
    d = obs_d,
    power = 0.80,
    alpha = 0.05
)
cat(sprintf("  power_t_test(d=%.4f, power=0.80) -> n = %.2f\n", obs_d, pwr_n$n))

# Two-sample t-test: solve for power
pwr_pow <- pwr.t.test(d = obs_d, n = 100, sig.level = 0.05, type = "two.sample")
results$power_t_power <- list(
    n = 100,
    d = obs_d,
    power = pwr_pow$power,
    alpha = 0.05
)

# Paired t-test: solve for n
pwr_paired <- pwr.t.test(d = 0.3, sig.level = 0.05, power = 0.80, type = "paired")
results$power_paired_n <- list(
    n_exact = pwr_paired$n,
    d = 0.3,
    power = 0.80
)

# Proportion test: solve for n
# Observed proportion of diabetes
p_diab <- mean(nhanes$diabetic, na.rm = TRUE)
p_ref <- 0.10  # reference rate
h <- 2 * asin(sqrt(p_diab)) - 2 * asin(sqrt(p_ref))
pwr_prop <- pwr.2p.test(h = abs(h), sig.level = 0.05, power = 0.80)
results$power_prop_n <- list(
    n_exact = pwr_prop$n,
    h = abs(h),
    p1 = p_diab,
    p2 = p_ref,
    power = 0.80
)

# One-way ANOVA power
pwr_anova <- pwr.anova.test(k = 3, f = 0.25, sig.level = 0.05, power = 0.80)
results$power_anova_n <- list(
    n_exact = pwr_anova$n,
    k = 3,
    f = 0.25,
    power = 0.80
)

# ──────────────────────────────────────────────────────────────────────
# Dose-Response (drc package)
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Dose-Response ===\n")

library(drc)

# Fit 5 representative compounds
compound_files <- list.files(fixtures_dir, pattern = "compound_.*\\.json$", full.names = TRUE)
results$doseresponse <- list()

for (cfile in compound_files) {
    comp <- fromJSON(cfile)
    cname <- comp$name
    cat(sprintf("  Fitting %s...\n", cname))

    dose <- comp$dose
    response <- comp$response

    # Replace dose=0 with small value for drc (log-logistic needs positive dose)
    dose[dose == 0] <- 1e-10

    tryCatch({
        drm_fit <- drm(response ~ dose, fct = LL.4(),
                       data = data.frame(dose = dose, response = response))
        s <- summary(drm_fit)

        # Extract parameters (drc order: b=hill, c=bottom, d=top, e=ec50)
        params <- coef(drm_fit)

        # ED50
        ed50 <- ED(drm_fit, 50, interval = "delta", display = FALSE)

        results$doseresponse[[cname]] <- list(
            hill = as.numeric(params["b:(Intercept)"]),
            bottom = as.numeric(params["c:(Intercept)"]),
            top = as.numeric(params["d:(Intercept)"]),
            ec50 = as.numeric(params["e:(Intercept)"]),
            ec50_se = as.numeric(ed50[, "Std. Error"]),
            ec50_lower = as.numeric(ed50[, "Lower"]),
            ec50_upper = as.numeric(ed50[, "Upper"]),
            rss = sum(residuals(drm_fit)^2),
            converged = TRUE
        )
    }, error = function(e) {
        cat(sprintf("    FAILED: %s\n", e$message))
        results$doseresponse[[cname]] <<- list(converged = FALSE, error = e$message)
    })
}

# ──────────────────────────────────────────────────────────────────────
# Diagnostic Accuracy (pROC package)
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Diagnostic Accuracy ===\n")

library(pROC)

# ROC for HbA1c -> diabetic
if ("LBXGH" %in% names(nhanes) && "diabetic" %in% names(nhanes)) {
    roc_hba1c <- roc(nhanes$diabetic, nhanes$LBXGH, direction = "<", quiet = TRUE)
    ci_hba1c <- ci.auc(roc_hba1c)

    results$roc_hba1c <- list(
        auc = as.numeric(roc_hba1c$auc),
        auc_se = as.numeric(sqrt(var(roc_hba1c))),
        auc_ci_lower = as.numeric(ci_hba1c[1]),
        auc_ci_upper = as.numeric(ci_hba1c[3]),
        direction = roc_hba1c$direction,
        n_positive = sum(nhanes$diabetic == 1),
        n_negative = sum(nhanes$diabetic == 0)
    )
    cat(sprintf("  HbA1c AUC: %.6f\n", roc_hba1c$auc))
}

# ROC for glucose -> diabetic
if ("LBXSGL" %in% names(nhanes) && "diabetic" %in% names(nhanes)) {
    roc_gluc <- roc(nhanes$diabetic, nhanes$LBXSGL, direction = "<", quiet = TRUE)
    ci_gluc <- ci.auc(roc_gluc)

    results$roc_glucose <- list(
        auc = as.numeric(roc_gluc$auc),
        auc_se = as.numeric(sqrt(var(roc_gluc))),
        auc_ci_lower = as.numeric(ci_gluc[1]),
        auc_ci_upper = as.numeric(ci_gluc[3])
    )
    cat(sprintf("  Glucose AUC: %.6f\n", roc_gluc$auc))

    # DeLong test: HbA1c vs Glucose
    if (exists("roc_hba1c")) {
        delong <- roc.test(roc_hba1c, roc_gluc, method = "delong")
        results$roc_test_delong <- list(
            statistic = as.numeric(delong$statistic),
            p_value = delong$p.value,
            auc1 = as.numeric(roc_hba1c$auc),
            auc2 = as.numeric(roc_gluc$auc)
        )
        cat(sprintf("  DeLong test p-value: %.6f\n", delong$p.value))
    }
}

# Batch AUC: compute AUC for all biomarker columns
biomarker_cols <- c("LBXSATSI", "LBXSAL", "LBXSAPSI", "LBXSC3SI",
                    "LBXSBU", "LBXSCA", "LBXSCH", "LBXSGL",
                    "LBXSIR", "LBXSKSI", "LBXSNASI", "LBXSPH",
                    "LBXSTB", "LBXSTP", "LBXSTR", "LBXSUA")
available_bio <- biomarker_cols[biomarker_cols %in% names(nhanes)]

batch_aucs <- numeric(length(available_bio))
batch_ses <- numeric(length(available_bio))
names(batch_aucs) <- available_bio
names(batch_ses) <- available_bio

for (i in seq_along(available_bio)) {
    col <- available_bio[i]
    tryCatch({
        r <- roc(nhanes$diabetic, nhanes[[col]], direction = "<", quiet = TRUE)
        batch_aucs[i] <- as.numeric(r$auc)
        batch_ses[i] <- as.numeric(sqrt(var(r)))
    }, error = function(e) {
        batch_aucs[i] <<- NA
        batch_ses[i] <<- NA
    })
}

results$batch_auc <- list(
    columns = available_bio,
    auc = as.numeric(batch_aucs),
    se = as.numeric(batch_ses)
)
cat(sprintf("  Batch AUC: %d biomarkers computed\n", sum(!is.na(batch_aucs))))

# Optimal cutoff (Youden)
if (exists("roc_hba1c")) {
    sens <- roc_hba1c$sensitivities
    spec <- roc_hba1c$specificities
    j <- sens + spec - 1
    best_idx <- which.max(j)
    results$optimal_cutoff_hba1c <- list(
        cutoff = as.numeric(roc_hba1c$thresholds[best_idx]),
        sensitivity = as.numeric(sens[best_idx]),
        specificity = as.numeric(spec[best_idx]),
        youden_j = as.numeric(j[best_idx])
    )
    cat(sprintf("  Optimal HbA1c cutoff (Youden): %.2f\n", roc_hba1c$thresholds[best_idx]))
}

# ──────────────────────────────────────────────────────────────────────
# Pharmacokinetics (PKNCA)
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Pharmacokinetics ===\n")

library(PKNCA)

pk_subjects <- fromJSON(file.path(fixtures_dir, "pk_subjects.json"), simplifyVector = FALSE)
results$pk <- list()

for (i in seq_along(pk_subjects)) {
    subj <- pk_subjects[[i]]
    sname <- paste0("subject_", i)
    cat(sprintf("  NCA for %s (%s)...\n", sname, subj$route))

    time <- unlist(subj$time)
    conc <- unlist(subj$concentration)
    dose_val <- subj$dose
    route <- subj$route

    tryCatch({
        # Build PKNCA objects
        conc_df <- data.frame(time = as.numeric(time), conc = as.numeric(conc), subject = 1)
        conc_obj <- PKNCAconc(conc_df, conc ~ time | subject)
        dose_obj <- PKNCAdose(
            data.frame(time = 0, dose = as.numeric(dose_val), subject = 1),
            dose ~ time | subject
        )
        data_obj <- PKNCAdata(conc_obj, dose_obj,
                              intervals = data.frame(
                                  start = 0, end = max(as.numeric(time)),
                                  auclast = TRUE, cmax = TRUE, tmax = TRUE,
                                  half.life = TRUE, aucinf.obs = TRUE
                              ))
        nca_result <- pk.nca(data_obj)
        res <- as.data.frame(nca_result$result)

        get_param <- function(pname) {
            val <- res$PPORRES[res$PPTESTCD == pname]
            if (length(val) == 0) return(NA)
            return(as.numeric(val[1]))
        }

        results$pk[[sname]] <- list(
            auc_last = get_param("auclast"),
            auc_inf = get_param("aucinf.obs"),
            cmax = get_param("cmax"),
            tmax = get_param("tmax"),
            half_life = get_param("half.life"),
            route = route,
            dose = dose_val
        )

        cat(sprintf("    AUC_last=%.2f, Cmax=%.2f, t1/2=%.2f\n",
                    get_param("auclast"), get_param("cmax"),
                    ifelse(is.na(get_param("half.life")), -1, get_param("half.life"))))
    }, error = function(e) {
        cat(sprintf("    FAILED: %s\n", e$message))
        results$pk[[sname]] <<- list(error = e$message)
    })
}

# ──────────────────────────────────────────────────────────────────────
# Save all results
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Saving results ===\n")
out_path <- file.path(fixtures_dir, "r_results.json")
write_json(results, out_path, digits = 17, auto_unbox = TRUE, pretty = TRUE)
cat("Saved to:", out_path, "\n")
cat("Done.\n")
