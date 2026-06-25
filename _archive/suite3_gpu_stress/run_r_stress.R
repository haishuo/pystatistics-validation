#!/usr/bin/env Rscript
#
# R Reference for GPU Stress Tests (TCGA BRCA)
#
# WARNING: This script takes ~1-2.5 hours to complete.
# It generates reference results for GPU comparison.
#
# Run from /mnt/projects/test:
#   Rscript suite3_gpu_stress/run_r_stress.R
#

library(jsonlite)
library(pROC)
library(boot)

options(digits = 22)

fixtures_dir <- "suite3_gpu_stress/fixtures"
results <- list()
timings <- list()

# ──────────────────────────────────────────────────────────────────────
# Batch AUC: compute AUC for all genes
# ──────────────────────────────────────────────────────────────────────
cat("=== Batch AUC (all genes) ===\n")

full_data <- read.csv(file.path(fixtures_dir, "tcga_full_for_r.csv"))
labels <- full_data$label
gene_cols <- setdiff(names(full_data), "label")
n_genes <- length(gene_cols)
cat(sprintf("  %d samples, %d genes\n", nrow(full_data), n_genes))

batch_aucs <- numeric(n_genes)
batch_ses <- numeric(n_genes)
names(batch_aucs) <- gene_cols
names(batch_ses) <- gene_cols

cat("  Computing AUC for each gene...\n")
t_start <- proc.time()["elapsed"]

for (i in seq_along(gene_cols)) {
    col <- gene_cols[i]
    tryCatch({
        r <- roc(labels, full_data[[col]], direction = "<", quiet = TRUE)
        batch_aucs[i] <- as.numeric(r$auc)
        batch_ses[i] <- as.numeric(sqrt(var(r)))
    }, error = function(e) {
        batch_aucs[i] <<- NA
        batch_ses[i] <<- NA
    })
    if (i %% 1000 == 0) {
        elapsed <- proc.time()["elapsed"] - t_start
        eta <- elapsed / i * (n_genes - i)
        cat(sprintf("    %d / %d (%.0fs elapsed, ~%.0fs remaining)\n",
                    i, n_genes, elapsed, eta))
    }
}

t_batch_auc <- proc.time()["elapsed"] - t_start
timings$batch_auc <- t_batch_auc
cat(sprintf("  Done: %.1f seconds\n", t_batch_auc))

results$batch_auc <- list(
    auc = as.numeric(batch_aucs),
    se = as.numeric(batch_ses),
    gene_names = gene_cols,
    n_computed = sum(!is.na(batch_aucs))
)

# Top 50 genes by AUC (for ranking comparison)
auc_dir <- pmax(batch_aucs, 1 - batch_aucs)
top50_idx <- order(auc_dir, decreasing = TRUE)[1:50]
results$batch_auc_top50 <- list(
    genes = gene_cols[top50_idx],
    auc = as.numeric(batch_aucs[top50_idx])
)

# ──────────────────────────────────────────────────────────────────────
# GLM binomial: label ~ top 500 genes
# ──────────────────────────────────────────────────────────────────────
cat("\n=== GLM Binomial (top 500 genes) ===\n")

top500_data <- read.csv(file.path(fixtures_dir, "tcga_top500.csv"))
top500_labels <- top500_data$label
top500_genes <- setdiff(names(top500_data), "label")

# Use first 500 gene columns (or all if fewer)
n_use <- min(500, length(top500_genes))
X <- as.matrix(top500_data[, top500_genes[1:n_use]])

cat(sprintf("  Fitting GLM: %d samples x %d predictors\n", nrow(X), ncol(X)))
t_start <- proc.time()["elapsed"]

tryCatch({
    glm_fit <- glm(top500_labels ~ X, family = binomial)
    t_glm <- proc.time()["elapsed"] - t_start
    timings$glm <- t_glm

    results$glm <- list(
        coefficients = as.numeric(coef(glm_fit)),
        deviance = glm_fit$deviance,
        null_deviance = glm_fit$null.deviance,
        aic = glm_fit$aic,
        converged = glm_fit$converged
    )
    cat(sprintf("  Done: %.1f seconds, deviance=%.2f\n", t_glm, glm_fit$deviance))
}, error = function(e) {
    cat(sprintf("  GLM FAILED: %s\n", e$message))
    # Try with fewer genes
    n_use2 <- min(100, n_use)
    X2 <- X[, 1:n_use2]
    glm_fit2 <- glm(top500_labels ~ X2, family = binomial)
    t_glm <- proc.time()["elapsed"] - t_start
    timings$glm <<- t_glm
    results$glm <<- list(
        coefficients = as.numeric(coef(glm_fit2)),
        deviance = glm_fit2$deviance,
        n_predictors = n_use2,
        note = "reduced to fewer predictors due to convergence"
    )
})

# ──────────────────────────────────────────────────────────────────────
# Bootstrap AUC for top gene
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Bootstrap AUC (top gene) ===\n")

# Find the top gene
top_gene_idx <- which.max(auc_dir)
top_gene_name <- gene_cols[top_gene_idx]
top_gene_vals <- full_data[[top_gene_name]]

cat(sprintf("  Top gene: %s (AUC=%.4f)\n", top_gene_name, auc_dir[top_gene_idx]))

auc_stat <- function(data, indices) {
    d <- data[indices, ]
    tryCatch({
        r <- roc(d$label, d$predictor, direction = "<", quiet = TRUE)
        return(as.numeric(r$auc))
    }, error = function(e) return(NA))
}

boot_df <- data.frame(label = labels, predictor = top_gene_vals)
set.seed(42)

cat("  Running 5000 bootstrap resamples...\n")
t_start <- proc.time()["elapsed"]
boot_result <- boot(boot_df, auc_stat, R = 5000)
t_boot <- proc.time()["elapsed"] - t_start
timings$bootstrap <- t_boot
cat(sprintf("  Done: %.1f seconds\n", t_boot))

boot_ci_result <- boot.ci(boot_result, type = "perc")

results$bootstrap <- list(
    t0 = as.numeric(boot_result$t0),
    se = sd(boot_result$t, na.rm = TRUE),
    ci_perc_lower = as.numeric(boot_ci_result$percent[4]),
    ci_perc_upper = as.numeric(boot_ci_result$percent[5]),
    top_gene = top_gene_name
)

# ──────────────────────────────────────────────────────────────────────
# Permutation test for top gene
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Permutation Test (top gene, 50K permutations) ===\n")

observed_diff <- mean(top_gene_vals[labels == 1]) - mean(top_gene_vals[labels == 0])

set.seed(43)
n_perm <- 50000
perm_diffs <- numeric(n_perm)

t_start <- proc.time()["elapsed"]
for (p in 1:n_perm) {
    perm_labels <- sample(labels)
    perm_diffs[p] <- mean(top_gene_vals[perm_labels == 1]) - mean(top_gene_vals[perm_labels == 0])
}
t_perm <- proc.time()["elapsed"] - t_start
timings$permutation <- t_perm

p_value <- mean(abs(perm_diffs) >= abs(observed_diff))

results$permutation <- list(
    observed = observed_diff,
    p_value = p_value,
    n_perm = n_perm,
    top_gene = top_gene_name
)
cat(sprintf("  Done: %.1f seconds, p=%.6f\n", t_perm, p_value))

# ──────────────────────────────────────────────────────────────────────
# Save all results and timings
# ──────────────────────────────────────────────────────────────────────
cat("\n=== Saving results ===\n")
write_json(results, file.path(fixtures_dir, "r_results.json"),
           digits = 17, auto_unbox = TRUE, pretty = TRUE)
write_json(timings, file.path(fixtures_dir, "r_timings.json"),
           digits = 6, auto_unbox = TRUE, pretty = TRUE)
cat("Saved to:", file.path(fixtures_dir, "r_results.json"), "\n")
cat("Saved timings to:", file.path(fixtures_dir, "r_timings.json"), "\n")
cat(sprintf("\nTotal R timings:\n"))
for (name in names(timings)) {
    cat(sprintf("  %s: %.1f seconds\n", name, timings[[name]]))
}
cat("Done.\n")
