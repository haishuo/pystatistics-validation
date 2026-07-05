#!/usr/bin/env Rscript
# montecarlo validation — R reference dispatcher (boot / boot.ci / permutation).
#
# One job: given a JSON job {func, data, params}, compute the R reference for one
# Monte-Carlo task and print the result as JSON to stdout. The Python driver dumps
# the EXACT fp64 values it feeds pystatistics into `data` (n x p, row-major), so
# both engines operate on identical numbers (R17 shared-input discipline).
#
# Stochastic-method contract (see reports/montecarlo-*.md):
#   - "boot" runs a GENUINE boot::boot (R's own RNG) and returns t0, the replicate
#     vector t, the 1-based resample index matrix (boot.array), boot.ci for every
#     type, and the BCa ingredients z0 / a(reg) / a(jack). The Python driver feeds
#     the SAME indices to pystatistics -> identical t -> the TIGHT tier isolates the
#     statistic + CI arithmetic from cross-language RNG divergence.
#   - "perm_exact" enumerates ALL C(n, n1) group assignments for a two-sample
#     mean-difference statistic and returns the DETERMINISTIC exact p-values.
#   - "perm_mc" runs an independent-RNG Monte-Carlo permutation for the
#     statistical-equivalence tier.
#
# Usage:  Rscript r_reference.R job.json  > out.json

suppressMessages({
  library(jsonlite)
  library(boot)
})

args <- commandArgs(trailingOnly = TRUE)
job <- fromJSON(args[[1]], simplifyVector = TRUE)
f <- job$func
p <- job$params

# data arrives as an (n x p) matrix (row-major list-of-lists) or a flat vector.
to_matrix <- function(d) {
  if (is.null(dim(d))) matrix(as.numeric(d), ncol = 1) else as.matrix(d)
}
data <- to_matrix(job$data)

# ---- statistic registry (must mirror the pystatistics driver exactly) -------
# Each takes (data-matrix, index-vector) and returns a scalar (boot stype="i").
STAT <- list(
  corr     = function(d, i) cor(d[i, 1], d[i, 2]),
  ratio    = function(d, i) sum(d[i, 2]) / sum(d[i, 1]),   # city: x over u
  mean     = function(d, i) mean(d[i, 1]),
  median   = function(d, i) median(d[i, 1]),
  variance = function(d, i) var(d[i, 1])
)

out <- switch(f,

  # ---- genuine boot + boot.ci; export shared resample indices ---------------
  "boot" = {
    stat <- STAT[[p$statistic]]
    set.seed(p$seed)
    R <- as.integer(p$R)
    b <- boot(data, stat, R = R)
    idx <- boot.array(b, indices = TRUE)          # R x n, 1-based
    fa <- boot.array(b)                           # R x n resample frequencies
    res <- list(
      t0 = as.numeric(b$t0),
      t = as.numeric(b$t),
      idx = as.integer(t(idx)),                   # row-major flatten
      freq = as.integer(t(fa)),                   # row-major R x n frequencies
      n = ncol(idx), R = R,
      bias = as.numeric(mean(b$t) - b$t0),
      se = as.numeric(sd(b$t))
    )
    # boot.ci for all non-studentized types (conf from params)
    conf <- if (is.null(p$conf)) 0.95 else p$conf
    ci <- boot.ci(b, conf = conf, type = c("norm", "basic", "perc", "bca"))
    res$ci_normal <- as.numeric(ci$normal[2:3])
    res$ci_basic  <- as.numeric(ci$basic[4:5])
    res$ci_perc   <- as.numeric(ci$percent[4:5])
    res$ci_bca    <- as.numeric(ci$bca[4:5])
    # BCa ingredients (deterministic given the replicates + data)
    L_reg  <- empinf(b, index = 1)
    L_jack <- empinf(data = data, statistic = stat, stype = "i",
                     type = "jack", index = 1)
    res$z0     <- as.numeric(qnorm(sum(b$t < b$t0) / R))
    res$a_reg  <- as.numeric(sum(L_reg^3)  / (6 * sum(L_reg^2)^1.5))
    res$a_jack <- as.numeric(sum(L_jack^3) / (6 * sum(L_jack^2)^1.5))
    res
  },

  # ---- studentized (bootstrap-t) CI for the mean ----------------------------
  # A 2-column statistic (mean, var-of-mean) so boot.ci type='stud' can form the
  # pivot z* = (t*-t0)/sqrt(var*). Export the shared indices + both columns so
  # pystatistics computes its studentized CI on the identical replicates.
  "boot_stud" = {
    stat2 <- function(d, i) {
      x <- d[i, 1]; m <- mean(x); c(m, var(x) / length(x))
    }
    set.seed(p$seed); R <- as.integer(p$R)
    b <- boot(data, stat2, R = R)
    idx <- boot.array(b, indices = TRUE)
    conf <- if (is.null(p$conf)) 0.95 else p$conf
    ci <- boot.ci(b, conf = conf, type = "stud", var.t0 = b$t0[2],
                  index = c(1, 2))
    list(
      t0 = as.numeric(b$t0[1]), var_t0 = as.numeric(b$t0[2]),
      t = as.numeric(b$t[, 1]), var_t = as.numeric(b$t[, 2]),
      idx = as.integer(t(idx)), n = ncol(idx), R = R,
      ci_stud = as.numeric(ci$student[4:5])
    )
  },

  # ---- exact permutation enumeration (deterministic) ------------------------
  # data col 1 = value, col 2 = group (two distinct codes). Statistic = the
  # difference in group means. Enumerates all C(n, n1) splits.
  "perm_exact" = {
    val <- data[, 1]; grp <- data[, 2]
    codes <- sort(unique(grp))
    stopifnot(length(codes) == 2)
    n <- length(val); n1 <- sum(grp == codes[1])
    obs <- mean(val[grp == codes[1]]) - mean(val[grp == codes[2]])
    combos <- combn(n, n1)
    total <- ncol(combos)
    tot_sum <- sum(val); s2n <- n - n1
    stats <- apply(combos, 2, function(ix) {
      s1 <- sum(val[ix]); s1 / n1 - (tot_sum - s1) / s2n
    })
    pg <- mean(stats >= obs - 1e-12); pl <- mean(stats <= obs + 1e-12)
    list(
      observed = obs, n_perms = total,
      # 2*min-tail two-sided (matches pystatistics >= 4.6.8)
      p_two_sided = min(1, 2 * min(pg, pl)),
      p_greater   = pg, p_less = pl
    )
  },

  # ---- independent-RNG Monte-Carlo permutation (equivalence tier) -----------
  "perm_mc" = {
    val <- data[, 1]; grp <- data[, 2]
    codes <- sort(unique(grp))
    n <- length(val); n1 <- sum(grp == codes[1])
    obs <- mean(val[grp == codes[1]]) - mean(val[grp == codes[2]])
    set.seed(p$seed); R <- as.integer(p$R)
    tot_sum <- sum(val); s2n <- n - n1
    stats <- replicate(R, {
      ix <- sample.int(n, n1); s1 <- sum(val[ix])
      s1 / n1 - (tot_sum - s1) / s2n
    })
    # Phipson-Smyth (count+1)/(R+1); 2*min-tail two-sided (matches pystatistics)
    pg <- (sum(stats >= obs) + 1) / (R + 1)
    pl <- (sum(stats <= obs) + 1) / (R + 1)
    list(
      observed = obs, R = R,
      p_two_sided = min(1, 2 * min(pg, pl)),
      p_greater = pg, p_less = pl
    )
  },

  # ---- timing: elapsed seconds for a boot run (excludes R startup) ----------
  "boot_bench" = {
    stat <- STAT[[p$statistic]]
    R <- as.integer(p$R); reps <- if (is.null(p$reps)) 3L else as.integer(p$reps)
    best <- Inf
    for (k in seq_len(reps)) {
      set.seed(p$seed + k)
      el <- system.time(boot(data, stat, R = R))[["elapsed"]]
      best <- min(best, el)
    }
    list(elapsed = best, R = R, statistic = p$statistic)
  },

  # ---- permutation timing ---------------------------------------------------
  "perm_bench" = {
    val <- data[, 1]; grp <- data[, 2]; codes <- sort(unique(grp))
    n <- length(val); n1 <- sum(grp == codes[1]); tot <- sum(val); s2n <- n - n1
    R <- as.integer(p$R); reps <- if (is.null(p$reps)) 3L else as.integer(p$reps)
    best <- Inf
    for (k in seq_len(reps)) {
      set.seed(p$seed + k)
      el <- system.time(replicate(R, {
        ix <- sample.int(n, n1); s1 <- sum(val[ix]); s1 / n1 - (tot - s1) / s2n
      }))[["elapsed"]]
      best <- min(best, el)
    }
    list(elapsed = best, R = R)
  },

  stop(paste("unknown func:", f))
)

cat(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"))
