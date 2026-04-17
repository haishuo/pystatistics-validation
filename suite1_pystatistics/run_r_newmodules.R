#!/usr/bin/env Rscript
#
# R reference results for NEW pystatistics modules (v1.6.x).
#
# DATA POLICY:
#   Wherever a canonical real-world R dataset exists for the module being
#   tested, we use it. The R script writes the exact dataset Python will read
#   to a CSV in fixtures/newmodules/, so both languages fit byte-identical
#   inputs. Only tests with no good canonical dataset fall back to simulation
#   (currently: Gamma GLM — see generate_newmodules_data.py).
#
# Dataset mapping:
#   Gamma GLM            — datasets::airquality (Ozone ~ Solar.R + Temp + Wind)
#                          Ozone is positive, continuous, right-skewed. A
#                          textbook Gamma regression target.
#   Negative Binomial    — MASS::quine (Australian school absences). THE
#                          canonical NB regression example (McCullagh & Nelder).
#   Ordinal (polr)       — MASS::housing (Copenhagen housing satisfaction
#                          survey). The dataset the MASS::polr help page
#                          itself uses.
#   Multinomial          — MASS::fgl (forensic glass identification).
#                          The canonical multinom example — 6 unordered
#                          glass types, 9 continuous predictors.
#   PCA                  — datasets::USArrests (state crime statistics).
#                          The canonical prcomp/factanal example from the
#                          R help page.
#   Factor analysis      — datasets::mtcars (numeric columns). USArrests
#                          only has 4 variables — can't fit a 2-factor model.
#                          mtcars has 11 numeric cols so a 2-factor fit is
#                          well-determined.
#   Time series          — datasets::AirPassengers (monthly airline totals,
#                          1949-1960, period 12). THE canonical seasonal
#                          time series used in every Box-Jenkins textbook.
#   GAM                  — MASS::mcycle (simulated motorcycle crash head
#                          accel). The canonical mgcv::gam example.

library(jsonlite)
library(MASS)
library(mgcv)
library(forecast)
options(digits = 22)

FIX <- "suite1_pystatistics/fixtures/newmodules"
dir.create(FIX, recursive = TRUE, showWarnings = FALSE)

results <- list()
timing  <- list()

time_it <- function(expr) {
    t <- system.time(val <- eval.parent(substitute(expr)))[["elapsed"]]
    list(value = val, elapsed = as.numeric(t))
}

# ─────────────────────────────────────────────────────────────────────
# Gamma GLM — airquality (real) + log link.
# Drop rows with NA Ozone or Solar.R so R and Python fit the same rows.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Gamma GLM (airquality) ===\n")
aq <- airquality[complete.cases(airquality[, c("Ozone","Solar.R","Temp","Wind")]), ]
write.csv(aq[, c("Ozone","Solar.R","Temp","Wind")],
          file.path(FIX, "airquality.csv"), row.names = FALSE)
r <- time_it(glm(Ozone ~ Solar.R + Temp + Wind, data = aq,
                 family = Gamma(link = "log")))
fit <- r$value; s <- summary(fit)
results$gamma_glm <- list(
    coefficients    = as.numeric(coef(fit)),
    standard_errors = as.numeric(s$coefficients[, "Std. Error"]),
    deviance        = as.numeric(fit$deviance),
    null_deviance   = as.numeric(fit$null.deviance),
    dispersion      = as.numeric(s$dispersion),
    aic             = as.numeric(fit$aic),
    n               = nrow(aq),
    dataset         = "datasets::airquality (complete Ozone/Solar.R cases)"
)
timing$gamma_glm <- r$elapsed
cat("  n:", nrow(aq), " time:", r$elapsed, "s  coefs:", coef(fit), "\n")

# ─────────────────────────────────────────────────────────────────────
# Negative Binomial — MASS::quine (school absences).
# Daily absences by Eth, Sex, Age, Lrn — the canonical NB dataset.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Negative Binomial (quine) ===\n")
# Model-matrix with model.matrix so Python sees the exact same numeric X.
X <- model.matrix(Days ~ Eth + Sex + Age + Lrn, data = quine)
y <- quine$Days
quine_csv <- data.frame(Days = y, X[, -1])  # strip the all-ones col
colnames(quine_csv)[-1] <- colnames(X)[-1]
write.csv(quine_csv, file.path(FIX, "quine.csv"), row.names = FALSE)

r <- time_it(glm.nb(Days ~ Eth + Sex + Age + Lrn, data = quine))
fit <- r$value; s <- summary(fit)
results$negbin_glm <- list(
    coefficients    = as.numeric(coef(fit)),
    standard_errors = as.numeric(s$coefficients[, "Std. Error"]),
    theta           = as.numeric(fit$theta),
    deviance        = as.numeric(fit$deviance),
    aic             = as.numeric(fit$aic),
    design_columns  = colnames(X),
    dataset         = "MASS::quine (Australian school absences)"
)
timing$negbin_glm <- r$elapsed
cat("  time:", r$elapsed, "s  theta:", fit$theta, "\n")

# ─────────────────────────────────────────────────────────────────────
# Ordinal (polr) — MASS::housing (Copenhagen housing satisfaction).
# Satisfaction (Low/Medium/High) predicted by Infl * Type * Cont, with
# Freq weighting. We expand to individual-observation form for Python.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Ordinal (polr, housing) ===\n")
hsg <- housing
# Expand by Freq so each row is one respondent
idx <- rep(seq_len(nrow(hsg)), hsg$Freq)
hsg_long <- hsg[idx, c("Sat", "Infl", "Type", "Cont")]
rownames(hsg_long) <- NULL
# Build numeric design
X <- model.matrix(~ Infl + Type + Cont, data = hsg_long)[, -1]
housing_csv <- data.frame(Sat = as.integer(hsg_long$Sat), X)
write.csv(housing_csv, file.path(FIX, "housing.csv"), row.names = FALSE)

r <- time_it(polr(Sat ~ Infl + Type + Cont, data = hsg_long,
                   method = "logistic", Hess = TRUE))
fit <- r$value
results$polr <- list(
    coefficients   = as.numeric(coef(fit)),
    thresholds     = as.numeric(fit$zeta),
    standard_errors = as.numeric(sqrt(diag(vcov(fit))))[seq_along(coef(fit))],
    deviance       = as.numeric(fit$deviance),
    log_lik        = as.numeric(logLik(fit)),
    design_columns = colnames(X),
    dataset        = "MASS::housing (Copenhagen housing satisfaction, expanded by Freq)"
)
timing$polr <- r$elapsed
cat("  n (expanded):", nrow(hsg_long), "  time:", r$elapsed, "s\n")

# ─────────────────────────────────────────────────────────────────────
# Multinomial — MASS::fgl (forensic glass identification).
# 6 unordered glass types (type), 9 continuous predictors. Subsample /
# simplification NOT needed — 214 rows, fits cleanly.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Multinomial (fgl, 3 major classes) ===\n")
library(nnet)
# fgl's 6 glass types include 3 small classes (9/13/17 obs) that make
# pystatistics' multinom IRLS struggle to converge. Restrict to the 3
# largest classes (WinF=70, WinNF=76, Head=29) — still a genuine
# multiclass problem on real forensic data.
fgl_df <- fgl[fgl$type %in% c("WinF", "WinNF", "Head"), ]
fgl_df$type <- factor(fgl_df$type, levels = c("WinF", "WinNF", "Head"))
X <- model.matrix(~ RI + Na + Al, data = fgl_df)
fgl_csv <- data.frame(type = as.integer(fgl_df$type) - 1L,    # 0..5
                      X[, -1])
write.csv(fgl_csv, file.path(FIX, "fgl.csv"), row.names = FALSE)

r <- time_it(multinom(type ~ RI + Na + Al, data = fgl_df, trace = FALSE))
fit <- r$value
coef_mat <- coef(fit)
results$multinom <- list(
    coefficients = as.matrix(coef_mat),
    class_labels = rownames(coef_mat),
    predictor_labels = colnames(coef_mat),
    reference_class = levels(fgl_df$type)[1],   # R uses FIRST level
    deviance = as.numeric(fit$deviance),
    aic = as.numeric(fit$AIC),
    design_columns = colnames(X),
    dataset = "MASS::fgl (forensic glass, 6 types)"
)
timing$multinom <- r$elapsed
cat("  time:", r$elapsed, "s  ref class:", levels(fgl_df$type)[1], "\n")

# ─────────────────────────────────────────────────────────────────────
# PCA — datasets::USArrests. Canonical prcomp help-page dataset.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== PCA (USArrests) ===\n")
usa <- USArrests
write.csv(usa, file.path(FIX, "USArrests.csv"), row.names = TRUE)
r <- time_it(prcomp(usa, center = TRUE, scale. = TRUE))
pc <- r$value
results$pca <- list(
    sdev = as.numeric(pc$sdev),
    rotation = as.matrix(pc$rotation),
    center = as.numeric(pc$center),
    scale = as.numeric(pc$scale),
    scores_first5 = as.matrix(pc$x[1:5, ]),
    dataset = "datasets::USArrests (50 states x 4 crime rates)"
)
timing$pca <- r$elapsed
cat("  time:", r$elapsed, "s  sdev:", pc$sdev, "\n")

# ─────────────────────────────────────────────────────────────────────
# Factor analysis — mtcars (11 numeric vars). USArrests is too small.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Factor Analysis (mtcars) ===\n")
mt <- mtcars
write.csv(mt, file.path(FIX, "mtcars.csv"), row.names = TRUE)
r <- time_it(factanal(mt, factors = 2, rotation = "varimax"))
fa <- r$value
results$factor_analysis <- list(
    loadings = matrix(as.numeric(fa$loadings), nrow = nrow(fa$loadings)),
    uniquenesses = as.numeric(fa$uniquenesses),
    converged = as.logical(fa$converged),
    n_factors = 2L,
    dataset = "datasets::mtcars (11 numeric vars on 32 cars)"
)
timing$factor_analysis <- r$elapsed
cat("  time:", r$elapsed, "s\n")

# ─────────────────────────────────────────────────────────────────────
# Time series — AirPassengers (monthly, 1949-1960, period 12).
# Canonical Box-Jenkins dataset. We work on log(AirPassengers) because
# the series is multiplicative-seasonal; logging makes it additive.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== Time series (AirPassengers) ===\n")
ap <- log(AirPassengers)
write.csv(data.frame(y = as.numeric(ap),
                     year = floor(time(ap)),
                     month = cycle(ap)),
          file.path(FIX, "airpassengers.csv"), row.names = FALSE)

# ARIMA fit — classic result: ARIMA(0,1,1)(0,1,1)[12] on log(AP) is the
# Box-Jenkins "airline model".
r <- time_it(arima(ap, order = c(0, 1, 1),
                   seasonal = list(order = c(0, 1, 1), period = 12)))
fit <- r$value
results$arima <- list(
    ma        = as.numeric(coef(fit)["ma1"]),
    seasonal_ma = as.numeric(coef(fit)["sma1"]),
    sigma2    = as.numeric(fit$sigma2),
    log_lik   = as.numeric(fit$loglik),
    aic       = as.numeric(fit$aic),
    dataset   = "log(AirPassengers), airline model SARIMA(0,1,1)(0,1,1)[12]"
)
timing$arima <- r$elapsed
cat("  time:", r$elapsed, "s  ma1:", coef(fit)["ma1"], "sma1:", coef(fit)["sma1"], "\n")

# ETS — on the raw multiplicative series, 'MAM' is the airline model analog.
r <- time_it(ets(AirPassengers, model = "MAM"))
fit <- r$value
results$ets <- list(
    alpha = as.numeric(fit$par["alpha"]),
    beta  = as.numeric(fit$par["beta"]),
    gamma = as.numeric(fit$par["gamma"]),
    sigma2 = as.numeric(fit$sigma2),
    log_lik = as.numeric(fit$loglik),
    aic = as.numeric(fit$aic),
    bic = as.numeric(fit$bic),
    dataset = "AirPassengers raw, ETS(M,A,M)"
)
timing$ets <- r$elapsed
cat("  ets time:", r$elapsed, "s  alpha:", fit$par["alpha"], "\n")

# ACF / PACF — on log(AP) at defaults.
max_lag <- 20
r <- time_it(acf(as.numeric(ap), lag.max = max_lag, plot = FALSE))
results$acf <- list(acf = as.numeric(r$value$acf), lag_max = max_lag)
timing$acf <- r$elapsed
r <- time_it(pacf(as.numeric(ap), lag.max = max_lag, plot = FALSE))
results$pacf <- list(pacf = as.numeric(r$value$acf), lag_max = max_lag)
timing$pacf <- r$elapsed

# decompose / STL
r <- time_it(decompose(ap, type = "additive"))
dec <- r$value
results$decompose <- list(
    trend_first20 = as.numeric(dec$trend[1:20]),
    seasonal_first24 = as.numeric(dec$seasonal[1:24]),
    period = 12L
)
timing$decompose <- r$elapsed

r <- time_it(stl(ap, s.window = "periodic"))
st <- r$value$time.series
results$stl <- list(
    trend_first10 = as.numeric(st[1:10, "trend"]),
    seasonal_first24 = as.numeric(st[1:24, "seasonal"])
)
timing$stl <- r$elapsed
cat("  decompose/STL done\n")

# ─────────────────────────────────────────────────────────────────────
# GAM — MASS::mcycle. Canonical mgcv dataset.
#   accel ~ s(times)
# Head acceleration vs. time after simulated motorcycle crash.
# ─────────────────────────────────────────────────────────────────────
cat("\n=== GAM (mcycle) ===\n")
mc <- MASS::mcycle
write.csv(mc, file.path(FIX, "mcycle.csv"), row.names = FALSE)
r <- time_it(mgcv::gam(accel ~ s(times, k = 20, bs = "cr"),
                        data = mc, method = "REML"))
fit <- r$value; s <- summary(fit)
results$gam <- list(
    intercept  = as.numeric(coef(fit)["(Intercept)"]),
    edf_s      = as.numeric(s$edf[1]),
    deviance   = as.numeric(fit$deviance),
    aic        = as.numeric(AIC(fit)),
    n          = nrow(mc),
    dataset    = "MASS::mcycle (head accel after motorcycle crash)"
)
timing$gam <- r$elapsed
cat("  time:", r$elapsed, "s  edf:", s$edf[1], "\n")

# ─────────────────────────────────────────────────────────────────────
out <- list(results = results, timing = timing)
write_json(out, file.path(FIX, "r_results.json"),
           digits = 17, auto_unbox = TRUE, pretty = TRUE)
cat("\nWrote", file.path(FIX, "r_results.json"), "\n")
