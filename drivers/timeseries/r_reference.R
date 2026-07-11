#!/usr/bin/env Rscript
# timeseries validation — R reference dispatcher.
#
# One job: given a JSON job {func, data, params}, compute the R reference for one
# timeseries function and print the result as JSON to stdout. The Python driver
# dumps the EXACT fp64 values it feeds pystatistics into `data`, so both engines
# operate on identical numbers (float32-store shared-input discipline, R17).
#
# References matched per function:
#   acf/pacf -> stats::acf / stats::pacf   (lag-0 convention, demean)
#   diff     -> base::diff
#   ndiffs   -> forecast::ndiffs           (test matched explicitly)
#   adf      -> tseries::adf.test          (lag order, statistic, interp p-value)
#   kpss     -> tseries::kpss.test         (level/trend, interp p-value)
#
# Usage:  Rscript r_reference.R job.json  > out.json

suppressMessages({
  library(jsonlite)
  library(forecast)
  library(tseries)
})

args <- commandArgs(trailingOnly = TRUE)
job <- fromJSON(args[[1]], simplifyVector = TRUE)
f <- job$func
p <- job$params
x <- as.numeric(job$data)                # NA <- JSON null (jsonlite)
period <- if (!is.null(job$period)) as.integer(job$period) else 1L
xt <- ts(x, frequency = period)

out <- switch(f,

  "acf" = {
    r <- acf(x, lag.max = p$max_lag, demean = p$demean, plot = FALSE,
             na.action = na.pass)
    list(acf = as.numeric(r$acf), lags = as.integer(r$lag),
         n_used = r$n.used)
  },

  "pacf" = {
    r <- pacf(x, lag.max = p$max_lag, plot = FALSE, na.action = na.pass)
    list(pacf = as.numeric(r$acf), lags = as.integer(r$lag),
         n_used = r$n.used)
  },

  "diff" = {
    r <- diff(x, lag = p$lag, differences = p$differences)
    list(diff = as.numeric(r), n = length(r))
  },

  "ndiffs" = {
    r <- ndiffs(x, test = p$test, alpha = p$alpha, max.d = p$max_d)
    list(ndiffs = as.integer(r))
  },

  "adf" = {
    # tseries::adf.test uses a fixed default lag = trunc((n-1)^(1/3)); allow
    # an explicit k to match pystatistics' lag order exactly.
    r <- if (is.null(p$k)) adf.test(x, alternative = "stationary")
         else adf.test(x, alternative = "stationary", k = p$k)
    list(statistic = as.numeric(r$statistic), p_value = as.numeric(r$p.value),
         lag = as.numeric(r$parameter))
  },

  "kpss" = {
    r <- kpss.test(x, null = p$null)
    list(statistic = as.numeric(r$statistic), p_value = as.numeric(r$p.value),
         lag = as.numeric(r$parameter))
  },

  "stl" = {
    sw <- if (is.null(p$s_window)) "periodic" else p$s_window
    sdeg <- if (is.null(p$s_degree)) 0 else p$s_degree
    tw <- if (is.null(p$t_window)) NULL else p$t_window
    tdeg <- if (is.null(p$t_degree)) 1 else p$t_degree
    lw <- if (is.null(p$l_window)) NULL else p$l_window
    ldeg <- if (is.null(p$l_degree)) tdeg else p$l_degree
    extra <- list()
    if (!is.null(p$s_jump)) extra$s.jump <- p$s_jump
    if (!is.null(p$t_jump)) extra$t.jump <- p$t_jump
    if (!is.null(p$l_jump)) extra$l.jump <- p$l_jump
    r <- do.call(stl, c(list(x = xt, s.window = sw, s.degree = sdeg,
             t.window = tw, t.degree = tdeg, l.window = lw, l.degree = ldeg,
             inner = p$n_inner, outer = p$n_outer, robust = p$robust), extra))
    ts <- r$time.series
    list(seasonal = as.numeric(ts[, "seasonal"]),
         trend = as.numeric(ts[, "trend"]),
         remainder = as.numeric(ts[, "remainder"]))
  },

  "decompose" = {
    r <- decompose(xt, type = p$type)
    list(seasonal = as.numeric(r$seasonal),
         trend = as.numeric(r$trend),
         random = as.numeric(r$random))
  },

  "ets" = {
    # Match the model string exactly; no auto-selection (Z) here.
    r <- ets(xt, model = p$model, damped = p$damped,
             opt.crit = "lik", ic = "aicc")
    par <- r$par
    getp <- function(nm) if (nm %in% names(par)) as.numeric(par[[nm]]) else NA
    list(loglik = as.numeric(r$loglik), aic = as.numeric(r$aic),
         aicc = as.numeric(r$aicc), bic = as.numeric(r$bic),
         alpha = getp("alpha"), beta = getp("beta"),
         gamma = getp("gamma"), phi = getp("phi"),
         sigma2 = as.numeric(r$sigma2),
         fitted = as.numeric(fitted(r)))
  },

  "arima" = {
    ord <- c(p$order[1], p$order[2], p$order[3])
    seas <- if (is.null(p$seasonal)) c(0, 0, 0)
            else c(p$seasonal[1], p$seasonal[2], p$seasonal[3])
    r <- arima(xt, order = ord,
               seasonal = list(order = seas, period = period),
               include.mean = p$include_mean, method = p$method)
    list(loglik = as.numeric(r$loglik), aic = as.numeric(r$aic),
         sigma2 = as.numeric(r$sigma2),
         coef = as.numeric(r$coef), coef_names = names(r$coef))
  },

  "forecast_arima" = {
    ord <- c(p$order[1], p$order[2], p$order[3])
    seas <- if (is.null(p$seasonal)) c(0, 0, 0)
            else c(p$seasonal[1], p$seasonal[2], p$seasonal[3])
    fit <- arima(xt, order = ord,
                 seasonal = list(order = seas, period = period),
                 include.mean = p$include_mean, method = p$method)
    pr <- predict(fit, n.ahead = p$h)
    list(mean = as.numeric(pr$pred), se = as.numeric(pr$se))
  },

  # Regression with ARIMA errors (VA-4). xreg travels as a JSON list-of-rows
  # (simplifyVector -> an n x k matrix); include_drift appends a 1..n trend
  # column (as pystatistics does); fixed is a nan->null vector (NA = estimate).
  # stats::arima is the exact-ML reference (MLE sigma2; predict.Arima SE uses it).
  "arima_xreg" = {
    ord <- as.integer(c(p$order[1], p$order[2], p$order[3]))
    seas <- if (is.null(p$seasonal)) NULL else as.integer(p$seasonal[1:3])
    xreg <- if (is.null(p$xreg)) NULL else {
      m <- p$xreg
      if (is.null(dim(m))) matrix(as.numeric(m), ncol = 1)
      else matrix(as.numeric(m), nrow = nrow(m))
    }
    if (isTRUE(p$include_drift)) {
      tt <- matrix(seq_len(length(x)), ncol = 1)
      xreg <- if (is.null(xreg)) tt else cbind(tt, xreg)
    }
    fx <- if (is.null(p$fixed)) NULL else as.numeric(p$fixed)  # NA <- null
    args <- list(x = xt, order = ord, xreg = xreg, method = p$method)
    if (!is.null(seas)) args$seasonal <- list(order = seas, period = period)
    if (!is.null(fx)) args$fixed <- fx
    if (!is.null(p$include_mean)) args$include.mean <- p$include_mean
    r <- do.call(stats::arima, args)
    se <- sqrt(diag(r$var.coef))
    list(loglik = as.numeric(r$loglik), aic = as.numeric(r$aic),
         sigma2 = as.numeric(r$sigma2), coef = as.numeric(r$coef),
         coef_names = names(r$coef), se = as.numeric(se))
  },

  "forecast_arima_xreg" = {
    ord <- as.integer(c(p$order[1], p$order[2], p$order[3]))
    seas <- if (is.null(p$seasonal)) NULL else as.integer(p$seasonal[1:3])
    xreg <- if (is.null(p$xreg)) NULL else {
      m <- p$xreg
      if (is.null(dim(m))) matrix(as.numeric(m), ncol = 1)
      else matrix(as.numeric(m), nrow = nrow(m))
    }
    if (isTRUE(p$include_drift)) {
      tt <- matrix(seq_len(length(x)), ncol = 1)
      xreg <- if (is.null(xreg)) tt else cbind(tt, xreg)
    }
    args <- list(x = xt, order = ord, xreg = xreg, method = p$method)
    if (!is.null(seas)) args$seasonal <- list(order = seas, period = period)
    if (!is.null(p$include_mean)) args$include.mean <- p$include_mean
    fit <- do.call(stats::arima, args)
    h <- p$h
    nx <- if (is.null(p$newxreg)) NULL else {
      m <- p$newxreg
      if (is.null(dim(m))) matrix(as.numeric(m), ncol = 1)
      else matrix(as.numeric(m), nrow = nrow(m))
    }
    if (isTRUE(p$include_drift)) {
      fut <- matrix((length(x) + 1):(length(x) + h), ncol = 1)
      nx <- if (is.null(nx)) fut else cbind(fut, nx)
    }
    pr <- predict(fit, n.ahead = h, newxreg = nx)
    list(mean = as.numeric(pr$pred), se = as.numeric(pr$se))
  },

  "forecast_ets" = {
    fit <- ets(xt, model = p$model, damped = p$damped,
               opt.crit = "lik", ic = "aicc")
    fc <- forecast(fit, h = p$h, level = 95)
    list(mean = as.numeric(fc$mean),
         lower = as.numeric(fc$lower), upper = as.numeric(fc$upper))
  },

  "auto" = {
    r <- auto.arima(xt, ic = p$ic, stepwise = p$stepwise,
                    max.p = p$max_p, max.q = p$max_q, max.d = p$max_d,
                    seasonal = (period > 1))
    ord <- arimaorder(r)
    list(order = as.integer(ord[c("p", "d", "q")]),
         seasonal = if ("P" %in% names(ord))
                      as.integer(ord[c("P", "D", "Q")]) else NULL,
         loglik = as.numeric(r$loglik), aic = as.numeric(r$aic),
         aicc = as.numeric(r$aicc),
         coef = as.numeric(r$coef), coef_names = names(r$coef))
  },

  stop(paste("unknown func:", f))
)

cat(toJSON(out, auto_unbox = TRUE, digits = 15, na = "null"))
