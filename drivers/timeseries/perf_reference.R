#!/usr/bin/env Rscript
# timeseries validation — R timing dispatcher.
# Given {func, data, params, reps}, time the R reference fit `reps` times and
# print the MIN elapsed seconds (JSON) — min is the least noisy wall-clock. The
# subprocess spawn is one-time per call, excluded from the timed region.

suppressMessages({library(jsonlite); library(forecast); library(tseries)})
args <- commandArgs(trailingOnly = TRUE)
job <- fromJSON(args[[1]], simplifyVector = TRUE)
f <- job$func; p <- job$params; x <- as.numeric(job$data)
period <- if (!is.null(job$period)) as.integer(job$period) else 1L
reps <- if (!is.null(job$reps)) as.integer(job$reps) else 5L
xt <- ts(x, frequency = period)

once <- switch(f,
  "acf"   = function() acf(x, lag.max = p$max_lag, plot = FALSE),
  "arima" = function() arima(xt, order = c(p$order[1], p$order[2], p$order[3]),
                             include.mean = TRUE, method = p$method),
  "ets"   = function() ets(xt, model = p$model, damped = FALSE),
  "stl"   = function() stl(xt, s.window = "periodic"),
  "adf"   = function() suppressWarnings(adf.test(x)),
  stop(paste("unknown perf func:", f)))

# Time the whole reps loop in one block and amortize — robust for sub-ms fits
# that individual system.time() calls round to 0.
invisible(once())  # warm up outside the timed region (suppress auto-print)
total <- system.time(for (i in seq_len(reps)) invisible(once()))["elapsed"]
cat(toJSON(list(elapsed = as.numeric(total) / reps, reps = reps),
           auto_unbox = TRUE))
