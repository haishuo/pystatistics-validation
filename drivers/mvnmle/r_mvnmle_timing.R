#!/usr/bin/env Rscript
# Time R's mvnmle::mlest on a CSV matrix (NA = missing) and emit JSON.
#
# One job: load the data once, run `warmup` discarded fits then `reps` timed
# fits (so R's startup cost is NOT included in the timing), and write the
# estimates + elapsed-time summary as JSON.
#
# Usage: Rscript r_mvnmle_timing.R <csv_in> <json_out> <warmup> <reps>

suppressWarnings(suppressMessages({
  library(mvnmle)
  library(jsonlite)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: r_mvnmle_timing.R <csv_in> <json_out> <warmup> <reps>")
csv_in  <- args[[1]]
json_out <- args[[2]]
warmup  <- as.integer(args[[3]])
reps    <- as.integer(args[[4]])

data <- as.matrix(read.csv(csv_in, header = FALSE, na.strings = "NA"))
storage.mode(data) <- "double"

# Warmup (discarded) — keeps cold caches out of the timed reps.
if (warmup > 0) for (i in seq_len(warmup)) invisible(mlest(data))

times <- numeric(reps)
res <- NULL
for (i in seq_len(reps)) {
  el <- system.time(res <- mlest(data))[["elapsed"]]
  times[i] <- el
}

ts <- sort(times)
med <- if (reps %% 2 == 1) ts[(reps + 1) %/% 2] else mean(ts[c(reps %/% 2, reps %/% 2 + 1)])

out <- list(
  wall_median_s = med,
  wall_min_s    = ts[1],
  wall_max_s    = ts[reps],
  wall_times_s  = as.numeric(times),
  muhat         = as.numeric(res$muhat),
  sigmahat      = as.numeric(t(res$sigmahat)),   # row-major flatten
  value         = as.numeric(res$value),
  iterations    = if (!is.null(res$iterations)) as.integer(res$iterations) else NA,
  stop_code     = if (!is.null(res$stop.code)) as.integer(res$stop.code) else NA,
  n             = nrow(data),
  p             = ncol(data)
)
write_json(out, json_out, auto_unbox = TRUE, digits = 12, na = "null")
