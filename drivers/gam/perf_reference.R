#!/usr/bin/env Rscript
# mgcv::gam timing for one design. Usage: Rscript perf_reference.R <spec.json> <out.json>
#   spec.json: {data_csv, formula, family, method, reps}
# Reports the MINIMUM fit wall-time over `reps` (compute only; data already
# in memory), matching how the Python side times its own fit.
suppressMessages({library(mgcv); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
spec <- fromJSON(args[[1]]); out_path <- args[[2]]
d <- read.csv(spec$data_csv)
fam <- switch(spec$family, gaussian = gaussian(), poisson = poisson(),
              binomial = binomial(), stop("family"))
form <- as.formula(spec$formula)
meth <- if (spec$method == "GCV") "GCV.Cp" else spec$method
reps <- as.integer(spec$reps)
times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  m <- gam(form, data = d, family = fam, method = meth)
  times[i] <- proc.time()[["elapsed"]] - t0
}
write_json(list(min_s = min(times), median_s = median(times),
                edf = sum(m$edf), reps = reps),
           out_path, auto_unbox = TRUE, digits = 10)
