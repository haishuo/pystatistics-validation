suppressMessages(library(mgcv))
set.seed(2026)
n <- 500
x1 <- sort(runif(n)); x2 <- runif(n)
f <- 1.4*sin(2*pi*x1) + cos(2*pi*x2)

# One dataset per family, written for py to consume
ds <- list()
ds$poisson  <- rpois(n, exp(f - mean(f) + 1.0))
ds$binomial <- rbinom(n, 1, plogis(2*f))
ds$Gamma    <- rgamma(n, shape=4, scale=exp(0.8*f + 1.0)/4)
ds$gaussian <- exp(0.6*f) * (1 + rnorm(n, 0, 0.08))
ds$nb       <- rnbinom(n, size=3, mu=exp(f - mean(f) + 1.0))
for (k in names(ds)) write.csv(data.frame(y=ds[[k]], x1=x1, x2=x2),
                               sprintf("fs_%s.csv", k), row.names=FALSE)

run <- function(key, fam, meth) {
  d <- read.csv(sprintf("fs_%s.csv", key))
  m <- gam(y ~ s(x1,k=10,bs="cr") + s(x2,k=8,bs="cr"), family=fam, method=meth, data=d)
  th <- if (grepl("Negative Binomial", m$family$family))
          as.numeric(sub(".*\\((.*)\\).*", "\\1", m$family$family)) else NA
  cat(sprintf("%-16s meth=%-6s edf=%.6f sp=%s dev=%.6f theta=%s\n",
      key, meth, sum(m$edf), paste(sprintf("%.8g", m$sp), collapse=","),
      deviance(m), ifelse(is.na(th), "-", sprintf("%.5f", th))))
  write.csv(data.frame(fitted=fitted(m)), sprintf("fs_%s_%s_fitted.csv", key, meth), row.names=FALSE)
}
run("poisson",  poisson,               "REML")
run("poisson",  poisson,               "GCV.Cp")
run("binomial", binomial,              "REML")
run("binomial", binomial,              "GCV.Cp")
run("binomial-probit", binomial(link=probit), "REML")
run("binomial-probit", binomial(link=probit), "GCV.Cp")
run("Gamma-log", Gamma(link=log),      "GCV.Cp")
run("gaussian-log", gaussian(link=log),"GCV.Cp")
run("nb-fixed",  negbin(theta=3.0),    "REML")
run("nb-est",    nb(),                 "REML")
