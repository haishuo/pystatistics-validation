suppressMessages(library(mgcv))
# H4 protocol: n=2000, m in 1..6 smooths, poisson REML + GCV.Cp; best-of-5 timing
set.seed(99)
n <- 2000
xs <- lapply(1:6, function(j) runif(n))
f <- 1.2*sin(2*pi*xs[[1]]) + 0.8*cos(2*pi*xs[[2]]) + 0.6*sin(3*pi*xs[[3]]) +
     0.5*xs[[4]]^2*3 + 0.7*sin(2*pi*xs[[5]]) + 0.4*cos(3*pi*xs[[6]])
y <- rpois(n, exp(0.6*(f - mean(f)) + 1.0))
d <- data.frame(y=y); for (j in 1:6) d[[paste0("x",j)]] <- xs[[j]]
write.csv(d, "bench.csv", row.names=FALSE)
for (m in 1:6) {
  fml <- as.formula(paste("y ~", paste(sprintf("s(x%d,k=10,bs='cr')", 1:m), collapse="+")))
  for (meth in c("REML","GCV.Cp")) {
    ts <- replicate(5, system.time(gam(fml, family=poisson, method=meth, data=d))[3])
    mm <- gam(fml, family=poisson, method=meth, data=d)
    cat(sprintf("m=%d meth=%s  best=%.4fs  edf=%.5f  sp=%s\n", m, meth, min(ts),
        sum(mm$edf), paste(sprintf("%.6g", mm$sp), collapse=",")))
  }
}
