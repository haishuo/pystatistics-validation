suppressMessages(library(mgcv))
d <- read.csv("fs_binomial.csv")
# fixed at mgcv's own REML-selected sp
m <- gam(y ~ s(x1,k=10,bs="cr") + s(x2,k=8,bs="cr"), family=binomial(link=probit),
         method="REML", sp=c(40.00448,23.14561), data=d)
cat(sprintf("mgcv probit @fixed sp: edf=%.8f dev=%.8f reml=%.8f\n", sum(m$edf), deviance(m), m$gcv.ubre))
write.csv(data.frame(fitted=fitted(m)), "probit_t1_fitted.csv", row.names=FALSE)
