suppressMessages(library(mgcv))
d <- read.csv("sep_probit.csv")
r <- tryCatch({
  m <- gam(y ~ s(x1,k=10,bs="cr"), family=binomial(link=probit), method="REML", data=d)
  sprintf("mgcv completes: sp=%.5g edf=%.4f reml=%.6f warnings_in_fit=?", m$sp, sum(m$edf), m$gcv.ubre)
}, error=function(e) paste("mgcv ERROR:", conditionMessage(e)))
cat(r, "\n")
