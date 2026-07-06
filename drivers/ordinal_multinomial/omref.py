"""R reference bridge for MASS::polr and nnet::multinom.

One job: fit the R reference on the SAME numeric design the driver dumps (so
agreement is to fp64 round-off of identical numbers), matching the estimator and
convention between engines, and return every quantity the validation compares —
including R's convergence flag and any WARNINGS it emits (R10 requires matching
R's failures/warnings, not just its numbers).

Conventions matched:
- ``polr``: X carries no intercept (thresholds are the intercepts); the response
  is an ORDERED factor; ``method`` == the link (logistic/probit/cloglog).
- ``multinom``: nnet baselines on the FIRST factor level, pystatistics on the
  LAST code. The caller passes ``r_levels`` = ``[K-1, 0, 1, ..., K-2]`` so
  nnet's baseline coincides with pystatistics' reference; ``decay=0`` matches the
  unpenalized softmax MLE. Fitted-probability columns come back labelled so the
  caller can reorder them to code order 0..K-1.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def have_rscript() -> bool:
    import shutil
    return shutil.which("Rscript") is not None


def _run_r(script: str, timeout_s: float = 600.0) -> dict[str, Any]:
    proc = subprocess.run(["Rscript", "-e", script],
                          capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"R error (rc={proc.returncode}): "
                           f"{proc.stderr.strip()[:600]}")
    # The script prints exactly one JSON blob on stdout (its last cat()).
    txt = proc.stdout.strip()
    start = txt.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON in R output: {txt[:300]}")
    return json.loads(txt[start:])


_WARN_WRAP = r"""
.warns <- character(0)
.res <- withCallingHandlers(
  tryCatch({{ {body} ; "ok" }},
           error = function(e) paste0("ERROR:", conditionMessage(e))),
  warning = function(w) {{ .warns <<- c(.warns, conditionMessage(w));
                          invokeRestart("muffleWarning") }})
"""


def r_polr(y: NDArray[np.integer], X: NDArray[np.floating], link: str,
           timeout_s: float = 600.0) -> dict[str, Any]:
    """Fit MASS::polr on (y, X); return coefficients, thresholds, SEs, loglik,
    AIC, fitted probs, convergence, and any warnings.

    ``link`` is one of 'logistic', 'probit', 'cloglog' (MASS::polr ``method``).
    """
    tmp = Path(tempfile.mkdtemp(prefix="om_polr_"))
    np.savetxt(tmp / "X.csv", np.asarray(X, float), delimiter=",")
    np.savetxt(tmp / "y.csv", np.asarray(y, int), fmt="%d")
    K = int(np.max(y)) + 1
    body = (
        f'X<-as.matrix(read.csv("{tmp}/X.csv",header=FALSE));'
        f'yc<-scan("{tmp}/y.csv",quiet=TRUE);'
        f'y<-factor(yc,levels=0:{K-1},ordered=TRUE);'
        f'df<-data.frame(y=y,X);'
        f'm<-MASS::polr(y~.,data=df,method="{link}",Hess=TRUE)'
    )
    script = _WARN_WRAP.format(body=body) + rf"""
if (identical(.res,"ok")) {{
  se <- sqrt(diag(vcov(m)))
  p <- ncol(X); nb <- p; nz <- {K-1}
  probs <- predict(m, type="probs")
  out <- list(ok=TRUE,
    coef=unname(as.numeric(coef(m))),
    zeta=unname(as.numeric(m$zeta)),
    se_beta=unname(as.numeric(se[1:nb])),
    se_zeta=unname(as.numeric(se[(nb+1):(nb+nz)])),
    loglik=as.numeric(logLik(m)), aic=AIC(m), edf=m$edf,
    deviance=deviance(m), n_iter=if(is.null(m$niter)) NA else m$niter,
    fitted=as.matrix(probs), fitted_cols=colnames(probs),
    warnings=.warns)
}} else {{ out <- list(ok=FALSE, error=.res, warnings=.warns) }}
library(jsonlite); cat(toJSON(out, digits=14, auto_unbox=TRUE, na="null"))
"""
    return _run_r(script, timeout_s)


def r_multinom(y: NDArray[np.integer], Xf: NDArray[np.floating],
               r_levels: list[int], maxit: int = 2000,
               reltol: float = 1e-12, timeout_s: float = 600.0) -> dict[str, Any]:
    """Fit nnet::multinom(decay=0) on (y, Xf) with the baseline releveled to
    coincide with pystatistics' last-code reference.

    ``Xf`` is the design WITHOUT the intercept column (nnet adds it). Returns the
    (K-1)x(1+p) coefficient and SE matrices, loglik, AIC, deviance, convergence
    flag, fitted probs (with column labels), and any warnings.
    """
    tmp = Path(tempfile.mkdtemp(prefix="om_multinom_"))
    np.savetxt(tmp / "X.csv", np.asarray(Xf, float), delimiter=",")
    np.savetxt(tmp / "y.csv", np.asarray(y, int), fmt="%d")
    rlev = ",".join(str(v) for v in r_levels)
    body = (
        f'X<-as.matrix(read.csv("{tmp}/X.csv",header=FALSE));'
        f'yc<-scan("{tmp}/y.csv",quiet=TRUE);'
        f'y<-factor(yc,levels=c({rlev}));'
        f'df<-data.frame(y=y,X);'
        f'm<-nnet::multinom(y~.,data=df,decay=0,maxit={maxit},'
        f'reltol={reltol},trace=FALSE)'
    )
    script = _WARN_WRAP.format(body=body) + r"""
if (identical(.res,"ok")) {
  cf <- coef(m); se <- summary(m)$standard.errors
  if (is.null(dim(cf))) { cf <- matrix(cf, nrow=1); se <- matrix(se, nrow=1) }
  ft <- fitted(m)
  out <- list(ok=TRUE,
    coef=cf, se=se, coef_rows=rownames(cf),
    loglik=as.numeric(logLik(m)), aic=AIC(m), edf=m$edf,
    deviance=deviance(m), conv=m$convergence,
    fitted=as.matrix(ft), fitted_cols=colnames(ft),
    warnings=.warns)
} else { out <- list(ok=FALSE, error=.res, warnings=.warns) }
library(jsonlite); cat(toJSON(out, digits=14, auto_unbox=TRUE, na="null"))
"""
    return _run_r(script, timeout_s)


def reorder_fitted(fitted: NDArray, cols: list[str], n_classes: int) -> NDArray:
    """Reorder R fitted-probability columns (labelled by class code) to 0..K-1."""
    fitted = np.atleast_2d(np.asarray(fitted, float))
    label_to_idx = {str(c): j for j, c in enumerate(cols)}
    order = [label_to_idx[str(k)] for k in range(n_classes)]
    return fitted[:, order]


def r_time_polr(y, X, link: str, reps: int = 5,
                timeout_s: float = 1200.0) -> float:
    """Best-of-``reps`` MASS::polr fit wall time (seconds), measured inside R."""
    tmp = Path(tempfile.mkdtemp(prefix="om_tpolr_"))
    np.savetxt(tmp / "X.csv", np.asarray(X, float), delimiter=",")
    np.savetxt(tmp / "y.csv", np.asarray(y, int), fmt="%d")
    K = int(np.max(y)) + 1
    script = rf"""
suppressMessages({{library(MASS); library(jsonlite)}})
X<-as.matrix(read.csv("{tmp}/X.csv",header=FALSE))
y<-factor(scan("{tmp}/y.csv",quiet=TRUE),levels=0:{K-1},ordered=TRUE)
df<-data.frame(y=y,X)
best<-Inf
for (i in 1:{reps}) {{
  t<-system.time(m<-MASS::polr(y~.,data=df,method="{link}",Hess=TRUE))[["elapsed"]]
  best<-min(best,t)
}}
cat(toJSON(list(elapsed=best),auto_unbox=TRUE))
"""
    return float(_run_r(script, timeout_s)["elapsed"])


def r_time_multinom(y, Xf, r_levels: list[int], reps: int = 5,
                    maxit: int = 2000, timeout_s: float = 1200.0) -> float:
    """Best-of-``reps`` nnet::multinom(decay=0) fit wall time (seconds)."""
    tmp = Path(tempfile.mkdtemp(prefix="om_tmulti_"))
    np.savetxt(tmp / "X.csv", np.asarray(Xf, float), delimiter=",")
    np.savetxt(tmp / "y.csv", np.asarray(y, int), fmt="%d")
    rlev = ",".join(str(v) for v in r_levels)
    script = rf"""
suppressMessages({{library(nnet); library(jsonlite)}})
X<-as.matrix(read.csv("{tmp}/X.csv",header=FALSE))
y<-factor(scan("{tmp}/y.csv",quiet=TRUE),levels=c({rlev}))
df<-data.frame(y=y,X)
best<-Inf
for (i in 1:{reps}) {{
  t<-system.time(m<-nnet::multinom(y~.,data=df,decay=0,maxit={maxit},
                                   trace=FALSE))[["elapsed"]]
  best<-min(best,t)
}}
cat(toJSON(list(elapsed=best),auto_unbox=TRUE))
"""
    return float(_run_r(script, timeout_s)["elapsed"])


def r_package_versions() -> dict[str, str]:
    script = r"""
library(jsonlite)
v <- function(p) tryCatch(as.character(packageVersion(p)), error=function(e) NA)
cat(toJSON(list(MASS=v("MASS"), nnet=v("nnet"),
                R=paste0(R.version$major,".",R.version$minor)),
           auto_unbox=TRUE, na="null"))
"""
    try:
        return _run_r(script, timeout_s=120)
    except Exception:
        return {}
