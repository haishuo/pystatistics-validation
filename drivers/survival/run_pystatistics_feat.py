"""Fit the A1+VA-8 survival feature cluster in pystatistics → comparable records.

One job: run each feature-cluster procedure (stratified / counting-process /
left-truncation / robust-cluster Cox, stratified & left-truncated KM, cox.zph)
under the shared timing protocol, returning a ``validation-run/v1`` record whose
``summary`` matches the shape the R feature worker emits (so ``agreement`` can
reduce them quantity-by-quantity).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.measure import measure
from pystatsval.record import make_record


def run_coxfeat_record(
    time: NDArray, event: NDArray, X: NDArray, *,
    dataset: str, ties: str = "efron", start: NDArray | None = None,
    strata: NDArray | None = None, cluster: NDArray | None = None,
    robust: bool = False, zph_transform: str = "",
    reps: int = 5, warmup: int = 1,
) -> dict[str, Any]:
    """Fit a feature-cluster Cox model; optionally append its cox.zph table."""
    from pystatistics.survival import coxph, cox_zph

    def _call():
        return coxph(time, event, X, start=start, strata=strata,
                     cluster=cluster, robust=robust, ties=ties)

    wall, sol = measure(_call, device="cpu", repeats=reps, warmup=warmup)
    summary: dict[str, Any] = {
        "coefficients": np.asarray(sol.coefficients).tolist(),
        "standard_errors": np.asarray(sol.standard_errors).tolist(),
        "loglik_model": float(sol.loglik[1]),
        "concordance": float(sol.concordance),
    }
    if sol.robust:
        summary["naive_se"] = np.asarray(sol.naive_standard_errors).tolist()
    if zph_transform:
        z = cox_zph(sol, transform=zph_transform)
        summary["zph_chisq"] = np.asarray(z.chisq).tolist()
        summary["zph_p"] = np.asarray(z.p_values).tolist()
    return make_record(
        engine="pystatistics:cpu", dataset=dataset,
        n=int(sol.n_observations), p=len(summary["coefficients"]),
        loglik=float(sol.loglik[1]), n_iter=int(sol.n_iter),
        converged=bool(sol.converged), wall=wall,
        backend_name=sol.backend_name, precision="fp64",
        parameterization=f"ties={ties};robust={sol.robust};n_strata={sol.n_strata}",
        summary=summary,
        extra={"procedure": "coxfeat", "warnings": list(sol.warnings)},
    )


def run_kmfeat_record(
    time: NDArray, event: NDArray, *,
    dataset: str, entry: NDArray | None = None, strata: NDArray | None = None,
    conf_type: str = "log", reps: int = 5, warmup: int = 1,
) -> dict[str, Any]:
    """Fit a stratified / left-truncated KM; concatenate curves in R's stratum
    (ascending-label) order so the reference comparison lines up."""
    from pystatistics.survival import kaplan_meier
    from pystatistics.survival._km_strata import StratifiedKMSolution

    def _call():
        return kaplan_meier(time, event, entry=entry, strata=strata,
                            conf_type=conf_type)

    wall, sol = measure(_call, device="cpu", repeats=reps, warmup=warmup)

    if isinstance(sol, StratifiedKMSolution):
        curves = [sol[label] for label in sorted(sol.strata, key=str)]
    else:
        curves = [sol]

    def _cat(attr):
        return np.concatenate([np.asarray(getattr(c, attr)) for c in curves]) \
            if curves else np.array([])

    summary = {
        "time": _cat("time").tolist(),
        "survival": _cat("survival").tolist(),
        "n_risk": _cat("n_risk").tolist(),
        "se": _cat("se").tolist(),
        "ci_lower": _cat("ci_lower").tolist(),
        "ci_upper": _cat("ci_upper").tolist(),
    }
    n_total = sum(int(c.n_observations) for c in curves) if curves else len(time)
    return make_record(
        engine="pystatistics:cpu", dataset=dataset,
        n=n_total, p=0, loglik=None, n_iter=None, converged=True, wall=wall,
        backend_name="cpu_km", precision="fp64",
        parameterization=f"conf_type={conf_type}",
        summary=summary, extra={"procedure": "kmfeat"},
    )
