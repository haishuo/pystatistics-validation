"""Run and time pystatistics survival fits → canonical validation records.

One job: fit each survival procedure (kaplan_meier, survdiff, coxph,
discrete_time) under the shared device-aware timing protocol and return one
``validation-run/v1`` record per procedure, carrying the full estimate vectors
so the study generator can compare quantity-for-quantity against R.

Consumes the pystatistics library; never modifies it. Timing/memory and the
record envelope come from the shared harness ``pystatsval``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.measure import measure
from pystatsval.record import make_record


def _as_list(v: Any) -> list[float] | None:
    if v is None:
        return None
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]


def run_km_record(time: NDArray, event: NDArray, *, reps: int = 7, warmup: int = 1,
                  ) -> dict[str, Any]:
    """Fit + time kaplan_meier(time, event) → a record with the KM curve vectors."""
    from pystatistics.survival import kaplan_meier

    def _call():
        return kaplan_meier(time, event)

    wall, sol = measure(_call, device="cpu", repeats=reps, warmup=warmup)
    summary = {
        "time": _as_list(sol.time),
        "survival": _as_list(sol.survival),
        "n_risk": _as_list(sol.n_risk),
        "n_events": _as_list(sol.n_events),
        "std_err": _as_list(sol.se),
        "ci_lower": _as_list(sol.ci_lower),
        "ci_upper": _as_list(sol.ci_upper),
        "median_survival": (None if sol.median_survival is None
                            else float(sol.median_survival)),
    }
    return make_record(
        engine="pystatistics:cpu", dataset="lung_km", n=int(sol.n_observations),
        backend_name=sol.backend_name, precision="fp64", wall=wall,
        summary=summary,
        extra={"procedure": "kaplan_meier",
               "n_events_total": int(sol.n_events_total),
               "conf_type": sol.conf_type})


def run_logrank_record(time: NDArray, event: NDArray, group: NDArray, *,
                       reps: int = 7, warmup: int = 1) -> dict[str, Any]:
    """Fit + time survdiff(time, event, group) → a record."""
    from pystatistics.survival import survdiff

    def _call():
        return survdiff(time, event, group)

    wall, sol = measure(_call, device="cpu", repeats=reps, warmup=warmup)
    summary = {
        "statistic": float(sol.statistic),
        "df": int(sol.df),
        "p_value": float(sol.p_value),
        "observed": _as_list(sol.observed),
        "expected": _as_list(sol.expected),
    }
    return make_record(
        engine="pystatistics:cpu", dataset="lung_km", n=int(sol.n_per_group.sum()),
        backend_name=sol.backend_name, precision="fp64", wall=wall,
        summary=summary, extra={"procedure": "survdiff"})


def run_coxph_record(time: NDArray, event: NDArray, X: NDArray, names: list[str], *,
                     reps: int = 7, warmup: int = 1) -> dict[str, Any]:
    """Fit + time coxph(time, event, X) → a record with coef/HR/SE/z/p + diagnostics."""
    from pystatistics.survival import coxph

    def _call():
        return coxph(time, event, X, names=names)

    wall, sol = measure(_call, device="cpu", repeats=reps, warmup=warmup)
    null_ll, model_ll = sol.loglik
    summary = {
        "coefficients": _as_list(sol.coefficients),
        "hazard_ratios": _as_list(sol.hazard_ratios),
        "standard_errors": _as_list(sol.standard_errors),
        "z_values": _as_list(sol.z_values),
        "p_values": _as_list(sol.p_values),
        "concordance": float(sol.concordance),
        "loglik_null": float(null_ll),
        "loglik_model": float(model_ll),
    }
    return make_record(
        engine="pystatistics:cpu", dataset="lung_coxph", n=int(sol.n_observations),
        p=len(sol.coefficients), backend_name=sol.backend_name, precision="fp64",
        n_iter=int(sol.n_iter), converged=bool(sol.converged), wall=wall,
        summary=summary,
        extra={"procedure": "coxph", "n_events": int(sol.n_events),
               "coef_names": names, "ties": sol.ties})


def run_discrete_record(time: NDArray, event: NDArray, X: NDArray, names: list[str],
                        interval_bounds: NDArray, *, reps: int = 7, warmup: int = 1,
                        ) -> dict[str, Any]:
    """Fit + time discrete_time(time, event, X, intervals=bounds) → a record.

    Records ``person_period_n`` and ``n_intervals`` so the generator can assert
    the library's internal expansion matches the driver's reconstruction (the
    guard that keeps the R head-to-head honest).
    """
    from pystatistics.survival import discrete_time

    def _call():
        return discrete_time(time, event, X, names=names,
                             intervals=interval_bounds, backend="cpu")

    wall, sol = measure(_call, device="cpu", repeats=reps, warmup=warmup)
    summary = {
        "coefficients": _as_list(sol.coefficients),
        "standard_errors": _as_list(sol.standard_errors),
        "z_values": _as_list(sol.z_values),
        "p_values": _as_list(sol.p_values),
        "hazard_ratios": _as_list(sol.hazard_ratios),
        "glm_deviance": float(sol.glm_deviance),
        "glm_aic": float(sol.glm_aic),
    }
    return make_record(
        engine="pystatistics:cpu", dataset="lung_discrete", n=int(sol.n_observations),
        p=len(sol.coefficients), backend_name=sol.backend_name, precision="fp64",
        wall=wall, summary=summary,
        extra={"procedure": "discrete_time", "n_events": int(sol.n_events),
               "n_intervals": int(sol.n_intervals),
               "person_period_n": int(sol.person_period_n),
               "covariate_names": names})
