"""Reduce a pystatistics-vs-R survival pair to per-quantity agreement rows.

One job: for each procedure, compare the pystatistics estimate vectors/scalars
against R's, quantity by quantity, into the scalar agreement metrics
(max |Δ|, max relative Δ) that become the correctness table. Vectors are
length-checked and fail loud on a shape mismatch — a misaligned comparison is a
silent lie, not a small error.

Each row: ``{procedure, quantity, n_elements, max_abs, max_rel}``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative gap |a-b|/max(|b|, tiny); b is the reference (R)."""
    denom = np.maximum(np.abs(b), 1e-300)
    return np.abs(a - b) / denom


def _vec_row(procedure: str, quantity: str, sut: list[float], ref: list[float],
             ) -> dict[str, Any]:
    s = np.asarray(sut, float)
    r = np.asarray(ref, float)
    if s.shape != r.shape:
        raise ValueError(
            f"{procedure}.{quantity}: vector length mismatch "
            f"(pystatistics {s.shape} vs R {r.shape}) — refusing to compare "
            "misaligned quantities")
    abs_d = np.abs(s - r)
    rel_d = _rel(s, r)
    return {"procedure": procedure, "quantity": quantity,
            "n_elements": int(s.size),
            "max_abs": float(abs_d.max()), "max_rel": float(rel_d.max())}


def km_rows(sut: dict[str, Any], ref: dict[str, Any]) -> list[dict[str, Any]]:
    """KM curve agreement: time alignment, survival, n_risk, std_err, CI."""
    rows = []
    for q in ("time", "survival", "n_risk", "std_err", "ci_lower", "ci_upper"):
        rows.append(_vec_row("kaplan_meier", q, sut[q], ref[q]))
    rows.append(_vec_row("kaplan_meier", "median_survival",
                         [sut["median_survival"]], [ref["median_survival"]]))
    return rows


def logrank_rows(sut: dict[str, Any], ref: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        _vec_row("survdiff", "statistic", [sut["statistic"]], [ref["statistic"]]),
        _vec_row("survdiff", "p_value", [sut["p_value"]], [ref["p_value"]]),
        _vec_row("survdiff", "observed", sut["observed"], ref["observed"]),
        _vec_row("survdiff", "expected", sut["expected"], ref["expected"]),
    ]
    return rows


def coxph_rows(sut: dict[str, Any], ref: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for q in ("coefficients", "hazard_ratios", "standard_errors",
              "z_values", "p_values"):
        rows.append(_vec_row("coxph", q, sut[q], ref[q]))
    rows.append(_vec_row("coxph", "concordance",
                         [sut["concordance"]], [ref["concordance"]]))
    rows.append(_vec_row("coxph", "loglik_model",
                         [sut["loglik_model"]], [ref["loglik_model"]]))
    return rows


def discrete_rows(sut: dict[str, Any], ref: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for q in ("coefficients", "standard_errors", "z_values", "p_values"):
        rows.append(_vec_row("discrete_time", q, sut[q], ref[q]))
    return rows


def coxfeat_rows(procedure: str, sut: dict[str, Any], ref: dict[str, Any],
                 ) -> list[dict[str, Any]]:
    """Agreement for a feature-cluster Cox fit (stratified / start-stop /
    robust / cluster). ``procedure`` labels the specific feature row. Compares
    the quantities present in both payloads (robust/naive SE and the cox.zph
    table only when the fit carried them)."""
    rows = [
        _vec_row(procedure, "coefficients", sut["coefficients"],
                 ref["coefficients"]),
        _vec_row(procedure, "standard_errors", sut["standard_errors"],
                 ref["standard_errors"]),
        _vec_row(procedure, "loglik_model", [sut["loglik_model"]],
                 [ref["loglik_model"]]),
        _vec_row(procedure, "concordance", [sut["concordance"]],
                 [ref["concordance"]]),
    ]
    if "naive_se" in sut and "naive_se" in ref:
        rows.append(_vec_row(procedure, "naive_se", sut["naive_se"],
                             ref["naive_se"]))
    if "zph_chisq" in sut and "zph_chisq" in ref:
        rows.append(_vec_row(procedure, "zph_chisq", sut["zph_chisq"],
                             ref["zph_chisq"]))
        rows.append(_vec_row(procedure, "zph_p", sut["zph_p"], ref["zph_p"]))
    return rows


def kmfeat_rows(procedure: str, sut: dict[str, Any], ref: dict[str, Any],
                ) -> list[dict[str, Any]]:
    """Agreement for a feature-cluster KM curve (left truncation / strata).
    Curves are concatenated across strata in R's stratum order; ``std_err`` is
    on the survival scale (summary(survfit)$std.err) on both sides."""
    rows = []
    for q in ("time", "survival", "n_risk", "se"):
        rows.append(_vec_row(procedure, q, sut[q], ref[q]))
    for q in ("ci_lower", "ci_upper"):
        if q in sut and q in ref:
            # R marks an undefined CI bound with a -1 sentinel; compare only the
            # positions R actually defines.
            r = np.asarray(ref[q], float)
            s = np.asarray(sut[q], float)
            mask = r >= 0
            if mask.any():
                rows.append(_vec_row(procedure, q, s[mask].tolist(),
                                     r[mask].tolist()))
    return rows
