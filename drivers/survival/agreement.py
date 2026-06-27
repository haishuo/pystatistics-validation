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
