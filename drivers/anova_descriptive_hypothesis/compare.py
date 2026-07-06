"""Shared comparison + artifact utilities for the anova/descriptive/hypothesis
validation drivers.

One job: turn a pair of (pystatistics value, reference value) into a JSON-ready
comparison record — max absolute and relative difference, and a pass flag
against a stated tolerance — so every artifact row is a frozen, renderable
number (R5: the report never hand-types a figure).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Default tolerances. These are DETERMINISTIC closed-form quantities: the only
# gap between pystatistics and R is fp64 round-off of identical inputs, so the
# bar is machine precision, not an optimizer tier.
TOL_EXACT = 1e-12      # closed-form statistics / exact p-values
TOL_TIGHT = 1e-9       # iterative-but-deterministic (Welch df, HL estimate, GG)
TOL_LOOSE = 1e-6       # anything with an internal root-find (fisher cond-MLE OR/CI)


def _arr(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).ravel()


def diff(py: Any, ref: Any) -> dict[str, float]:
    """Max abs/rel diff between two scalars-or-arrays (elementwise, aligned)."""
    a, b = _arr(py), _arr(ref)
    if a.shape != b.shape:
        return {"max_abs": float("inf"), "max_rel": float("inf"),
                "shape_mismatch": [list(a.shape), list(b.shape)]}
    if a.size == 0:
        return {"max_abs": 0.0, "max_rel": 0.0}
    # Matching infinities (same sign) are exact agreement (e.g. a one-sided CI's
    # +Inf bound, or a boundary odds-ratio) but a-b would be NaN. Zero them out;
    # a sign mismatch or finite-vs-inf stays +Inf and fails loudly.
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    with np.errstate(invalid="ignore"):
        abs_d = np.where(both_inf, 0.0, np.abs(a - b))
    denom = np.maximum(np.abs(b), 1e-300)
    rel_d = abs_d / denom
    return {"max_abs": float(np.max(abs_d)), "max_rel": float(np.max(rel_d))}


def case(name: str, quantity: str, py: Any, ref: Any, tol: float,
         *, scipy: Any = None, extra: dict | None = None,
         finding: str | None = None) -> dict[str, Any]:
    """Build one comparison record. ``pass`` is by max_abs OR max_rel <= tol
    (either satisfies — abs guards near-zero, rel guards large magnitudes).

    ``finding`` tags a case as a KNOWN identified divergence from R (F1, F2, …):
    it may not match R, but it is a catalogued convention gap, not an unexpected
    correctness regression. Summaries separate the two so a report never buries a
    real finding but also does not conflate it with an unexplained failure."""
    d = diff(py, ref)
    passed = (d["max_abs"] <= tol) or (d.get("max_rel", float("inf")) <= tol)
    rec: dict[str, Any] = {
        "case": name, "quantity": quantity, "tol": tol,
        "py": _to_list(py), "r": _to_list(ref),
        "max_abs": d["max_abs"], "max_rel": d.get("max_rel"),
        "pass": bool(passed),
    }
    if finding:
        rec["finding"] = finding
    if "shape_mismatch" in d:
        rec["shape_mismatch"] = d["shape_mismatch"]
    if scipy is not None:
        sd = diff(py, scipy)
        rec["scipy"] = _to_list(scipy)
        rec["scipy_max_abs"] = sd["max_abs"]
    if extra:
        rec["extra"] = extra
    return rec


def _to_list(v: Any) -> Any:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        return float(a)
    return [None if (x != x) else float(x) for x in a.ravel().tolist()]


def write_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    return p


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate R-match cases from tagged findings. ``clean`` = cases with no
    ``finding`` tag; those MUST all pass. Tagged findings are reported apart so a
    known convention gap never inflates the failure count nor gets hidden."""
    clean = [c for c in cases if not c.get("finding")]
    tagged = [c for c in cases if c.get("finding")]
    n_clean_pass = sum(1 for c in clean if c.get("pass"))
    worst = max((c["max_abs"] for c in clean
                 if c.get("max_abs") is not None and np.isfinite(c["max_abs"])),
                default=0.0)
    return {
        "n_cases": len(cases),
        "n_clean": len(clean), "n_clean_pass": n_clean_pass,
        "n_clean_fail": len(clean) - n_clean_pass,
        "clean_all_pass": n_clean_pass == len(clean),
        "worst_clean_max_abs": worst,
        "findings": sorted({c["finding"] for c in tagged}),
        "n_finding_cases": len(tagged),
    }
