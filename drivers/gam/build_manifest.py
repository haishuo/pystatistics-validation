"""Flatten the gam run artifacts into the render pipeline's manifest + CSVs.

Reads the frozen runs/correctness.json and runs/performance.json produced by
the drivers and emits, under artifacts/gam/v<ver>/:
  - runs/<study>_summary.csv  (one flat table per study; scalar cells only)
  - manifest.json             (studies referencing those CSVs + run files)

Numbers are copied verbatim from the runs — this script only reshapes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pystatistics

_VER = pystatistics.__version__
_ROOT = Path(__file__).resolve().parents[2] / f"artifacts/gam/v{_VER}"
_RUNS = _ROOT / "runs"
_FROZEN_UTC = "2026-07-02"


def _load(name: str) -> dict:
    return json.loads((_RUNS / name).read_text())


def _write_csv(name: str, cols: list[str], rows: list[dict]) -> None:
    path = _RUNS / name
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _g(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}"


def main() -> None:
    corr = _load("correctness.json")
    perf = _load("performance.json")
    records = [r for r in corr["records"] if r.get("key") != "fidelity"]
    fidelity = next(r for r in corr["records"] if r.get("key") == "fidelity")

    # --- Study 1: tier-1 fixed-sp tight ---
    t1_cols = ["case", "n", "fitted_max_abs", "total_edf_abs",
               "param_coef_max_abs", "param_se_max_rel", "scale_rel",
               "aic_abs", "smooth_stat_kind", "smooth_stat_max_rel"]
    t1_rows = []
    for r in records:
        t = r.get("tier1_fixedsp")
        if not t:
            continue
        t1_rows.append({
            "case": r["key"], "n": r["n"],
            "fitted_max_abs": _g(t["fitted"]["max_abs"]),
            "total_edf_abs": _g(t["total_edf"]["abs"]),
            "param_coef_max_abs": _g(t["param_coef"]["max_abs"]),
            "param_se_max_rel": _g(t["param_se"]["max_rel"]),
            "scale_rel": _g(t["scale"]["rel"]),
            "aic_abs": _g(t["aic_abs"]),
            "smooth_stat_kind": t["smooth_stat_kind"],
            "smooth_stat_max_rel": _g(t["smooth_stat"]["max_rel"]),
        })
    _write_csv("tier1_fixedsp_summary.csv", t1_cols, t1_rows)

    # --- Study 2: tier-2 free selection ---
    t2_cols = ["case", "family", "method", "n", "n_coef_py", "n_coef_r",
               "total_edf_abs", "smooth_edf_max_abs", "score_rel",
               "deviance_rel", "fitted_max_abs", "sp_rel_nonnull"]
    t2_rows = []
    for r in records:
        t = r["tier2_free"]
        # sp on non-null smooths only (null terms are lambda->inf, unidentified)
        sp_nonnull = "n/a"
        if "per_smooth" in t:
            nn = [ps["sp_rel"] for ps in t["per_smooth"]
                  if not ps["near_null"]]
            sp_nonnull = _g(max(nn)) if nn else "all-null"
        t2_rows.append({
            "case": r["key"], "family": r["family"], "method": r["method"],
            "n": r["n"], "n_coef_py": r["n_coef"]["py"],
            "n_coef_r": r["n_coef"]["r"],
            "total_edf_abs": _g(t["total_edf"]["abs"]),
            "smooth_edf_max_abs": _g(t["smooth_edf"]["max_abs"]),
            "score_rel": _g(t["score"]["rel"]),
            "deviance_rel": _g(t["deviance"]["rel"]),
            "fitted_max_abs": _g(t["fitted"]["max_abs"]),
            "sp_rel_nonnull": sp_nonnull,
        })
    _write_csv("tier2_free_summary.csv", t2_cols, t2_rows)

    # --- Study 3: per-smooth sp/EDF (null-term identifiability story) ---
    ps_cols = ["case", "term", "sp_py", "sp_r", "sp_rel", "edf_py", "edf_r",
               "edf_abs", "near_null"]
    ps_rows = []
    for r in records:
        for ps in r["tier2_free"].get("per_smooth", []):
            ps_rows.append({
                "case": r["key"], "term": ps["term"],
                "sp_py": _g(ps["sp_py"]), "sp_r": _g(ps["sp_r"]),
                "sp_rel": _g(ps["sp_rel"]),
                "edf_py": _g(ps["edf_py"], 4), "edf_r": _g(ps["edf_r"], 4),
                "edf_abs": _g(ps["edf_abs"]), "near_null": ps["near_null"],
            })
    _write_csv("per_smooth_sp_summary.csv", ps_cols, ps_rows)

    # --- Study 4: fidelity (fail-loud) ---
    fid_cols = ["check", "why", "raised", "pass"]
    fid_rows = [{"check": c["check"], "why": c["why"],
                 "raised": c["raised"] or "(nothing)", "pass": c["pass"]}
                for c in fidelity["checks"]]
    _write_csv("fidelity_summary.csv", fid_cols, fid_rows)

    # --- Study 5: performance ---
    pf_cols = ["axis", "n", "n_smooths", "k", "py_ms", "r_ms", "ratio_py_over_r"]
    pf_rows = [{"axis": r["axis"], "n": r["n"], "n_smooths": r["n_smooths"],
                "k": r["k"], "py_ms": _g(r["py_s"] * 1e3, 4),
                "r_ms": _g(r["r_s"] * 1e3, 4),
                "ratio_py_over_r": _g(r["ratio_py_over_r"])}
               for r in perf["records"]]
    _write_csv("performance_summary.csv", pf_cols, pf_rows)

    env = corr["env"]
    exp_py = perf["config"]["empirical_exponent_py"]
    exp_r = perf["config"]["empirical_exponent_r"]

    manifest = {
        "schema": "validation-artifact-manifest/v1",
        "subsystem": "gam",
        "pystatistics_version": _VER,
        "install_source": env["install_source"],
        "evidence_state": "native-harness",
        "frozen_utc": _FROZEN_UTC,
        "provenance": {
            "note": (
                f"First-time validation of pystatistics.gam at {_VER}, immediately "
                "after the 4.6.0 numerical rewrite this validation drove. The prior "
                "gam (through 4.5.7) was never validated against mgcv and shipped a "
                "silently-wrong automatic smoothing selection (finding H0) plus a "
                "silently-wrong fp32 GPU path (H1); both were fixed/removed at 4.6.0. "
                "Reference mgcv " + env.get("mgcv", "?") + ". Identical float64 inputs "
                "to both engines (each case dumps its data to a temp CSV both read). "
                "Two-tier contract: fixed-sp tight (cr) + free-selection optimizer "
                "tier. Adversarially reviewed (2 HIGH findings H2/H3 fixed pre-release)."
            )
        },
        "hosts": {
            f"{env.get('cpu', 'cpu')}+r": {
                "device": f"CPU (numpy {env.get('numpy', '?')}) + R mgcv "
                          f"{env.get('mgcv', '?')}",
            }
        },
        "reference": {"name": "R mgcv::gam (Wood)", "kind": "cran"},
        "studies": [
            {
                "id": "g1_tier1_fixedsp",
                "title": "G1 correctness — TIER 1 (tight): at mgcv's selected sp, "
                         "fed to both engines (cr designs)",
                "device": ["cpu"], "claim": "agreement", "host": f"{env.get('cpu','cpu')}+r",
                "summary": "runs/tier1_fixedsp_summary.csv",
                "run": "runs/correctness.json",
                "summary_cols": t1_cols,
                "note": "At a SHARED sp the penalized fit is one stable-QR solve, so "
                        "fitted values / EDF / coefficients / posterior SEs / scale "
                        "agree to fp64 arithmetic (≤~1e-9, ≤~1e-13 for Gaussian). "
                        "aic_abs is the documented edf2 df convention difference; "
                        "smooth_stat_max_rel is the approximate Wood-2013 test — see "
                        "limitations (widest on the concurvity case).",
            },
            {
                "id": "g1_tier2_free",
                "title": "G1 correctness — TIER 2 (optimizer): free selection, each "
                         "engine picks its own sp — families + R10 + R15",
                "device": ["cpu"], "claim": "agreement", "host": f"{env.get('cpu','cpu')}+r",
                "summary": "runs/tier2_free_summary.csv",
                "run": "runs/correctness.json",
                "summary_cols": t2_cols,
                "note": "R15 default (mcycle), Gaussian/Poisson/Binomial gamSim 4-smooth, "
                        "and R10 hard cases. sp_rel_nonnull is the max sp disagreement "
                        "over IDENTIFIED (non-null) cr smooths; null/near-linear terms "
                        "(λ→∞) are unidentified in sp and compared by EDF instead "
                        "(next study). n_coef matches mgcv (k−1 per smooth after "
                        "centering). tp cases compared by fitted/EDF only.",
            },
            {
                "id": "g1_per_smooth_sp",
                "title": "G1 — per-smooth sp vs EDF: null/near-linear terms are "
                         "sp-unidentified (λ→∞) but EDF-matched",
                "device": ["cpu"], "claim": "agreement", "host": f"{env.get('cpu','cpu')}+r",
                "summary": "runs/per_smooth_sp_summary.csv",
                "run": "runs/correctness.json",
                "summary_cols": ps_cols,
                "note": "Identified smooths agree in sp to ~1e-5; near_null=True terms "
                        "(EDF→1) show a large sp ratio because λ→∞ is only weakly "
                        "identified — there EDF agrees to ≤1e-4 and the fit is "
                        "identical. This is correct, not a defect.",
            },
            {
                "id": "g2_fidelity",
                "title": "G2 fidelity — fail-loud, no silent substitution (A6)",
                "device": ["cpu"], "claim": "fail-loud", "host": env.get("cpu", "cpu"),
                "summary": "runs/fidelity_summary.csv",
                "run": "runs/correctness.json",
                "summary_cols": fid_cols,
                "note": "Unsupported basis/family/method/k and the removed GPU backend "
                        "all raise rather than silently substituting; wrong-length "
                        "names raises rather than mislabelling coefficients (H3).",
            },
            {
                "id": "g3_performance",
                "title": "G3 performance — pystatistics CPU vs mgcv::gam",
                "device": ["cpu"], "claim": "benchmark", "host": f"{env.get('cpu','cpu')}+r",
                "summary": "runs/performance_summary.csv",
                "run": "runs/performance.json",
                "summary_cols": pf_cols,
                "note": f"Same estimator (penalised spline + GCV), CPU-vs-CPU vs mgcv's "
                        f"compiled C, best-of-{perf['config']['reps']}. Single smooth: "
                        f"parity across n (py/mgcv ≈1.0–1.8×, 0.97× at n=50000; "
                        f"empirical exponent py={exp_py:.2f} vs r={exp_r:.2f}). "
                        f"Multi-smooth lags with the sp count (finite-difference search "
                        f"vs Newton) — a documented R2 cost (H4).",
            },
        ],
        "record_fields": {
            "note": "See runs/correctness.json and runs/performance.json for the full "
                    "per-case records; the CSVs are the rendered summaries."
        },
    }
    (_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {_ROOT / 'manifest.json'} + {len(manifest['studies'])} summary CSVs")


if __name__ == "__main__":
    main()
