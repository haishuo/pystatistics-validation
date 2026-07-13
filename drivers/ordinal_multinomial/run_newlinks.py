"""G1 correctness — the two polr links added after 4.6.10: cauchit and loglog.

pystatistics 5.0 exposes polr(link='cauchit') and polr(link='loglog'), which the
4.6.10 surface did not implement (they used to fail loud). This runner validates
both as first-class supported links:

  loglog  — vs MASS::polr(method='loglog'). MASS is a RELIABLE reference for the
            log-log link; agreement is optimizer-tier (coef/zeta/se/loglik).
  cauchit — vs an INDEPENDENT standard-cauchit MLE (om_cauchit_ref), NOT MASS.
            MASS::polr's cauchit under-converges on the heavy-tailed Cauchy
            likelihood: its `optim` stops at a strictly worse optimum than the true
            MLE, so it is not a trustworthy anchor. We record MASS's loglik gap as
            evidence and validate pystatistics against the independent MLE, which
            pystatistics matches to optimizer tier.

Emits artifacts/ordinal_multinomial/v<ver>/runs/newlinks.json.
Run: MVNMLE_DATA_DIR=Dev/datasets python drivers/ordinal_multinomial/run_newlinks.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omdata  # noqa: E402
from omref import r_polr, r_package_versions  # noqa: E402
from omcompare import arr_cmp, scalar_cmp, within, to_native  # noqa: E402
from om_cauchit_ref import cauchit_ordinal_mle  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402
from pystatistics.ordinal import polr  # noqa: E402

_ART = (Path(__file__).resolve().parents[2]
        / "artifacts/ordinal_multinomial/v{ver}/runs/newlinks.json")


def _cauchit_case(des, coef_abs=5e-4, thr_abs=5e-4, ll_abs=1e-3) -> dict:
    """polr cauchit vs the independent MLE; record MASS's under-convergence."""
    sol = polr(des.y, des.X, link="cauchit")
    ref = cauchit_ordinal_mle(des.y, des.X, des.n_levels)
    coef = arr_cmp(sol.coefficients, ref["coef"])
    thr = arr_cmp(sol.threshold_values, ref["thr"])
    ll = scalar_cmp(sol.log_likelihood, ref["loglik"])
    ok = (within(coef, abs_tol=coef_abs) and within(thr, abs_tol=thr_abs)
          and within(ll, abs_tol=ll_abs))
    # MASS reference (recorded, NOT used as anchor): show it under-converges.
    r = r_polr(des.y, des.X, "cauchit")
    mass_ll = float(r["loglik"]) if r.get("ok") else None
    mass_gap = (ref["loglik"] - mass_ll) if mass_ll is not None else None
    return {"group": "cauchit_vs_independent", "dataset": des.key,
            "link": "cauchit", "reference": "independent scipy cauchit MLE",
            "n": len(des.y), "K": des.n_levels,
            "coef": coef, "thr": thr, "loglik": ll,
            "py_loglik": float(sol.log_likelihood), "indep_loglik": ref["loglik"],
            "mass_loglik": mass_ll, "mass_underconv_gap": mass_gap,
            "mass_note": (f"MASS::polr cauchit under-converges: loglik {mass_ll:.4f} "
                          f"vs independent MLE {ref['loglik']:.4f} "
                          f"(worse by {mass_gap:.4f}); pystatistics reaches the true "
                          f"MLE ({float(sol.log_likelihood):.4f})."
                          if mass_ll is not None else "MASS::polr cauchit errored"),
            "converged": bool(sol.converged), "pass": bool(ok)}


def _loglog_case(des, coef_abs=2e-3, se_rel=5e-2, ll_abs=2e-3) -> dict:
    """polr loglog vs MASS::polr(method='loglog') — MASS is reliable here."""
    sol = polr(des.y, des.X, link="loglog")
    r = r_polr(des.y, des.X, "loglog")
    if not r.get("ok"):
        return {"group": "loglog_vs_mass", "dataset": des.key, "link": "loglog",
                "r_error": r.get("error"), "pass": False}
    coef = arr_cmp(sol.coefficients, r["coef"])
    zeta = arr_cmp(sol.threshold_values, r["zeta"])
    se_b = arr_cmp(sol.standard_errors, r["se_beta"])
    ll = scalar_cmp(sol.log_likelihood, r["loglik"])
    ok = (within(coef, abs_tol=coef_abs) and within(zeta, abs_tol=coef_abs)
          and within(se_b, rel_tol=se_rel) and within(ll, abs_tol=ll_abs))
    return {"group": "loglog_vs_mass", "dataset": des.key, "link": "loglog",
            "reference": "MASS::polr(method='loglog')",
            "n": len(des.y), "K": des.n_levels,
            "coef": coef, "zeta": zeta, "se_beta": se_b, "loglik": ll,
            "py_loglik": float(sol.log_likelihood), "mass_loglik": float(r["loglik"]),
            "converged": bool(sol.converged), "pass": bool(ok)}


def run_newlinks() -> list[dict]:
    housing = omdata.load_housing()
    synth = omdata.synth_ordinal()
    recs = []
    for des in (housing, synth):
        recs.append(_cauchit_case(des))
    for des in (housing, synth):
        recs.append(_loglog_case(des))
    return recs


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions()
    recs = run_newlinks()
    run = build_run(
        env=env,
        config={"suite": "ordinal-newlinks",
                "reference": "independent cauchit MLE (cauchit) / MASS::polr (loglog)",
                "contract": "cauchit and loglog links added after 4.6.10, validated "
                "as supported: cauchit vs an independent MLE because MASS::polr "
                "cauchit under-converges; loglog vs MASS::polr (reliable). Both "
                "agree with pystatistics to optimizer tier."},
        records=to_native([{"key": "newlinks", "checks": recs}]),
    )
    out = Path(str(_ART).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    print(f"  newlinks {sum(bool(r.get('pass')) for r in recs)}/{len(recs)} pass")
    for r in recs:
        tag = r.get("link", "?")
        if "mass_underconv_gap" in r and r["mass_underconv_gap"] is not None:
            print(f"   {tag}/{r['dataset']}: py={r['py_loglik']:.4f} "
                  f"indep={r['indep_loglik']:.4f} MASS={r['mass_loglik']:.4f} "
                  f"(MASS worse by {r['mass_underconv_gap']:.3f}) pass={r['pass']}")
        else:
            print(f"   {tag}/{r['dataset']}: pass={r.get('pass')}")


if __name__ == "__main__":
    main()
