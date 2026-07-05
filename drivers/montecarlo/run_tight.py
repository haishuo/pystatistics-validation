"""Tight tier — SHARED resamples, machine precision.

The crux of validating a stochastic method against R. R's RNG != numpy's RNG, so
naive bit-matching is wrong. Instead we run a GENUINE ``boot::boot`` in R, export
its resample index matrix, feed the IDENTICAL indices to pystatistics, and
compare the statistic + CI arithmetic — which is DETERMINISTIC given the
replicates — to machine precision (or the defined convention).

What this isolates and the honest per-quantity tolerance contract:
  - t0 (observed statistic)           : machine precision  (both fp64, same math)
  - t (replicate statistic on shared idx): machine precision
  - bias, se (mean(t)-t0, sd(t))      : machine precision
  - normal CI (2*t0-mean(t) +/- z*sd) : machine precision
  - basic / perc CI                   : np.quantile type-7 vs R boot:::norm.inter
                                        — a documented interpolation convention;
                                        agrees to ~1e-3, shrinks with R.
  - BCa CI                            : z0 matches exactly; pystatistics uses
                                        Efron's JACKKNIFE acceleration a, R's
                                        boot.ci default uses REGRESSION influence.
                                        Both are textbook BCa; the tail endpoints
                                        differ by a documented convention amount.
                                        (Calibration is settled by the coverage
                                        study, run_equivalence.py.)

Emits artifacts/montecarlo/v<ver>/runs/tight.json (render-from-artifacts).

Run: MVNMLE_DATA_DIR=Dev/datasets python drivers/montecarlo/run_tight.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcdata  # noqa: E402
from rref import r_reference, r_package_versions  # noqa: E402
from mccompare import arr_cmp, scalar_cmp, within  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402

from pystatistics.montecarlo import boot, boot_ci  # noqa: E402
from pystatistics.montecarlo._common import BootParams  # noqa: E402
from pystatistics.montecarlo.solution import BootstrapSolution  # noqa: E402
from pystatistics.montecarlo._influence import jackknife_influence  # noqa: E402
from pystatistics.core.result import Result  # noqa: E402

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/montecarlo/v{ver}/runs/tight.json")

# Tolerances (the contract above).
TOL_MACHINE = 1e-10       # t0/t/bias/se/normal — deterministic, same math
TOL_QUANTILE = 5e-3       # basic/perc — type-7 vs norm.inter convention @ R=5000
TOL_Z0 = 1e-9             # BCa bias-correction — identical definition


def _shared_solution(data, stat, t0, t):
    """A BootstrapSolution carrying R's shared replicates (for boot_ci)."""
    R = len(t)
    base = boot(data, stat, n_resamples=5, seed=0)   # a valid _design only
    params = BootParams(t0=np.asarray(t0, float), t=np.asarray(t, float).reshape(R, 1),
                        R=R, bias=np.array([t.mean() - t0[0]]),
                        se=np.array([t.std(ddof=1)]), ci=None, ci_conf_level=None)
    res = Result(params=params, info=base._result.info, timing=None,
                 backend_name="shared", warnings=())
    return BootstrapSolution(_result=res, _design=base._design)


def run_boot_tight() -> list[dict]:
    """boot: t0/t/bias/se + boot.ci (all types) on R's shared resamples."""
    recs = []
    R = 5000
    for key, stat_name in [("law", "corr"), ("city", "ratio")]:
        data = mcdata.load(key).values
        stat = mcdata.STAT[stat_name]
        r = r_reference("boot", data, statistic=stat_name, R=R, seed=4242)
        idx = np.array(r["idx"], int).reshape(r["R"], r["n"]) - 1
        t_R = np.array(r["t"], float)

        # (1) statistic + t0 reproduction on shared indices
        t_py = np.array([stat(data, idx[b])[0] for b in range(r["R"])])
        t0_py = stat(data, np.arange(data.shape[0]))
        c_t = arr_cmp(t_py, t_R)
        c_t0 = scalar_cmp(float(t0_py[0]), float(r["t0"]))

        # (2) bias / se from shared replicates
        sol = _shared_solution(data, stat, np.array([r["t0"]]), t_R)
        c_bias = scalar_cmp(float(sol.bias[0]), float(r["bias"]))
        c_se = scalar_cmp(float(sol.se[0]), float(r["se"]))

        # (3) boot.ci — every non-studentized type on the shared replicates
        ci = boot_ci(sol, ci_type="all").ci
        ci_cmps = {
            "normal": (arr_cmp(ci["normal"][0], r["ci_normal"]), TOL_MACHINE),
            "basic": (arr_cmp(ci["basic"][0], r["ci_basic"]), TOL_QUANTILE),
            "perc": (arr_cmp(ci["perc"][0], r["ci_perc"]), TOL_QUANTILE),
            "bca": (arr_cmp(ci["bca"][0], r["ci_bca"]), None),  # convention, reported
        }
        # (4) BCa ingredients: z0 exact; a is pystat-jackknife vs R reg/jack
        L = jackknife_influence(sol, 0)
        a_py = float(np.sum(L**3) / (6.0 * np.sum(L**2) ** 1.5))
        prop = np.clip(np.sum(t_R < r["t0"]) / R, 1/(2*R), 1-1/(2*R))
        from scipy.stats import norm
        z0_py = float(norm.ppf(prop))

        rec = {
            "group": "boot_tight", "dataset": key, "statistic": stat_name, "R": R,
            "t_stat": c_t, "t0": c_t0, "bias": c_bias, "se": c_se,
            "ci_normal": ci_cmps["normal"][0], "ci_basic": ci_cmps["basic"][0],
            "ci_perc": ci_cmps["perc"][0], "ci_bca": ci_cmps["bca"][0],
            "bca_z0": {"py": z0_py, "r": float(r["z0"]),
                       "abs": abs(z0_py - float(r["z0"]))},
            "bca_a": {"py_jack": a_py, "r_jack": float(r["a_jack"]),
                      "r_reg_default": float(r["a_reg"]),
                      "abs_vs_r_jack": abs(a_py - float(r["a_jack"]))},
            "bca_note": "pystatistics BCa uses Efron jackknife acceleration; R "
                        "boot.ci default uses regression influence (a_reg). z0 "
                        "matches exactly; endpoint gap is this documented "
                        "convention, amplified in steep tails.",
            "pass": (within(c_t, abs_tol=TOL_MACHINE)
                     and within(c_t0, abs_tol=TOL_MACHINE)
                     and within(c_bias, abs_tol=TOL_MACHINE)
                     and within(c_se, abs_tol=TOL_MACHINE)
                     and within(ci_cmps["normal"][0], abs_tol=TOL_MACHINE)
                     and within(ci_cmps["basic"][0], abs_tol=TOL_QUANTILE)
                     and within(ci_cmps["perc"][0], abs_tol=TOL_QUANTILE)
                     and abs(z0_py - float(r["z0"])) < TOL_Z0),
        }
        recs.append(rec)
    return recs


def run_stud_tight() -> list[dict]:
    """Studentized (bootstrap-t) CI for the mean on R's shared replicates.

    The bootstrap-t pivot z*=(t*-t0)/se* is heavy-tailed for tiny/skewed n, so
    its extreme quantiles magnify the type-7-vs-norm.inter convention. Use a
    moderate seeded sample (dumped to R identically) where the pivot is stable,
    isolating the studentized FORMULA — which pystatistics reproduces exactly
    (verified separately to equal the textbook type-7 computation). Agreement
    with R is then to the quantile convention (relative)."""
    recs = []
    R = 5000
    # moderate, mildly-skewed seeded sample -> stable pivot; shared with R.
    data = np.random.default_rng(20260707).gamma(6.0, 2.0, 80).reshape(-1, 1)
    stat = mcdata.STAT["mean"]
    r = r_reference("boot_stud", data, statistic="mean", R=R, seed=707)
    idx = np.array(r["idx"], int).reshape(r["R"], r["n"]) - 1
    t_R = np.array(r["t"], float)
    var_R = np.array(r["var_t"], float)

    t_py = np.array([data[idx[b], 0].mean() for b in range(R)])
    var_py = np.array([data[idx[b], 0].var(ddof=1) / r["n"] for b in range(R)])
    c_t = arr_cmp(t_py, t_R)
    c_var = arr_cmp(var_py, var_R)

    sol = _shared_solution(data, stat, np.array([r["t0"]]), t_R)
    ci = boot_ci(sol, ci_type="stud", var_t=var_R, var_t0=float(r["var_t0"])).ci
    c_ci = arr_cmp(ci["stud"][0], r["ci_stud"])
    recs.append({
        "group": "stud_tight", "dataset": "seeded_gamma_n80", "statistic": "mean",
        "R": R, "t_stat": c_t, "var_t": c_var, "ci_stud": c_ci,
        "note": "studentized formula reproduces the textbook type-7 result "
                "exactly; agreement with R boot.ci is the quantile convention "
                "(norm.inter vs type-7), amplified in the pivot's tail",
        "pass": (within(c_t, abs_tol=TOL_MACHINE)
                 and within(c_var, abs_tol=TOL_MACHINE)
                 and within(c_ci, rel_tol=1e-2)),
    })
    return recs


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions(("boot", "stats"))

    boot_tight = run_boot_tight()
    stud_tight = run_stud_tight()

    run = build_run(
        env=env,
        config={"suite": "montecarlo-tight",
                "reference": "R boot::boot + boot::boot.ci on SHARED resample "
                "indices (genuine R boot, indices exported to pystatistics)",
                "tolerance_contract": "TIGHT — statistic/t0/bias/se/normal to "
                f"machine precision ({TOL_MACHINE}); basic/perc to the quantile "
                f"convention ({TOL_QUANTILE}); BCa z0 exact, acceleration a is a "
                "documented jackknife-vs-regression convention (calibration via "
                "coverage study)"},
        records=[{"key": "boot_tight", "checks": boot_tight},
                 {"key": "stud_tight", "checks": stud_tight}],
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    write_run(out, run)
    print(f"wrote {out}")
    npass = sum(bool(r.get("pass")) for r in boot_tight)
    print(f"  boot_tight ... {npass}/{len(boot_tight)} pass")
    sp = sum(bool(r.get("pass")) for r in stud_tight)
    print(f"  stud_tight ... {sp}/{len(stud_tight)} pass  "
          f"(ci_stud max_abs={stud_tight[0]['ci_stud']['max_abs']:.1e})")
    for r in boot_tight:
        print(f"    {r['dataset']}/{r['statistic']}: t_stat max_abs={r['t_stat']['max_abs']:.1e} "
              f"normal={r['ci_normal']['max_abs']:.1e} perc={r['ci_perc']['max_abs']:.1e} "
              f"bca_absdiff={r['ci_bca']['max_abs']:.2e} z0Δ={r['bca_z0']['abs']:.1e}")


if __name__ == "__main__":
    main()
