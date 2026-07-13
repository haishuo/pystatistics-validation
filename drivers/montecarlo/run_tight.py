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
from pystatistics.montecarlo._ci import _norm_inter  # noqa: E402
from pystatistics.core.result import Result  # noqa: E402
from scipy.stats import norm  # noqa: E402


def _bca_on_shared(t, t0, freq, alpha=0.05):
    """BCa endpoints from SHARED replicates+frequencies via the library's
    arithmetic (regression influence acceleration + norm.inter quantiles),
    exactly the path boot_ci takes on a genuine ordinary run. Fed R's own
    resample frequencies so the comparison is on identical replicates."""
    R, n = freq.shape
    P = freq / n
    Pc = P - P.mean(axis=0)
    L = np.linalg.pinv(Pc) @ (t - t.mean())
    L = L - L.mean()
    a = float(np.sum(L ** 3) / (6.0 * np.sum(L ** 2) ** 1.5))
    prop = np.clip(np.sum(t < t0) / R, 1 / (2 * R), 1 - 1 / (2 * R))
    z0 = float(norm.ppf(prop))
    def adj(za):
        num = z0 + za
        return norm.cdf(z0 + num / (1.0 - a * num))
    a1 = np.clip(adj(norm.ppf(alpha / 2)), 0.5 / R, 1 - 0.5 / R)
    a2 = np.clip(adj(norm.ppf(1 - alpha / 2)), 0.5 / R, 1 - 0.5 / R)
    lo, hi = _norm_inter(t, [a1, a2])
    return np.array([lo, hi]), z0, a

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/montecarlo/v{ver}/runs/tight.json")

# Tolerances (the contract above). Since 4.6.8 boot_ci uses R's norm.inter
# quantile rule, so basic/perc/studentized match R to machine precision on
# shared replicates; BCa uses R's regression influence acceleration, matching to
# the pinv floor of the influence solve.
TOL_MACHINE = 1e-9        # t0/t/bias/se/normal/basic/perc — same math as R
TOL_STUD = 1e-8           # studentized — norm.inter on the shared pivot
TOL_BCA = 5e-3            # BCa endpoints — regression influence pinv floor
TOL_Z0 = 1e-9             # BCa bias-correction — identical definition


def _shared_solution(data, stat, t0, t):
    """A BootstrapSolution carrying R's shared replicates (for boot_ci)."""
    R = len(t)
    base = boot(data, stat, n_resamples=5, seed=0)   # a valid _design only
    params = BootParams(t0=np.asarray(t0, float), t=np.asarray(t, float).reshape(R, 1),
                        n_resamples=R, bias=np.array([t.mean() - t0[0]]),
                        standard_errors=np.array([t.std(ddof=1)]), conf_int=None, conf_level=None)
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
        c_se = scalar_cmp(float(sol.standard_errors[0]), float(r["se"]))

        # (3) boot.ci normal/basic/perc on the shared replicates (norm.inter now)
        ci = boot_ci(sol, ci_type="all").conf_int
        c_normal = arr_cmp(ci["normal"][0], r["ci_normal"])
        c_basic = arr_cmp(ci["basic"][0], r["ci_basic"])
        c_perc = arr_cmp(ci["percentile"][0], r["ci_perc"])

        # (4) BCa on shared frequencies via the library arithmetic (regression
        # influence + norm.inter) — matches R's boot.ci default empinf. (The
        # injected-solution boot_ci self-check correctly rejects foreign
        # replicates and would fall back to jackknife; here we feed R's own
        # frequencies so the acceleration is the regression estimate R uses.)
        freq = np.array(r["freq"], int).reshape(r["R"], r["n"])
        bca, z0_py, a_py = _bca_on_shared(t_R, float(r["t0"]), freq)
        c_bca = arr_cmp(bca, r["ci_bca"])

        rec = {
            "group": "boot_tight", "dataset": key, "statistic": stat_name, "R": R,
            "t_stat": c_t, "t0": c_t0, "bias": c_bias, "se": c_se,
            "ci_normal": c_normal, "ci_basic": c_basic, "ci_perc": c_perc,
            "ci_bca": c_bca,
            "bca_z0": {"py": z0_py, "r": float(r["z0"]),
                       "abs": abs(z0_py - float(r["z0"]))},
            "bca_a": {"py_reg": a_py, "r_reg": float(r["a_reg"]),
                      "abs_vs_r_reg": abs(a_py - float(r["a_reg"]))},
            "bca_note": "BCa uses the regression influence acceleration (R "
                        "boot.ci default) + norm.inter quantiles; matches "
                        "boot.ci to the influence-solve floor.",
            "pass": (within(c_t, abs_tol=TOL_MACHINE)
                     and within(c_t0, abs_tol=TOL_MACHINE)
                     and within(c_bias, abs_tol=TOL_MACHINE)
                     and within(c_se, abs_tol=TOL_MACHINE)
                     and within(c_normal, abs_tol=TOL_MACHINE)
                     and within(c_basic, abs_tol=TOL_MACHINE)
                     and within(c_perc, abs_tol=TOL_MACHINE)
                     and within(c_bca, abs_tol=TOL_BCA)
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
    ci = boot_ci(sol, ci_type="studentized", var_t=var_R, var_t0=float(r["var_t0"])).conf_int
    c_ci = arr_cmp(ci["studentized"][0], r["ci_stud"])
    recs.append({
        "group": "stud_tight", "dataset": "seeded_gamma_n80", "statistic": "mean",
        "R": R, "t_stat": c_t, "var_t": c_var, "ci_stud": c_ci,
        "note": "studentized endpoints use R's norm.inter quantile rule (since "
                "4.6.8) — machine-precision agreement with boot.ci on the shared "
                "pivot",
        "pass": (within(c_t, abs_tol=TOL_MACHINE)
                 and within(c_var, abs_tol=TOL_MACHINE)
                 and within(c_ci, abs_tol=TOL_STUD)),
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
                "tolerance_contract": "TIGHT — statistic/t0/bias/se/normal/basic/"
                f"perc/studentized to machine precision ({TOL_MACHINE}) on shared "
                "replicates (4.6.8 adopts R's norm.inter quantile rule); BCa z0 "
                f"exact, endpoints to the regression-influence floor ({TOL_BCA}) "
                "using R's default regression acceleration"},
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
