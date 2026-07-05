"""Red-team (R10) + fidelity (R15 / Guarantee 2) tier.

The hard cases a skeptic probes, plus the no-silent-substitution guarantees:

R10 hard cases:
  - small n bootstrap (n=6) vs R on shared resamples
  - a skewed statistic where percentile != BCa (the correction bites)
  - a boundary statistic (correlation near 1 — replicates pile at the edge)
  - permutation with ties (exact enumeration handles ties)
  - degenerate BCa (all-equal data -> zero jackknife acceleration): documented,
    no crash, no garbage
R15 default invocation:
  - bare boot()/permutation_test() with every default
Guarantee 2 fidelity:
  - two-sided permutation convention: |perm|>=|obs| is correct for a
    null-centered statistic (mean-difference) and matches exact enumeration, but
    it assumes null-centering — a non-centered statistic (ratio of means) is a
    documented footgun. Quantified here.
  - GPU opt-in (the 4.6.7 fix): backend='gpu' without gpu_statistic raises, a
    non-vectorizable config raises, a declared-but-wrong statistic raises — never
    a silent substitution.

Emits artifacts/montecarlo/v<ver>/runs/redteam.json (render-from-artifacts).

Run: MVNMLE_DATA_DIR=Dev/datasets python drivers/montecarlo/run_redteam.py
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcdata  # noqa: E402
from rref import r_reference, r_package_versions  # noqa: E402
from mccompare import arr_cmp, within, expect_raises  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402

from pystatistics.montecarlo import boot, boot_ci, permutation_test  # noqa: E402
from pystatistics.core.exceptions import ValidationError  # noqa: E402

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/montecarlo/v{ver}/runs/redteam.json")


def run_hard_cases() -> list[dict]:
    recs = []

    # (1) small n boot (n=6) vs R shared resamples
    data = mcdata.load("law").values[:6]
    r = r_reference("boot", data, statistic="corr", R=3000, seed=11)
    idx = np.array(r["idx"], int).reshape(r["R"], r["n"]) - 1
    t_py = np.array([mcdata.stat_corr(data, idx[b])[0] for b in range(r["R"])])
    c = arr_cmp(t_py, np.array(r["t"]))
    recs.append({"group": "hard", "case": "small_n=6", "statistic": "corr",
                 "t_stat_max_abs": c["max_abs"],
                 "pass": within(c, abs_tol=1e-10)})

    # (2) skewed statistic: percentile vs BCa separation (both computed; R shared)
    city = mcdata.load("city").values
    rr = r_reference("boot", city, statistic="ratio", R=8000, seed=22)
    idxr = np.array(rr["idx"], int).reshape(rr["R"], rr["n"]) - 1
    from run_tight import _shared_solution
    t_ratio = np.array([mcdata.stat_ratio(city, idxr[b])[0] for b in range(rr["R"])])
    sol = _shared_solution(city, mcdata.STAT["ratio"], np.array([rr["t0"]]), t_ratio)
    ci = boot_ci(sol, ci_type="all").ci
    perc, bca = ci["perc"][0], ci["bca"][0]
    sep = float(abs(perc[0] - bca[0]) + abs(perc[1] - bca[1]))
    width = float(perc[1] - perc[0])
    recs.append({"group": "hard", "case": "skew_perc_vs_bca", "statistic": "ratio",
                 "perc": perc.tolist(), "bca": bca.tolist(),
                 "endpoint_separation": sep, "rel_to_width": sep / width,
                 "note": "percentile != BCa here (skewed ratio) — BCa shifts the "
                         "interval; pystatistics computes both, perc matches R. "
                         "law/corr (tight tier) shows a much larger shift.",
                 "pass": sep / width > 0.01})   # correction materially present

    # (3) boundary statistic: correlation on near-perfectly-collinear data
    rng = np.random.default_rng(7)
    xb = rng.normal(0, 1, 20); yb = xb + rng.normal(0, 1e-3, 20)
    bdata = np.column_stack([xb, yb])
    rb = boot(bdata, mcdata.STAT["corr"], n_resamples=3000, seed=3)
    cib = boot_ci(rb, ci_type="all").ci
    upper_ok = cib["perc"][0][1] <= 1.0 + 1e-9   # correlation CI capped at 1
    recs.append({"group": "hard", "case": "boundary_corr~1", "statistic": "corr",
                 "t0": float(rb.t0[0]), "perc_ci": cib["perc"][0].tolist(),
                 "no_crash": True, "upper_within_1": bool(upper_ok),
                 "pass": bool(np.isfinite(rb.t0[0]) and upper_ok)})

    # (4) permutation with ties: coarse integer data -> many tied statistics
    tied = np.array([[1, 1], [1, 1], [2, 1], [2, 1], [3, 1],
                     [2, 2], [3, 2], [3, 2], [4, 2], [4, 2]], float)
    x = tied[tied[:, 1] == 1, 0]; y = tied[tied[:, 1] == 2, 0]
    exact = r_reference("perm_exact", tied, statistic="meandiff")
    pt = permutation_test(x, y, mcdata.mean_diff, n_resamples=100000, seed=8)
    recs.append({"group": "hard", "case": "perm_ties", "alternative": "two-sided",
                 "p_py": pt.p_value, "p_exact": exact["p_two_sided"],
                 "abs_diff": abs(pt.p_value - exact["p_two_sided"]),
                 "pass": abs(pt.p_value - exact["p_two_sided"]) < 0.01})

    # (5) degenerate BCa: all-equal data -> zero acceleration; documented
    const = np.full((12, 1), 3.5)
    rc = boot(const, mcdata.STAT["mean"], n_resamples=500, seed=1)
    outcome = expect_raises(lambda: boot_ci(rc, ci_type="bca"), ValidationError)
    # pystatistics does NOT raise here — it returns a degenerate [c, c] interval.
    dci = boot_ci(rc, ci_type="bca").ci["bca"][0]
    recs.append({"group": "hard", "case": "degenerate_bca_all_equal",
                 "t0": float(rc.t0[0]), "se": float(rc.se[0]),
                 "bca_ci": dci.tolist(),
                 "behavior": "returns degenerate [t0, t0] interval (a=0 fallback, "
                             "se=0); no crash, no garbage — documented",
                 "raised": outcome["raised"],
                 "pass": bool(np.allclose(dci, rc.t0[0]))})
    return recs


def _enum_two_sided(x, y, stat):
    """Full-enumeration two-sided p under two conventions (small n)."""
    combined = np.concatenate([x, y]); n = len(combined); n1 = len(x)
    obs = stat(x, y)
    stats = []
    for c in combinations(range(n), n1):
        mask = np.zeros(n, bool); mask[list(c)] = True
        stats.append(stat(combined[mask], combined[~mask]))
    stats = np.array(stats)
    p_abs = np.mean(np.abs(stats) >= abs(obs) - 1e-12)          # |T|>=|obs|
    p_tail = 2 * min(np.mean(stats >= obs - 1e-12),
                     np.mean(stats <= obs + 1e-12))             # 2*min tail
    return obs, float(p_abs), float(min(1.0, p_tail))


def run_fidelity() -> list[dict]:
    recs = []

    # (R15) default invocation — bare calls with every default must work
    law = mcdata.load("law").values
    b = boot(law, mcdata.STAT["corr"])                 # R=999, seed=None, cpu
    x, y = mcdata.two_sample("sleep")
    p = permutation_test(x, y, mcdata.mean_diff)       # R=9999, two-sided, cpu
    recs.append({"group": "fidelity", "case": "default_invocation",
                 "boot_R": b.R, "boot_backend": b.backend_name,
                 "perm_R": p.R, "perm_backend": p.backend_name,
                 "p_value_finite": bool(np.isfinite(p.p_value)),
                 "pass": bool(b.R == 999 and p.R == 9999
                              and "cpu" in b.backend_name
                              and np.isfinite(p.p_value))})

    # (G2) two-sided convention: null-centered (mean-diff) vs non-centered (ratio)
    xs = np.array([2.0, 3, 4, 5, 6]); ys = np.array([3.0, 4, 5, 6, 20])
    # null-centered: mean difference
    _, pa_md, pt_md = _enum_two_sided(xs, ys, lambda a, b: a.mean() - b.mean())
    py_md = permutation_test(xs, ys, mcdata.mean_diff, n_resamples=100000,
                             seed=1).p_value
    # non-centered: ratio of means (centered ~1, not 0)
    ratio = lambda a, b: a.mean() / b.mean()
    _, pa_ra, pt_ra = _enum_two_sided(xs, ys, ratio)
    py_ra = permutation_test(xs, ys, ratio, n_resamples=100000, seed=1).p_value
    recs.append({
        "group": "fidelity", "case": "two_sided_convention",
        "mean_diff": {"py": py_md, "exact_abs": pa_md, "exact_tail": pt_md,
                      "py_matches_abs": abs(py_md - pa_md) < 0.01},
        "ratio_noncentered": {"py": py_ra, "exact_abs": pa_ra, "exact_tail": pt_ra,
                              "py_matches_abs": abs(py_ra - pa_ra) < 0.01,
                              "abs_vs_tail_gap": abs(pa_ra - pt_ra)},
        "note": "permutation_test two-sided uses |perm|>=|obs|; correct & matches "
                "exact for a null-centered statistic (mean-diff). For a "
                "non-centered statistic (ratio ~1) the |.| convention diverges "
                "from the proper 2*min-tail two-sided — a documented footgun: "
                "pass a null-centered statistic (a difference), not a ratio.",
        # pass = pystatistics correctly implements its stated |.|-convention and
        # it is right for the intended (null-centered) use.
        "pass": bool(abs(py_md - pa_md) < 0.01),
    })

    # (G2 / R16 fix) GPU opt-in fail-loud — no silent substitution
    d = np.random.default_rng(0).gamma(2, 3, 200)
    med = mcdata.STAT["median"]
    checks = {
        "gpu_missing_flag_raises": expect_raises(
            lambda: boot(d, mcdata.STAT["mean"], n_resamples=99, seed=1,
                         backend="gpu"), ValidationError),
        "gpu_declared_but_median_raises": expect_raises(
            lambda: boot(d, med, n_resamples=99, seed=1, backend="gpu",
                         gpu_statistic="mean"), ValidationError),
        "gpu_balanced_raises": expect_raises(
            lambda: boot(d, mcdata.STAT["mean"], n_resamples=99, seed=1,
                         backend="gpu", gpu_statistic="mean", method="balanced"),
            ValidationError),
        "perm_gpu_missing_flag_raises": expect_raises(
            lambda: permutation_test(x, y, mcdata.mean_diff, n_resamples=99,
                                     seed=1, backend="gpu"), ValidationError),
    }
    cpu_default = boot(d, med, n_resamples=99, seed=1)   # default unaffected
    recs.append({"group": "fidelity", "case": "gpu_optin_fail_loud",
                 **{k: v["pass"] for k, v in checks.items()},
                 "cpu_default_backend": cpu_default.backend_name,
                 "pass": bool(all(v["pass"] for v in checks.values())
                              and "cpu" in cpu_default.backend_name)})
    return recs


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions(("boot", "stats"))

    hard = run_hard_cases()
    fidelity = run_fidelity()

    run = build_run(
        env=env,
        config={"suite": "montecarlo-redteam",
                "reference": "R boot / exact enumeration; fail-loud probes",
                "tolerance_contract": "hard cases match R on shared resamples or "
                "exact enumeration; fidelity = no silent substitution (G2), "
                "documented conventions (two-sided |.|, BCa acceleration)"},
        records=[{"key": "hard_cases", "checks": hard},
                 {"key": "fidelity", "checks": fidelity}],
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    write_run(out, run)
    print(f"wrote {out}")
    for grp, name in [(hard, "hard"), (fidelity, "fidelity")]:
        npass = sum(bool(r.get("pass")) for r in grp)
        print(f"  {name:10s}... {npass}/{len(grp)} pass")
        for r in grp:
            print(f"    {r['case']:26s} pass={r['pass']}")


if __name__ == "__main__":
    main()
