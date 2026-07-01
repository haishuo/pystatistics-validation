"""
Phase A: profile where lmm() spends time and where the DENSE impl falls over.

Two regimes:
  (1) single grouping factor, many small groups, random intercept
      -> q = G. Z'Z is block-diagonal (here diagonal). The library forms it DENSE.
  (2) small crossed design (students x instructors) -> q = G1 + G2.

We report: n, q, total wall time, per-section timing (from Result.timing),
peak memory (tracemalloc), and a cProfile top-20 for one representative size.

Deterministic: numpy default_rng(seed).
"""
from __future__ import annotations
import sys, time, gc, tracemalloc, cProfile, pstats, io
import numpy as np
from pystatistics.mixed import lmm


def gen_single_factor(n_groups, n_per, seed=0, slope=False):
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    x1 = rng.standard_normal(n)
    # true params
    b0 = rng.standard_normal(n_groups) * 0.8          # random intercept
    beta = np.array([1.0, 2.0])
    X = np.column_stack([np.ones(n), x1])
    eta = X @ beta + b0[group]
    if slope:
        b1 = rng.standard_normal(n_groups) * 0.5
        eta = eta + b1[group] * x1
    y = eta + rng.standard_normal(n) * 1.0
    groups = {"g": group}
    re = {"g": ["1", "x1"]} if slope else None
    rd = {"x1": x1} if slope else None
    return y, X, groups, re, rd


def gen_crossed(n_a, n_b, n_obs, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_a, n_obs)
    b = rng.integers(0, n_b, n_obs)
    x1 = rng.standard_normal(n_obs)
    ba = rng.standard_normal(n_a) * 0.7
    bb = rng.standard_normal(n_b) * 0.5
    beta = np.array([1.0, 1.5])
    X = np.column_stack([np.ones(n_obs), x1])
    y = X @ beta + ba[a] + bb[b] + rng.standard_normal(n_obs) * 1.0
    return y, X, {"a": a, "b": b}, None, None


def run_one(label, y, X, groups, re, rd, satter=False):
    q = 0
    for name, g in groups.items():
        J = len(np.unique(g))
        terms = 1 if (re is None or name not in re) else len(re[name])
        q += J * terms
    n = len(y)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        sol = lmm(y, X, groups, random_effects=re, random_data=rd,
                  compute_satterthwaite=satter)
        dt = time.perf_counter() - t0
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tim = sol._result.timing
        # timing sections
        secs = {}
        try:
            for s in tim.sections:
                secs[s.name] = s.elapsed
        except Exception:
            secs = dict(getattr(tim, "_sections", {}) or {})
        info = sol._result.info
        print(f"[{label}] n={n} q={q} | total={dt:.3f}s peakRAM={peak/1e6:.0f}MB "
              f"| n_iter={info.get('n_iter')} dev={info.get('deviance'):.2f}")
        sec_str = "  ".join(f"{k}={v:.3f}" for k, v in secs.items())
        print(f"    sections: {sec_str}")
        return dt, peak, q
    except Exception as e:
        dt = time.perf_counter() - t0
        tracemalloc.stop()
        print(f"[{label}] n={n} q={q} | FAILED after {dt:.1f}s: {type(e).__name__}: {str(e)[:120]}")
        return None, None, q


def main():
    print("=" * 70)
    print("REGIME 1: single factor, random intercept, growing #groups")
    print("=" * 70)
    for G in [50, 200, 500, 1000, 2000]:
        y, X, groups, re, rd = gen_single_factor(G, 20, seed=1)
        run_one(f"single-int G={G}", y, X, groups, re, rd, satter=False)

    print()
    print("=" * 70)
    print("REGIME 1b: single factor, random intercept+slope, growing #groups")
    print("=" * 70)
    for G in [50, 200, 500, 1000]:
        y, X, groups, re, rd = gen_single_factor(G, 20, seed=2, slope=True)
        run_one(f"single-slope G={G}", y, X, groups, re, rd, satter=False)

    print()
    print("=" * 70)
    print("REGIME 2: crossed (a x b), growing size")
    print("=" * 70)
    for (na, nb, nobs) in [(50, 50, 2000), (200, 100, 8000),
                           (500, 300, 20000), (1000, 500, 40000)]:
        y, X, groups, re, rd = gen_crossed(na, nb, nobs, seed=3)
        run_one(f"crossed {na}x{nb}", y, X, groups, re, rd, satter=False)

    print()
    print("=" * 70)
    print("cProfile: single factor G=1000 (intercept only), satter=False")
    print("=" * 70)
    y, X, groups, re, rd = gen_single_factor(1000, 20, seed=1)
    pr = cProfile.Profile()
    pr.enable()
    lmm(y, X, groups, compute_satterthwaite=False)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    main()
