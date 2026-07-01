"""
Phase B.4 (rock #5): GLMM / PIRLS - is it a better GPU candidate than LMM?

GLMM's inner loop is PIRLS = iterated penalized WLS, i.e. solve_pls called
~10-25 times per deviance eval (each with updated weights). So per outer
theta step it is strictly MORE of the SAME dense kernel the LMM benchmark
already covered. We confirm by profiling glmm() and checking that (a) the
cost concentrates in the same dense PLS kernel and (b) the GLMM SE path does
NOT have the n x n defect (it already uses RX).
"""
from __future__ import annotations
import time, cProfile, pstats, io
import numpy as np
from pystatistics.mixed import glmm


def gen_glmm_single(n_groups, n_per, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    x1 = rng.standard_normal(n)
    b0 = rng.standard_normal(n_groups) * 0.7
    eta = -0.2 + 0.8 * x1 + b0[group]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(float)
    X = np.column_stack([np.ones(n), x1])
    return y, X, {"g": group}


for G in [200, 500, 1000]:
    y, X, groups = gen_glmm_single(G, 20, seed=G)
    t0 = time.perf_counter()
    sol = glmm(y, X, groups, family="binomial")
    dt = time.perf_counter() - t0
    print(f"glmm G={G} n={len(y)} q={G}: total={dt:.3f}s converged={sol._result.info.get('converged')} "
          f"pirls_iter={sol._result.info.get('pirls_iter')}")

print("\ncProfile glmm G=500:")
y, X, groups = gen_glmm_single(500, 20, seed=99)
pr = cProfile.Profile(); pr.enable()
glmm(y, X, groups, family="binomial")
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
print(s.getvalue())
