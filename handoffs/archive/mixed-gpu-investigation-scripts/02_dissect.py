"""
Phase A.2: dissect the runtime.

(1) Confirm that the fixed-effect SE can be computed from the p x p factor RX
    (already returned by solve_pls) instead of the dense n x n solve in
    _compute_se. By Woodbury, X'V^-1 X = RtR = RX RX', so
        Var(beta) = sigma^2 (RX RX')^-1.
    Verify it matches the library's _compute_se to ~1e-10.

(2) Isolate the GPU-addressable cost: time the theta-optimization loop ALONE
    (build Z + scipy.minimize over profiled_deviance_lmm), separately from
    _compute_se and satterthwaite. This tells us how big the prize is for a
    GPU that only accelerates the deviance loop.
"""
from __future__ import annotations
import time, gc
import numpy as np
from scipy.optimize import minimize

from pystatistics.mixed._random_effects import (
    parse_random_effects, build_z_matrix, build_lambda,
    theta_lower_bounds, theta_start,
)
from pystatistics.mixed._pls import solve_pls
from pystatistics.mixed._deviance import profiled_deviance_lmm
from pystatistics.mixed.solvers import _compute_se
from pystatistics.mixed.design import MixedDesign


def gen_single_factor(n_groups, n_per, seed=0, slope=False):
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    x1 = rng.standard_normal(n)
    b0 = rng.standard_normal(n_groups) * 0.8
    beta = np.array([1.0, 2.0])
    X = np.column_stack([np.ones(n), x1])
    eta = X @ beta + b0[group]
    if slope:
        b1 = rng.standard_normal(n_groups) * 0.5
        eta = eta + b1[group] * x1
    y = eta + rng.standard_normal(n) * 1.0
    re = {"g": ["1", "x1"]} if slope else None
    rd = {"x1": x1} if slope else None
    return y, X, {"g": group}, re, rd


def se_from_rx(pls):
    """Cheap p x p SE: sigma^2 (RX RX')^-1, RX lower-triangular Cholesky."""
    RX_inv = np.linalg.inv(pls.RX)              # p x p
    vcov = pls.sigma_sq * (RX_inv.T @ RX_inv)   # (RX RX')^-1 = RX^-T RX^-1
    return np.sqrt(np.maximum(np.diag(vcov), 0.0))


def dissect(label, y, X, groups, re, rd):
    design = MixedDesign.validate(
        np.asarray(y, float), np.asarray(X, float), groups, re, rd)
    specs = parse_random_effects(design.groups, design.random_effects,
                                 design.random_data, design.n)
    Z = build_z_matrix(specs)
    theta0 = theta_start(specs)
    lb = theta_lower_bounds(specs)
    bounds = [(lb[i], None) for i in range(len(theta0))]
    n, p = design.X.shape
    q = Z.shape[1]

    # --- time the optimization loop alone ---
    gc.collect()
    n_eval = [0]
    def obj(th):
        n_eval[0] += 1
        return profiled_deviance_lmm(th, design.X, Z, design.y, specs, True)
    t0 = time.perf_counter()
    opt = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-7})
    t_opt = time.perf_counter() - t0

    theta_hat = opt.x
    Lam = build_lambda(theta_hat, specs)

    # --- time final solve ---
    t0 = time.perf_counter()
    pls = solve_pls(design.X, Z, design.y, Lam, reml=True)
    t_final = time.perf_counter() - t0

    # --- time the two SE methods + correctness ---
    t0 = time.perf_counter()
    se_dense = _compute_se(pls, design.X, Z, Lam, p)
    t_se_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    se_cheap = se_from_rx(pls)
    t_se_cheap = time.perf_counter() - t0

    rel = np.max(np.abs(se_cheap - se_dense) / (np.abs(se_dense) + 1e-30))

    print(f"[{label}] n={n} q={q} p={p} nfev={n_eval[0]}")
    print(f"    optimize_loop = {t_opt:8.3f}s   ({t_opt/max(n_eval[0],1)*1000:.1f} ms/deviance-eval)")
    print(f"    final_solve   = {t_final:8.3f}s")
    print(f"    SE dense(nxn) = {t_se_dense:8.3f}s   <-- current library")
    print(f"    SE cheap(RX)  = {t_se_cheap:8.4f}s  speedup={t_se_dense/max(t_se_cheap,1e-9):.0f}x  rel_err={rel:.2e}")
    print()


def main():
    print("Single-factor random INTERCEPT:")
    for G in [200, 500, 1000, 2000]:
        y, X, groups, re, rd = gen_single_factor(G, 20, seed=1)
        dissect(f"int G={G}", y, X, groups, re, rd)
    print("Single-factor random intercept+SLOPE:")
    for G in [200, 500, 1000]:
        y, X, groups, re, rd = gen_single_factor(G, 20, seed=2, slope=True)
        dissect(f"slope G={G}", y, X, groups, re, rd)


if __name__ == "__main__":
    main()
