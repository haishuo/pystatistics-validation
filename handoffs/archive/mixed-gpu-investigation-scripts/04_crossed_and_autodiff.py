"""
Phase B.2: the CROSSED regime (Z'Z NOT block-diagonal) + AUTODIFF (rock #4).

PART A - crossed, intercept-only factors (e.g. students x instructors):
  The batched per-group trick does NOT apply (groups share columns across
  factors). The correct CPU baseline is a SPARSE Cholesky of M = ZLam'ZLam + I
  (lme4 / MixedModels.jl approach). We implement it with scipy.sparse + splu
  (log|det M| from sum log|diag(U)|) and show it runs the sizes the dense
  library OOMs on, fast, on CPU fp64. This tells us whether the crossed case
  needs a GPU (vs. just a sparse CPU path the library lacks).

PART B - autodiff: the batched torch deviance is differentiable. Compare
  L-BFGS-B with an EXACT autograd gradient vs scipy's finite-difference
  gradient: #function evals and wall time. Independent of GPU.
"""
from __future__ import annotations
import time, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
import torch

from pystatistics.mixed._random_effects import parse_random_effects, build_z_matrix
from pystatistics.mixed._deviance import profiled_deviance_lmm


# ============================ PART A: crossed sparse ============================
def gen_crossed(n_a, n_b, n_obs, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_a, n_obs)
    b = rng.integers(0, n_b, n_obs)
    x1 = rng.standard_normal(n_obs)
    ba = rng.standard_normal(n_a) * 0.7
    bb = rng.standard_normal(n_b) * 0.5
    X = np.column_stack([np.ones(n_obs), x1])
    y = X @ np.array([1.0, 1.5]) + ba[a] + bb[b] + rng.standard_normal(n_obs)
    return y, X, {"a": a, "b": b}


def build_Z_sparse(groups):
    """Sparse Z for intercept-only crossed factors; returns (Z, factor_sizes)."""
    n = len(next(iter(groups.values())))
    blocks = []
    sizes = []
    for name, g in groups.items():
        g = np.asarray(g)
        _, ids = np.unique(g, return_inverse=True)
        J = ids.max() + 1
        Zk = sp.csc_matrix((np.ones(n), (np.arange(n), ids)), shape=(n, J))
        blocks.append(Zk); sizes.append(J)
    return sp.hstack(blocks, format="csc"), sizes


def deviance_crossed_sparse(theta, Z, sizes, X, y, reml=True, timed=False):
    """theta: one diagonal Cholesky scalar per factor (intercept-only)."""
    n, p = X.shape
    lam_vec = np.concatenate([np.full(J, theta[k]) for k, J in enumerate(sizes)])
    ZLam = Z.multiply(lam_vec[np.newaxis, :]).tocsc()      # column scaling
    q = ZLam.shape[1]
    M = (ZLam.T @ ZLam + sp.identity(q, format="csc")).tocsc()
    t0 = time.perf_counter()
    lu = splu(M)                                            # sparse Cholesky-equiv
    t_fac = time.perf_counter() - t0
    logdetM = np.sum(np.log(np.abs(lu.U.diagonal())))      # = log|det M| = log|L|^2
    ZLtX = (ZLam.T @ X)                                     # (q,p) dense-ish
    ZLty = (ZLam.T @ y)                                     # (q,)
    W = np.column_stack([lu.solve(ZLtX[:, j]) for j in range(p)])  # M^-1 ZLam'X
    m_y = lu.solve(ZLty)                                    # M^-1 ZLam'y
    RtR = X.T @ X - ZLtX.T @ W
    rhs = X.T @ y - ZLtX.T @ m_y
    beta = np.linalg.solve(RtR, rhs)
    RX = np.linalg.cholesky(RtR)
    logdetRX = 2.0 * np.sum(np.log(np.abs(np.diag(RX))))
    u = m_y - W @ beta
    b = lam_vec * u
    resid = y - X @ beta - Z @ b
    pwrss = float(resid @ resid + u @ u)
    if reml:
        df = n - p
        dev = logdetM + logdetRX + df * (1 + np.log(2 * np.pi * pwrss / df))
    else:
        dev = logdetM + n * (1 + np.log(2 * np.pi * pwrss / n))
    if timed:
        return float(dev), t_fac, q, M.nnz
    return float(dev)


def part_a():
    print("=" * 78)
    print("PART A: CROSSED regime - sparse CPU vs dense library")
    print("=" * 78)
    # validate on a small crossed case against the dense library
    y, X, groups = gen_crossed(40, 30, 1500, seed=11)
    specs = parse_random_effects(groups, None, None, len(y))
    Zd = build_z_matrix(specs)
    theta = np.array([0.6, 0.9])
    dev_lib = profiled_deviance_lmm(theta, X, Zd, y, specs, reml=True)
    Zs, sizes = build_Z_sparse(groups)
    dev_sp = deviance_crossed_sparse(theta, Zs, sizes, X, y, reml=True)
    print(f"VALIDATION (40x30): library={dev_lib:.6f} sparse={dev_sp:.6f} "
          f"d={abs(dev_lib-dev_sp):.2e}\n")

    print(f"{'design':>14} {'n':>8} {'q':>7} {'nnz(M)':>10} | {'dense lib':>10} {'sparse CPU/eval':>16}")
    cases = [(50, 50, 2000), (200, 100, 8000), (500, 300, 20000),
             (1000, 500, 40000), (3000, 1500, 100000), (5000, 3000, 200000)]
    for na, nb, nobs in cases:
        y, X, groups = gen_crossed(na, nb, nobs, seed=na)
        Zs, sizes = build_Z_sparse(groups)
        # sparse eval timing (3x)
        t0 = time.perf_counter()
        for _ in range(3):
            dev, tfac, q, nnz = deviance_crossed_sparse(np.array([0.6, 0.9]), Zs, sizes, X, y, timed=True)
        t_sp = (time.perf_counter() - t0) / 3 * 1000
        # dense library only where it won't OOM (q <= ~1500)
        if q <= 1600:
            specs = parse_random_effects(groups, None, None, len(y))
            Zd = build_z_matrix(specs)
            t0 = time.perf_counter()
            profiled_deviance_lmm(np.array([0.6, 0.9]), X, Zd, y, specs, reml=True)
            t_lib = (time.perf_counter() - t0) * 1000
            libs = f"{t_lib:10.1f}"
        else:
            libs = f"{'OOM/skip':>10}"
        print(f"{na:>5}x{nb:<8} {nobs:>8} {q:>7} {nnz:>10} | {libs} {t_sp:>14.2f}ms")


# ============================ PART B: autodiff ============================
def gen_single_factor(n_groups, n_per, q_t, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    slopes = [rng.standard_normal(n) for _ in range(q_t - 1)]
    X = np.column_stack([np.ones(n)] + slopes)
    eta = X @ np.arange(1, q_t + 1, dtype=float)
    for t in range(q_t):
        b = rng.standard_normal(n_groups) * (0.8 if t == 0 else 0.4)
        col = np.ones(n) if t == 0 else slopes[t - 1]
        eta = eta + b[group] * col
    y = eta + rng.standard_normal(n)
    return y, X, group, slopes


# reuse batched deviance from script 03
sys.path.insert(0, "/home/haishuo/build-scratch/mixedgpu/scripts")
from importlib import import_module
b3 = import_module("03_batched_deviance")


def part_b():
    print()
    print("=" * 78)
    print("PART B: AUTODIFF gradient vs finite differences (CPU fp64)")
    print("=" * 78)
    q_t = 3
    for G in [500, 2000]:
        y, X, group, slopes = gen_single_factor(G, 20, q_t, seed=5)
        packed = b3.make_batched_tensors(y, X, slopes, G, 20, q_t, "cpu", torch.float64)
        n_theta = q_t * (q_t + 1) // 2
        theta0 = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0][:n_theta])
        bounds = [(0, None) if i in _diag_idx(q_t) else (None, None) for i in range(n_theta)]

        # --- finite-diff (scipy default) ---
        ev_fd = [0]
        def f_fd(th):
            ev_fd[0] += 1
            t = torch.tensor(th, dtype=torch.float64)
            return float(b3.deviance_batched(t, packed, q_t))
        t0 = time.perf_counter()
        r_fd = minimize(f_fd, theta0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-7})
        t_fd = time.perf_counter() - t0

        # --- autograd exact gradient ---
        ev_ad = [0]
        def f_ad(th):
            ev_ad[0] += 1
            t = torch.tensor(th, dtype=torch.float64, requires_grad=True)
            dev = b3.deviance_batched(t, packed, q_t)
            g, = torch.autograd.grad(dev, t)
            return float(dev), g.numpy()
        t0 = time.perf_counter()
        r_ad = minimize(f_ad, theta0, method="L-BFGS-B", jac=True, bounds=bounds,
                        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-7})
        t_ad = time.perf_counter() - t0

        print(f"G={G} (n_theta={n_theta}):")
        print(f"  finite-diff: nfev={ev_fd[0]:3d}  time={t_fd*1000:7.1f}ms  dev={r_fd.fun:.5f}")
        print(f"  autograd   : nfev={ev_ad[0]:3d}  time={t_ad*1000:7.1f}ms  dev={r_ad.fun:.5f}")
        print(f"  theta agreement: {np.max(np.abs(r_fd.x - r_ad.x)):.2e}\n")


def _diag_idx(q_t):
    idx = []; k = 0
    for r in range(q_t):
        for c in range(r + 1):
            if r == c:
                idx.append(k)
            k += 1
    return set(idx)


if __name__ == "__main__":
    part_a()
    part_b()
