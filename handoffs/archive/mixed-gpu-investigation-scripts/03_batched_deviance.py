"""
Phase B core experiment: structure-exploiting BATCHED per-group profiled
deviance for a single grouping factor, vs the library's dense formulation.

For ONE grouping factor with J groups of q_t random-effect terms each, the
Gram (Lambda' Z' Z Lambda + I) is BLOCK-DIAGONAL with J blocks of size
q_t x q_t (observations in different groups share no RE columns). So the
expensive dense q x q Cholesky the library performs is, mathematically, a
batched [J, q_t, q_t] Cholesky -> the GPU-favorable regime.

This script:
  (1) implements the batched deviance in torch (CPU/CUDA, fp32/fp64),
  (2) VALIDATES it against pystatistics' dense profiled_deviance_lmm,
  (3) implements a faithful DENSE torch deviance too (for the dense GPU test),
  (4) CF-1 check: smallest eigenvalue of the per-group Gram + deviance, fp32 vs fp64,
  (5) benchmarks per-evaluation wall time across backends and group counts.

Equal group sizes are used for clean batching; ragged groups would use
padding or a segment reduction (noted in the report).
"""
from __future__ import annotations
import time, sys
import numpy as np
import torch

from pystatistics.mixed._random_effects import (
    parse_random_effects, build_z_matrix, build_lambda,
)
from pystatistics.mixed._deviance import profiled_deviance_lmm


# ----------------------------- data -----------------------------
def gen_single_factor(n_groups, n_per, q_t, seed=0):
    """q_t random-effect terms: intercept + (q_t-1) slopes."""
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    p = q_t  # fixed effects: intercept + (q_t-1) covariates (reuse slope vars)
    slopes = [rng.standard_normal(n) for _ in range(q_t - 1)]
    Xcols = [np.ones(n)] + slopes
    X = np.column_stack(Xcols)
    beta = np.arange(1, p + 1, dtype=float)
    eta = X @ beta
    for t in range(q_t):
        b = rng.standard_normal(n_groups) * (0.8 if t == 0 else 0.4)
        col = np.ones(n) if t == 0 else slopes[t - 1]
        eta = eta + b[group] * col
    y = eta + rng.standard_normal(n)
    terms = ["1"] + [f"s{t}" for t in range(q_t - 1)]
    re = {"g": terms}
    rd = {f"s{t}": slopes[t] for t in range(q_t - 1)}
    return y, X, {"g": group}, re, rd, slopes


def theta_to_T(theta, q_t):
    T = torch.zeros((q_t, q_t), dtype=theta.dtype, device=theta.device)
    idx = 0
    for r in range(q_t):
        for c in range(r + 1):
            T[r, c] = theta[idx]; idx += 1
    return T


# ----------------- batched (structure-exploiting) ----------------
def make_batched_tensors(y, X, slopes, n_groups, n_per, q_t, device, dtype):
    n = n_groups * n_per
    p = X.shape[1]
    # RE design columns per obs: [1, s0, s1, ...]
    re_cols = [np.ones(n)] + slopes[: q_t - 1]
    Zfull = np.column_stack(re_cols)             # (n, q_t)
    Zblk = Zfull.reshape(n_groups, n_per, q_t)   # (J, m, q_t)
    Xblk = X.reshape(n_groups, n_per, p)         # (J, m, p)
    yblk = y.reshape(n_groups, n_per)            # (J, m)
    t = lambda a: torch.tensor(a, dtype=dtype, device=device)
    return (t(Zblk), t(Xblk), t(yblk), t(X), t(y), n, p)


def deviance_batched(theta, packed, q_t, reml=True, return_diag=False):
    Zblk, Xblk, yblk, Xf, yf, n, p = packed
    T = theta_to_T(theta, q_t)
    J = Zblk.shape[0]
    I = torch.eye(q_t, dtype=Zblk.dtype, device=Zblk.device).expand(J, q_t, q_t)
    A = Zblk.transpose(1, 2) @ Zblk                  # (J,q,q)  Z_j'Z_j
    M = T.T @ A @ T + I                               # (J,q,q)  block Gram
    L = torch.linalg.cholesky(M)                      # (J,q,q)  batched chol
    ZtX = Zblk.transpose(1, 2) @ Xblk                # (J,q,p)
    Zty = (Zblk.transpose(1, 2) @ yblk.unsqueeze(2)) # (J,q,1)
    ZLtX = T.T @ ZtX                                 # (J,q,p)
    ZLty = T.T @ Zty                                 # (J,q,1)
    CX = torch.linalg.solve_triangular(L, ZLtX, upper=False)   # (J,q,p)
    cu = torch.linalg.solve_triangular(L, ZLty, upper=False)   # (J,q,1)
    XtX = Xf.T @ Xf                                  # (p,p)
    RtR = XtX - (CX.transpose(1, 2) @ CX).sum(0)     # (p,p)
    Xty = Xf.T @ yf                                  # (p,)
    rhs = Xty - (CX.transpose(1, 2) @ cu).squeeze(2).sum(0)    # (p,)
    RX = torch.linalg.cholesky(RtR)                  # (p,p)
    beta = torch.cholesky_solve(rhs.unsqueeze(1), RX).squeeze(1)
    # u per group: M_j u_j = ZLam'(y - Xβ)_j ; rhs_j = ZLty_j - ZLtX_j β
    rhs_u = (ZLty.squeeze(2) - (ZLtX @ beta))        # (J,q)
    u = torch.cholesky_solve(rhs_u.unsqueeze(2),
                             L).squeeze(2)            # (J,q)  via M=LL'
    b = (T @ u.unsqueeze(2)).squeeze(2)              # (J,q) conditional modes
    fit_fixed = Xblk @ beta                          # (J,m)
    fit_rand = (Zblk @ b.unsqueeze(2)).squeeze(2)    # (J,m)
    resid = yblk - fit_fixed - fit_rand
    wrss = (resid ** 2).sum()
    uu = (u ** 2).sum()
    pwrss = wrss + uu
    logdetL = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2).clamp_min(1e-30)).sum()
    logdetRX = 2.0 * torch.log(torch.diagonal(RX).abs().clamp_min(1e-30)).sum()
    if reml:
        df = n - p
        dev = logdetL + logdetRX + df * (1.0 + torch.log(2 * torch.pi * pwrss / df))
    else:
        dev = logdetL + n * (1.0 + torch.log(2 * torch.pi * pwrss / n))
    if return_diag:
        # CF-1: min eigenvalue across all per-group Gram blocks
        eig = torch.linalg.eigvalsh(M).min()
        return float(dev), float(eig), float(pwrss)
    return dev


# ----------------------- dense torch (faithful) ------------------
def deviance_dense_torch(theta, Xt, Zt, yt, q_t, reml=True, return_diag=False):
    # Build dense Lambda = T (x) I_J via kron-like placement
    J = Zt.shape[1] // q_t
    T = theta_to_T(theta, q_t)
    q = Zt.shape[1]
    Lam = torch.zeros((q, q), dtype=Zt.dtype, device=Zt.device)
    for r in range(q_t):
        for c in range(r + 1):
            for j in range(J):
                Lam[r * J + j, c * J + j] = T[r, c]
    ZLam = Zt @ Lam
    M = ZLam.T @ ZLam + torch.eye(q, dtype=Zt.dtype, device=Zt.device)
    L = torch.linalg.cholesky(M)
    cu = torch.linalg.solve_triangular(L, (ZLam.T @ yt).unsqueeze(1), upper=False)
    CX = torch.linalg.solve_triangular(L, ZLam.T @ Xt, upper=False)
    RtR = Xt.T @ Xt - CX.T @ CX
    rhs = (Xt.T @ yt).unsqueeze(1) - CX.T @ cu
    RX = torch.linalg.cholesky(RtR)
    beta = torch.cholesky_solve(rhs, RX).squeeze(1)
    ZLresid = (ZLam.T @ yt) - ZLam.T @ Xt @ beta
    u = torch.cholesky_solve(ZLresid.unsqueeze(1), L).squeeze(1)
    b = Lam @ u
    resid = yt - Xt @ beta - Zt @ b
    pwrss = (resid ** 2).sum() + (u ** 2).sum()
    n = Xt.shape[0]; p = Xt.shape[1]
    logdetL = 2.0 * torch.log(torch.diagonal(L).clamp_min(1e-30)).sum()
    logdetRX = 2.0 * torch.log(torch.diagonal(RX).abs().clamp_min(1e-30)).sum()
    if reml:
        df = n - p
        dev = logdetL + logdetRX + df * (1.0 + torch.log(2 * torch.pi * pwrss / df))
    else:
        dev = logdetL + n * (1.0 + torch.log(2 * torch.pi * pwrss / n))
    if return_diag:
        eig = torch.linalg.eigvalsh(M).min()
        return float(dev), float(eig), float(pwrss)
    return dev


# ------------------------------ bench ----------------------------
def bench(fn, *args, iters=20, cuda=False, **kw):
    if cuda:
        torch.cuda.synchronize()
    # warmup
    fn(*args, **kw)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kw)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    q_t = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_per = 20
    print(f"=== q_t = {q_t} random-effect terms, n_per={n_per} ===\n")

    # ---------- validation at a representative size ----------
    G0 = 300
    y, X, groups, re, rd, slopes = gen_single_factor(G0, n_per, q_t, seed=7)
    design_specs = parse_random_effects(groups, re, rd, len(y))
    Z = build_z_matrix(design_specs)
    n_theta = q_t * (q_t + 1) // 2
    rng = np.random.default_rng(0)
    theta_np = np.abs(rng.standard_normal(n_theta)) + 0.3  # positive-ish
    dev_lib = profiled_deviance_lmm(theta_np, X, Z, y, design_specs, reml=True)

    dev_cpu = "cpu"
    th64 = torch.tensor(theta_np, dtype=torch.float64)
    packed64 = make_batched_tensors(y, X, slopes, G0, n_per, q_t, "cpu", torch.float64)
    dev_bat = float(deviance_batched(th64, packed64, q_t, reml=True))
    Zt = torch.tensor(Z, dtype=torch.float64); Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    dev_den = float(deviance_dense_torch(th64, Xt, Zt, yt, q_t, reml=True))
    print(f"VALIDATION (G={G0}, random theta):")
    print(f"  library dense (numpy) = {dev_lib:.8f}")
    print(f"  torch dense           = {dev_den:.8f}   d={abs(dev_den-dev_lib):.2e}")
    print(f"  torch BATCHED         = {dev_bat:.8f}   d={abs(dev_bat-dev_lib):.2e}")
    print()

    has_cuda = torch.cuda.is_available()
    print(f"CUDA: {has_cuda} {torch.cuda.get_device_name(0) if has_cuda else ''}\n")

    # ---------- benchmark sweep ----------
    print(f"{'G':>6} {'n':>8} {'q':>7} | {'lib(np)':>9} {'denCPU64':>9} {'denGPU64':>9} {'denGPU32':>9} | "
          f"{'batCPU64':>9} {'batGPU64':>9} {'batGPU32':>9} | {'fp32 dev d':>11} {'mineig32':>10} {'mineig64':>10}")
    for G in [500, 1000, 2000, 4000, 8000]:
        y, X, groups, re, rd, slopes = gen_single_factor(G, n_per, q_t, seed=G)
        n = len(y); q = G * q_t
        th = torch.tensor(theta_np, dtype=torch.float64)
        # batched packs
        pk_cpu = make_batched_tensors(y, X, slopes, G, n_per, q_t, "cpu", torch.float64)
        t_bat_cpu = bench(deviance_batched, th, pk_cpu, q_t)
        # library dense (only when feasible: q<=2000 to avoid OOM/minutes)
        if q <= 2000:
            specs = parse_random_effects(groups, re, rd, n)
            Zl = build_z_matrix(specs)
            t0 = time.perf_counter()
            for _ in range(3):
                profiled_deviance_lmm(theta_np, X, Zl, y, specs, reml=True)
            t_lib = (time.perf_counter() - t0) / 3 * 1000
            # dense torch cpu
            Ztc = torch.tensor(Zl, dtype=torch.float64); Xtc = torch.tensor(X, dtype=torch.float64)
            ytc = torch.tensor(y, dtype=torch.float64)
            t_den_cpu = bench(deviance_dense_torch, th, Xtc, Ztc, ytc, q_t, iters=5)
        else:
            t_lib = float("nan"); t_den_cpu = float("nan")

        if has_cuda:
            pk_g64 = make_batched_tensors(y, X, slopes, G, n_per, q_t, "cuda", torch.float64)
            pk_g32 = make_batched_tensors(y, X, slopes, G, n_per, q_t, "cuda", torch.float32)
            th_g64 = torch.tensor(theta_np, dtype=torch.float64, device="cuda")
            th_g32 = torch.tensor(theta_np, dtype=torch.float32, device="cuda")
            t_bat_g64 = bench(deviance_batched, th_g64, pk_g64, q_t, cuda=True)
            t_bat_g32 = bench(deviance_batched, th_g32, pk_g32, q_t, cuda=True)
            # CF-1 fp32 vs fp64 on batched
            d64, e64, _ = deviance_batched(th_g64, pk_g64, q_t, return_diag=True)
            d32, e32, _ = deviance_batched(th_g32, pk_g32, q_t, return_diag=True)
            dev_d = abs(d32 - d64)
            # dense GPU (feasible q only)
            if q <= 4000:
                Ztg = torch.tensor(Zl if q <= 2000 else build_z_matrix(parse_random_effects(groups, re, rd, n)),
                                   dtype=torch.float64, device="cuda")
                Xtg = torch.tensor(X, dtype=torch.float64, device="cuda")
                ytg = torch.tensor(y, dtype=torch.float64, device="cuda")
                Ztg32 = Ztg.float(); Xtg32 = Xtg.float(); ytg32 = ytg.float()
                th_g32b = th_g32
                t_den_g64 = bench(deviance_dense_torch, th_g64, Xtg, Ztg, ytg, q_t, cuda=True, iters=5)
                t_den_g32 = bench(deviance_dense_torch, th_g32b, Xtg32, Ztg32, ytg32, q_t, cuda=True, iters=5)
                del Ztg, Xtg, ytg, Ztg32
                torch.cuda.empty_cache()
            else:
                t_den_g64 = float("nan"); t_den_g32 = float("nan")
        else:
            t_bat_g64 = t_bat_g32 = t_den_g64 = t_den_g32 = float("nan")
            dev_d = e32 = e64 = float("nan")

        print(f"{G:>6} {n:>8} {q:>7} | {t_lib:>9.2f} {t_den_cpu:>9.2f} {t_den_g64:>9.2f} {t_den_g32:>9.2f} | "
              f"{t_bat_cpu:>9.3f} {t_bat_g64:>9.3f} {t_bat_g32:>9.3f} | {dev_d:>11.2e} {e32:>10.2e} {e64:>10.2e}")
        if has_cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
