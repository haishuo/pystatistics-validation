"""
Phase C: the ONE GPU-favorable mixed-model formulation - low-rank / GRM.

Model: y = X beta + g + eps,  g ~ N(0, sigma_g^2 K),  K = W W' / M  (low rank M),
i.e. a single variance component with a low-rank covariance (genomic GRM / a
reduced-rank random effect). Reparam g = (W/sqrt(M)) u, u ~ N(0, sigma_g^2 I_M),
so Z = W/sqrt(M) is an (n x M) DENSE design and the Gram is a genuinely large
DENSE M x M matrix -> big dense Cholesky + n x M GEMMs = the cuBLAS/cuSOLVER
regime, UNLIKE the tiny per-group blocks of an lme4 design.

This is NOT "lme4 LMM on GPU". It is a different model (low-rank / GRM mixed
model) that classic REML on a sparse design cannot reach at scale - the
discrete-time-survival analogue.

We:
  (1) validate the Woodbury/M-space profiled deviance against the direct n x n
      "V-space" deviance (independent computation) -> correctness,
  (2) benchmark per-eval: CPU fp64 vs CUDA fp64 vs CUDA fp32,
  (3) CF-1 check: this path DOES form W'W (squares condition number). Measure
      min-eigenvalue of the M-space Gram and deviance, fp32 vs fp64 -> does the
      fp32 path silently lose accuracy? (the gate this formulation would need.)
"""
from __future__ import annotations
import time, sys
import numpy as np
import torch


def gen(n, M, p=2, seed=0, signal=0.5):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, M))
    W -= W.mean(0); W /= (W.std(0) + 1e-8)        # standardized "genotypes"
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(p - 1)])
    u = rng.standard_normal(M) * signal
    g = (W @ u) / np.sqrt(M)
    y = X @ np.arange(1, p + 1, dtype=float) + g + rng.standard_normal(n)
    return y, X, W


def dev_mspace(theta, Wt, Xt, yt, reml=True, return_diag=False):
    """Profiled REML deviance in M-space (Woodbury). theta = sigma_g/sigma_e."""
    n, M = Wt.shape
    p = Xt.shape[1]
    c = theta / (M ** 0.5)
    ZL = c * Wt                                    # (n, M)
    Gm = ZL.transpose(-1, -2) @ ZL + torch.eye(M, dtype=Wt.dtype, device=Wt.device)
    L = torch.linalg.cholesky(Gm)                  # (M, M) big dense chol
    ZLy = (ZL.transpose(-1, -2) @ yt.unsqueeze(1)) # (M,1)
    ZLX = ZL.transpose(-1, -2) @ Xt               # (M,p)
    cu = torch.linalg.solve_triangular(L, ZLy, upper=False)
    CX = torch.linalg.solve_triangular(L, ZLX, upper=False)
    RtR = Xt.T @ Xt - CX.transpose(-1, -2) @ CX
    rhs = (Xt.T @ yt).unsqueeze(1) - CX.transpose(-1, -2) @ cu
    RX = torch.linalg.cholesky(RtR)
    beta = torch.cholesky_solve(rhs, RX).squeeze(1)
    rhs_u = ZLy.squeeze(1) - (ZLX @ beta)
    u = torch.cholesky_solve(rhs_u.unsqueeze(1), L).squeeze(1)
    resid = yt - Xt @ beta - ZL @ u
    pwrss = (resid ** 2).sum() + (u ** 2).sum()
    logdetV = 2.0 * torch.log(torch.diagonal(L).clamp_min(1e-30)).sum()   # = log|V|
    logdetRX = 2.0 * torch.log(torch.diagonal(RX).abs().clamp_min(1e-30)).sum()
    if reml:
        df = n - p
        dev = logdetV + logdetRX + df * (1 + torch.log(2 * torch.pi * pwrss / df))
    else:
        dev = logdetV + n * (1 + torch.log(2 * torch.pi * pwrss / n))
    if return_diag:
        eig = torch.linalg.eigvalsh(Gm).min()
        return float(dev), float(eig)
    return dev


def dev_vspace(theta, W, X, y, reml=True):
    """Independent check: direct n x n V-space profiled REML deviance."""
    n, M = W.shape; p = X.shape[1]
    c2 = (theta ** 2) / M
    V = c2 * (W @ W.T) + np.eye(n)
    Lv = np.linalg.cholesky(V)
    Vi = np.linalg.inv(V)
    XtViX = X.T @ Vi @ X
    beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
    r = y - X @ beta
    rss = float(r @ Vi @ r)
    logdetV = 2.0 * np.sum(np.log(np.diag(Lv)))
    sign, logdetXVX = np.linalg.slogdet(XtViX)
    df = n - p
    return logdetV + logdetXVX + df * (1 + np.log(2 * np.pi * rss / df))


def bench(fn, *a, cuda=False, iters=20, **k):
    if cuda: torch.cuda.synchronize()
    fn(*a, **k)
    if cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn(*a, **k)
    if cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    # ---- validation ----
    y, X, W = gen(800, 150, seed=3)
    th = 1.3
    d_v = dev_vspace(th, W, X, y)
    d_m = float(dev_mspace(torch.tensor(th), torch.tensor(W), torch.tensor(X), torch.tensor(y)))
    print(f"VALIDATION n=800 M=150: V-space={d_v:.6f}  M-space={d_m:.6f}  d={abs(d_v-d_m):.2e}\n")

    has = torch.cuda.is_available()
    print(f"CUDA: {has} {torch.cuda.get_device_name(0) if has else ''}\n")
    n = 20000
    print(f"n={n} fixed; vary rank M (the dense Gram dimension)\n")
    print(f"{'M':>6} | {'CPU fp64':>9} {'GPU fp64':>9} {'GPU fp32':>9} | "
          f"{'fp64/gpu64':>10} {'fp64/gpu32':>10} | {'dev fp32 d':>11} {'mineig64':>10} {'mineig32':>10}")
    for M in [200, 500, 1000, 2000, 4000, 6000]:
        y, X, W = gen(n, M, seed=M)
        th = 1.2
        Wc = torch.tensor(W, dtype=torch.float64); Xc = torch.tensor(X, dtype=torch.float64)
        yc = torch.tensor(y, dtype=torch.float64); tc = torch.tensor(th, dtype=torch.float64)
        t_cpu = bench(dev_mspace, tc, Wc, Xc, yc, iters=5)
        if has:
            Wg = Wc.cuda(); Xg = Xc.cuda(); yg = yc.cuda(); tg = tc.cuda()
            Wg32 = Wg.float(); Xg32 = Xg.float(); yg32 = yg.float(); tg32 = tg.float()
            t_g64 = bench(dev_mspace, tg, Wg, Xg, yg, cuda=True, iters=10)
            t_g32 = bench(dev_mspace, tg32, Wg32, Xg32, yg32, cuda=True, iters=10)
            d64, e64 = dev_mspace(tg, Wg, Xg, yg, return_diag=True)
            d32, e32 = dev_mspace(tg32, Wg32, Xg32, yg32, return_diag=True)
            dd = abs(d32 - d64)
            sp64 = t_cpu / t_g64; sp32 = t_cpu / t_g32
            del Wg, Xg, yg, Wg32, Xg32, yg32; torch.cuda.empty_cache()
        else:
            t_g64 = t_g32 = sp64 = sp32 = dd = e64 = e32 = float("nan")
        print(f"{M:>6} | {t_cpu:>9.2f} {t_g64:>9.2f} {t_g32:>9.2f} | "
              f"{sp64:>9.2f}x {sp32:>9.2f}x | {dd:>11.2e} {e64:>10.2e} {e32:>10.2e}")


if __name__ == "__main__":
    main()
