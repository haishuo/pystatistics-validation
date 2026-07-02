"""GAM GPU-feasibility microbenchmark (local: CPU fp64 vs MPS fp32).

Measures the core per-evaluation cost of the GAM P-IRLS inner operation as a
function of n and p, decomposed into:
  (a) elementwise PIRLS n-vector updates (link/mu/w/z/deviance analogue)
  (b) the X'WX GEMM   (n x p -> p x p)   -- the ONLY n-scaling linear algebra
  (c) the p x p penalized solve + EDF hat-trace F = A^{-1} X'WX  (ill-conditioned)

Three execution models, mirroring the real backends:
  cpu_fp64        : everything numpy fp64 (the R-validated reference path)
  mps_hybrid      : GEMM on MPS fp32, p x p solve round-tripped to host fp64
                    (this is exactly what gpu_pirls.py does)
  mps_all_fp32    : everything on MPS fp32 incl. the solve (numerically UNSAFE
                    on near-singular A; measured only to show the launch profile)

MPS understates CUDA; the fp64-CUDA throttle ceiling is carried from the
measured GRM study (gpu_fp64 = 1.2-1.6x on the same Forge RTX card). This bench
establishes the *shape*: where (if anywhere) the GEMM overtakes the CPU, and how
much of the fit is the tiny ill-conditioned solve vs the n-scaling GEMM.
"""
from __future__ import annotations

import time
import numpy as np

try:
    import torch
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    torch = None
    HAS_MPS = False


def make_problem(n: int, p: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Design matrix with realistic-ish collinearity (spline bases overlap).
    X = rng.standard_normal((n, p))
    # induce mild collinearity between neighbouring columns (spline-like)
    for j in range(1, p):
        X[:, j] = 0.6 * X[:, j] + 0.4 * X[:, j - 1]
    w = rng.uniform(0.5, 1.5, size=n)          # IRLS weights
    z = rng.standard_normal(n)                  # working response
    # A banded 2nd-difference penalty (this is what a cr penalty looks like:
    # banded, bandwidth = spline degree). Padded into the pxp block.
    S = np.zeros((p, p))
    for i in range(p):
        S[i, i] = 2.0
        if i + 1 < p:
            S[i, i + 1] = S[i + 1, i] = -1.0
    lam = 1e-3                                   # small lambda -> near-singular A
    return X, w, z, S, lam


def cpu_inner(X, w, z, S, lam, reps: int) -> dict:
    times = {"gemm": 0.0, "solve": 0.0, "edf": 0.0, "elem": 0.0, "total": 0.0}
    p = X.shape[1]
    for _ in range(reps):
        t0 = time.perf_counter()
        # (a) elementwise PIRLS-style n-vector work
        eta = X @ np.zeros(p)          # placeholder linear predictor
        mu = eta
        _dev = float(np.sum((z - mu) ** 2 * w))
        t1 = time.perf_counter()
        # (b) X'WX GEMM
        XtW = X.T * w[np.newaxis, :]
        XtWX = XtW @ X
        b = XtW @ z
        t2 = time.perf_counter()
        # (c) penalized solve
        A = XtWX + lam * S
        try:
            L = np.linalg.cholesky(A)
            _beta = np.linalg.solve(L.T, np.linalg.solve(L, b))
        except np.linalg.LinAlgError:
            _beta = np.linalg.solve(A + 1e-8 * np.eye(p), b)
        t3 = time.perf_counter()
        # EDF hat-trace F = A^{-1} XtWX
        _F = np.linalg.solve(A, XtWX)
        _edf = float(np.trace(_F))
        t4 = time.perf_counter()
        times["elem"] += t1 - t0
        times["gemm"] += t2 - t1
        times["solve"] += t3 - t2
        times["edf"] += t4 - t3
        times["total"] += t4 - t0
    for k in times:
        times[k] /= reps
    return times


def mps_inner(X, w, z, S, lam, reps: int, hybrid: bool) -> dict:
    dev = torch.device("mps")
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
    zt = torch.as_tensor(z, dtype=torch.float32, device=dev)
    St = torch.as_tensor(S, dtype=torch.float32, device=dev)
    p = X.shape[1]
    # warmup
    _ = (Xt.T * wt.unsqueeze(0)) @ Xt
    torch.mps.synchronize()
    times = {"gemm": 0.0, "solve": 0.0, "edf": 0.0, "elem": 0.0, "total": 0.0,
             "roundtrip": 0.0}
    for _ in range(reps):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        eta = Xt @ torch.zeros(p, device=dev, dtype=torch.float32)
        mu = eta
        _dev = float(((zt - mu) ** 2 * wt).sum().item())
        torch.mps.synchronize()
        t1 = time.perf_counter()
        XtW = Xt.T * wt.unsqueeze(0)
        XtWX = XtW @ Xt
        b = XtW @ zt
        torch.mps.synchronize()
        t2 = time.perf_counter()
        A = XtWX + lam * St
        if hybrid:
            # round-trip the pxp solve to host fp64 (what gpu_pirls does)
            A_np = A.detach().cpu().double().numpy()
            b_np = b.detach().cpu().double().numpy()
            XtWX_np = XtWX.detach().cpu().double().numpy()
            tr0 = time.perf_counter()
            try:
                L = np.linalg.cholesky(A_np)
                _beta = np.linalg.solve(L.T, np.linalg.solve(L, b_np))
            except np.linalg.LinAlgError:
                _beta = np.linalg.solve(A_np + 1e-8 * np.eye(p), b_np)
            _F = np.linalg.solve(A_np, XtWX_np)
            _edf = float(np.trace(_F))
            tr1 = time.perf_counter()
            times["roundtrip"] += tr1 - tr0
            t3 = time.perf_counter()
            t4 = t3
        else:
            _beta = torch.linalg.solve(A, b)
            torch.mps.synchronize()
            t3 = time.perf_counter()
            _F = torch.linalg.solve(A, XtWX)
            _edf = float(torch.diagonal(_F).sum().item())
            torch.mps.synchronize()
            t4 = time.perf_counter()
        times["elem"] += t1 - t0
        times["gemm"] += t2 - t1
        times["solve"] += t3 - t2
        times["edf"] += t4 - t3
        times["total"] += t4 - t0
    for k in times:
        times[k] /= reps
    return times


def fmt_us(x):
    return f"{x*1e6:8.1f}us"


def main():
    ns = [500, 2000, 5000, 20000, 100000, 500000]
    ps = [50, 200]
    print(f"MPS available: {HAS_MPS}")
    print("=" * 100)
    for p in ps:
        print(f"\n### p = {p}  (typical GAM p~30-80; p=200 ~ several smooths / tensor product)")
        header = (f"{'n':>8} | {'cpu_total':>11} {'cpu_gemm':>10} {'cpu_solve':>10} "
                  f"{'cpu_edf':>10} | {'mpsH_total':>11} {'mpsH_gemm':>10} {'mpsH_rt':>10} "
                  f"| {'speedup':>8}")
        print(header)
        print("-" * len(header))
        for n in ns:
            reps = 5 if n >= 100000 else (10 if n >= 20000 else 30)
            X, w, z, S, lam = make_problem(n, p)
            c = cpu_inner(X, w, z, S, lam, reps)
            if HAS_MPS:
                m = mps_inner(X, w, z, S, lam, reps, hybrid=True)
                speed = c["total"] / m["total"]
                print(f"{n:>8} | {fmt_us(c['total'])} {fmt_us(c['gemm'])} "
                      f"{fmt_us(c['solve'])} {fmt_us(c['edf'])} | "
                      f"{fmt_us(m['total'])} {fmt_us(m['gemm'])} "
                      f"{fmt_us(m['roundtrip'])} | {speed:7.2f}x")
            else:
                print(f"{n:>8} | {fmt_us(c['total'])} {fmt_us(c['gemm'])} "
                      f"{fmt_us(c['solve'])} {fmt_us(c['edf'])} | (no mps)")


if __name__ == "__main__":
    main()
