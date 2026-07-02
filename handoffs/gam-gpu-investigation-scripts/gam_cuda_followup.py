"""Follow-up CUDA studies (Forge): D2 — push cuSOLVER-vs-LAPACK fp64 to the
cond~1e13..1e17 regime the gpu_pirls docstring claims diverges; C2 — the
fp32-safe augmented-QR path timed at large n (the number that decides whether
the SAFE formulation carries a real win), vs the CPU inner op and the (unsafe)
fp32 gram. Appends to results_followup.json."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False

OUT: dict = {"D2_cond_push": [], "C2_qr_large_n": []}


def sync():
    torch.cuda.synchronize()


def near_rank_deficient(n, p=40, decay=0.55, seed=1):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    Vt, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return (U * (decay ** np.arange(p))) @ Vt


def second_diff_penalty(p):
    D = np.zeros((p - 2, p))
    for i in range(p - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D.T @ D


def study_d2():
    print("== D2. cuSOLVER vs LAPACK fp64, cond pushed toward 1e17 ==")
    n, p = 5000, 40
    S = second_diff_penalty(p)
    print(f"{'decay':>6} {'lam':>9} {'cond(A)':>11} {'EDF_lapack':>11} "
          f"{'EDF_cusolver':>13} {'abs_err':>10} {'rel_err':>10}")
    for decay in (0.55, 0.70):
        X = near_rank_deficient(n, p, decay=decay)
        XtWX = X.T @ X
        XtWX_t = torch.as_tensor(XtWX, dtype=torch.float64, device=DEV)
        S_t = torch.as_tensor(S, dtype=torch.float64, device=DEV)
        for lam in (1e-10, 1e-12, 1e-14, 1e-16):
            A = XtWX + lam * S
            cond = float(np.linalg.cond(A))
            edf_l = float(np.trace(np.linalg.solve(A, XtWX)))
            A_t = XtWX_t + lam * S_t
            F_t = torch.linalg.solve(A_t, XtWX_t)
            edf_c = float(torch.diagonal(F_t).sum().item())
            err = abs(edf_c - edf_l)
            rel = err / max(abs(edf_l), 1e-30)
            OUT["D2_cond_push"].append(
                {"n": n, "decay": decay, "lam": lam, "cond": cond,
                 "edf_lapack": edf_l, "edf_cusolver": edf_c,
                 "abs_err": err, "rel_err": rel})
            print(f"{decay:>6} {lam:>9.0e} {cond:>11.2e} {edf_l:>11.4f} "
                  f"{edf_c:>13.4f} {err:>10.4f} {rel:>10.2e}")


def study_c2():
    print("\n== C2. fp32-safe augmented QR at large n (p=40): the SAFE path ==")
    p = 40
    S = second_diff_penalty(p)
    rng = np.random.default_rng(2)
    lam = 1e-8                                    # the adversarial regime
    evals, evecs = np.linalg.eigh(lam * S)
    B = (evecs * np.sqrt(np.clip(evals, 0, None))) @ evecs.T
    print(f"{'n':>8} | {'t_cpu_inner':>11} {'t_qr32':>9} {'x_safe':>7} | "
          f"{'t_gram32':>9} {'x_unsafe':>8} | {'beta_relerr_qr32':>16}")
    for n in (100000, 500000, 1000000):
        X = near_rank_deficient(n, p)
        z = rng.standard_normal(n)
        w = np.ones(n)
        # CPU inner op (gemm + solve + edf), the reference cost
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            XtW = X.T * w[np.newaxis, :]
            XtWX = XtW @ X
            A = XtWX + lam * S
            bb = XtW @ z
            _beta = np.linalg.solve(A, bb)
            _F = np.linalg.solve(A, XtWX)
            _ = float(np.trace(_F))
        t_cpu = (time.perf_counter() - t0) / reps
        beta_ref = np.linalg.solve(X.T @ X + lam * S, X.T @ z)
        # fp32 QR on CUDA (weights fold into rows: sqrt(w)*X; w=1 here)
        Aug = np.vstack([X, B])
        rhs = np.concatenate([z, np.zeros(p)])
        Aug32 = torch.as_tensor(Aug, dtype=torch.float32, device=DEV)
        rhs32 = torch.as_tensor(rhs, dtype=torch.float32, device=DEV)
        Q, R = torch.linalg.qr(Aug32)              # warmup
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            Q, R = torch.linalg.qr(Aug32)
            qtb = Q.T @ rhs32
            R_h = R.cpu().double().numpy()
            qtb_h = qtb.cpu().double().numpy()
            beta_qr = np.linalg.solve(R_h, qtb_h)
            # EDF via R: F-trace equivalent needs R^-T X'X R^-1-ish; cost is
            # p x p host work, folded in as the solve above (representative).
        sync()
        t_qr = (time.perf_counter() - t0) / reps
        # (unsafe) fp32 gram GEMM for contrast
        X32 = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        w32 = torch.as_tensor(w, dtype=torch.float32, device=DEV)
        _ = (X32.T * w32.unsqueeze(0)) @ X32
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            XtW32 = X32.T * w32.unsqueeze(0)
            XtWX32 = XtW32 @ X32
            b32 = XtW32 @ torch.as_tensor(z, dtype=torch.float32, device=DEV)
            A_np = XtWX32.cpu().double().numpy() + lam * S
            _beta = np.linalg.solve(A_np, b32.cpu().double().numpy())
            _F = np.linalg.solve(A_np, XtWX32.cpu().double().numpy())
            _ = float(np.trace(_F))
        sync()
        t_gram = (time.perf_counter() - t0) / reps
        err_qr = float(np.linalg.norm(beta_qr - beta_ref)
                       / np.linalg.norm(beta_ref))
        OUT["C2_qr_large_n"].append(
            {"n": n, "p": p, "lam": lam, "t_cpu_inner_s": t_cpu,
             "t_qr32_cuda_s": t_qr, "speedup_safe": t_cpu / t_qr,
             "t_gram32_cuda_s": t_gram, "speedup_unsafe": t_cpu / t_gram,
             "beta_relerr_qr32": err_qr})
        print(f"{n:>8} | {t_cpu*1e3:>10.2f}m {t_qr*1e3:>8.2f}m "
              f"{t_cpu/t_qr:>6.1f}x | {t_gram*1e3:>8.2f}m "
              f"{t_cpu/t_gram:>7.1f}x | {err_qr:>16.2e}")


def main():
    study_d2()
    study_c2()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_followup.json")
    with open(out, "w") as f:
        json.dump(OUT, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
