"""GAM GPU-feasibility CUDA study — Forge (RTX 5070 Ti), gpumice env.

Self-contained (numpy + torch only). Re-proves ON CUDA every claim the
gam-gpu-investigation memo previously made from local MPS / carried evidence:

  A. Crossover: cpu_fp64 (numpy) vs cuda_fp32_hybrid (GEMM fp32 on device,
     p x p solve on host fp64 -- what gpu_pirls.py does) vs cuda_fp64_hybrid
     vs cuda_fp64_device (all on-device fp64). Pins the crossover point and
     the REAL fp64 ceiling for the GAM kernel shape on the target card.
  B. CF-1 band (R13 re-proof on CUDA): fp32 X'WX formed by cuBLAS ->
     EDF corruption vs fp64, near-rank-deficient basis, lambda sweep.
     Plus a TF32-enabled variant (R14 hazard: a torch default flip would
     change the regime).
  C. Augmented-QR (Wood 2011) in fp32 ON CUDA (cuSOLVER geqrf): beta accuracy
     vs fp64 reference, and cost vs the gram path at large n.
  D. cuSOLVER vs LAPACK divergence on the near-singular fp64 solve
     (the gpu_pirls._solve_for_edf docstring claim, never measured).
  E. Batched multi-lambda evaluation (chip regime 3): B weight-vectors'
     X'W_bX as one batched op vs sequential CPU evals.

Writes results.json next to itself. Deterministic seeds. Runtime ~2-4 min.
"""
from __future__ import annotations

import io
import json
import os
import time
from contextlib import redirect_stdout

import numpy as np
import torch

DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

RESULTS: dict = {"env": {}, "A_crossover": [], "B_cf1_band": [],
                 "B_tf32": [], "C_aug_qr": [], "D_solver_divergence": [],
                 "E_batched_lambda": []}


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------- problems

def make_problem(n: int, p: int, seed: int = 0):
    """Well-conditioned spline-like design (crossover timing study)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    for j in range(1, p):
        X[:, j] = 0.6 * X[:, j] + 0.4 * X[:, j - 1]
    w = rng.uniform(0.5, 1.5, size=n)
    z = rng.standard_normal(n)
    S = np.zeros((p, p))
    for i in range(p):
        S[i, i] = 2.0
        if i + 1 < p:
            S[i, i + 1] = S[i + 1, i] = -1.0
    return X, w, z, S, 1e-3


def near_rank_deficient(n: int, p: int = 40, decay: float = 0.55,
                        seed: int = 1):
    """Geometric singular-value decay — the spectrum a high-k spline basis
    on clustered x actually has (many near-null directions)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    Vt, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return (U * (decay ** np.arange(p))) @ Vt


def second_diff_penalty(p: int):
    D = np.zeros((p - 2, p))
    for i in range(p - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D.T @ D            # rank p-2, null space dim 2 (const + linear)


# ---------------------------------------------------------------- study A

def cpu_inner(X, w, z, S, lam, reps):
    p = X.shape[1]
    t = {"elem": 0.0, "gemm": 0.0, "solve": 0.0, "edf": 0.0, "total": 0.0}
    for _ in range(reps):
        t0 = time.perf_counter()
        eta = X @ np.zeros(p)
        mu = eta
        _dev = float(np.sum((z - mu) ** 2 * w))
        t1 = time.perf_counter()
        XtW = X.T * w[np.newaxis, :]
        XtWX = XtW @ X
        b = XtW @ z
        t2 = time.perf_counter()
        A = XtWX + lam * S
        try:
            L = np.linalg.cholesky(A)
            _beta = np.linalg.solve(L.T, np.linalg.solve(L, b))
        except np.linalg.LinAlgError:
            _beta = np.linalg.solve(A + 1e-8 * np.eye(p), b)
        t3 = time.perf_counter()
        _F = np.linalg.solve(A, XtWX)
        _edf = float(np.trace(_F))
        t4 = time.perf_counter()
        t["elem"] += t1 - t0
        t["gemm"] += t2 - t1
        t["solve"] += t3 - t2
        t["edf"] += t4 - t3
        t["total"] += t4 - t0
    return {k: v / reps for k, v in t.items()}


def cuda_inner(X, w, z, S, lam, reps, mode):
    """mode: fp32_hybrid | fp64_hybrid | fp64_device."""
    dtype = torch.float32 if mode == "fp32_hybrid" else torch.float64
    Xt = torch.as_tensor(X, dtype=dtype, device=DEV)
    wt = torch.as_tensor(w, dtype=dtype, device=DEV)
    zt = torch.as_tensor(z, dtype=dtype, device=DEV)
    St = torch.as_tensor(S, dtype=dtype, device=DEV)
    p = X.shape[1]
    _ = (Xt.T * wt.unsqueeze(0)) @ Xt                      # warmup
    sync()
    t = {"elem": 0.0, "gemm": 0.0, "solve_edf": 0.0, "total": 0.0}
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        eta = Xt @ torch.zeros(p, device=DEV, dtype=dtype)
        mu = eta
        _dev = float(((zt - mu) ** 2 * wt).sum().item())
        sync()
        t1 = time.perf_counter()
        XtW = Xt.T * wt.unsqueeze(0)
        XtWX = XtW @ Xt
        b = XtW @ zt
        sync()
        t2 = time.perf_counter()
        A = XtWX + lam * St
        if mode in ("fp32_hybrid", "fp64_hybrid"):
            A_np = A.detach().cpu().double().numpy()
            b_np = b.detach().cpu().double().numpy()
            XtWX_np = XtWX.detach().cpu().double().numpy()
            try:
                L = np.linalg.cholesky(A_np)
                _beta = np.linalg.solve(L.T, np.linalg.solve(L, b_np))
            except np.linalg.LinAlgError:
                _beta = np.linalg.solve(A_np + 1e-8 * np.eye(p), b_np)
            _F = np.linalg.solve(A_np, XtWX_np)
            _edf = float(np.trace(_F))
        else:                                              # fp64_device
            _beta = torch.linalg.solve(A, b)
            _F = torch.linalg.solve(A, XtWX)
            _edf = float(torch.diagonal(_F).sum().item())
        sync()
        t3 = time.perf_counter()
        t["elem"] += t1 - t0
        t["gemm"] += t2 - t1
        t["solve_edf"] += t3 - t2
        t["total"] += t3 - t0
    return {k: v / reps for k, v in t.items()}


def study_a():
    print("\n== A. crossover: cpu_fp64 vs cuda paths ==")
    for p in (50, 200):
        print(f"-- p={p}")
        hdr = (f"{'n':>8} | {'cpu64':>10} | {'cu32hyb':>10} {'x':>6} | "
               f"{'cu64hyb':>10} {'x':>6} | {'cu64dev':>10} {'x':>6}")
        print(hdr)
        for n in (500, 2000, 5000, 20000, 100000, 500000, 1000000):
            reps = 5 if n >= 500000 else (10 if n >= 20000 else 30)
            X, w, z, S, lam = make_problem(n, p)
            c = cpu_inner(X, w, z, S, lam, reps)
            r32h = cuda_inner(X, w, z, S, lam, reps, "fp32_hybrid")
            r64h = cuda_inner(X, w, z, S, lam, reps, "fp64_hybrid")
            r64d = cuda_inner(X, w, z, S, lam, reps, "fp64_device")
            row = {"n": n, "p": p, "cpu": c, "cuda_fp32_hybrid": r32h,
                   "cuda_fp64_hybrid": r64h, "cuda_fp64_device": r64d}
            RESULTS["A_crossover"].append(row)
            def us(x): return f"{x*1e6:9.0f}u"
            print(f"{n:>8} | {us(c['total'])} | {us(r32h['total'])} "
                  f"{c['total']/r32h['total']:5.2f}x | {us(r64h['total'])} "
                  f"{c['total']/r64h['total']:5.2f}x | {us(r64d['total'])} "
                  f"{c['total']/r64d['total']:5.2f}x")


# ---------------------------------------------------------------- study B

def edf_from_gram(XtWX, S, lam, XtWX_ref=None):
    A = XtWX + lam * S
    ref = XtWX if XtWX_ref is None else XtWX_ref
    return float(np.trace(np.linalg.solve(A, XtWX)))


def study_b():
    print("\n== B. CF-1 band on CUDA: fp32 cuBLAS X'WX -> EDF ==")
    p = 40
    S = second_diff_penalty(p)
    print(f"{'n':>8} {'lam':>8} {'cond(A)':>11} {'EDF64':>9} "
          f"{'EDF32cuda':>10} {'err':>8}")
    for n in (5000, 100000):
        X = near_rank_deficient(n, p)
        XtWX64 = X.T @ X
        Xt32 = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        XtWX32 = (Xt32.T @ Xt32).cpu().double().numpy()
        for lam in (1e-2, 1e-4, 1e-6, 1e-8):
            A64 = XtWX64 + lam * S
            cond = float(np.linalg.cond(A64))
            e64 = edf_from_gram(XtWX64, S, lam)
            e32 = edf_from_gram(XtWX32, S, lam)
            row = {"n": n, "lam": lam, "cond": cond, "edf_fp64": e64,
                   "edf_fp32_cuda": e32, "abs_err": abs(e32 - e64)}
            RESULTS["B_cf1_band"].append(row)
            print(f"{n:>8} {lam:>8.0e} {cond:>11.2e} {e64:>9.4f} "
                  f"{e32:>10.4f} {abs(e32-e64):>8.3f}")

    # TF32 variant (R14 hazard): what a flipped torch default would do.
    print("-- TF32-enabled gram (allow_tf32=True), n=100000:")
    torch.backends.cuda.matmul.allow_tf32 = True
    X = near_rank_deficient(100000, p)
    XtWX64 = X.T @ X
    Xt32 = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    XtWX_tf32 = (Xt32.T @ Xt32).cpu().double().numpy()
    torch.backends.cuda.matmul.allow_tf32 = False
    rel = float(np.linalg.norm(XtWX_tf32 - XtWX64) / np.linalg.norm(XtWX64))
    for lam in (1e-2, 1e-4, 1e-6, 1e-8):
        e64 = edf_from_gram(XtWX64, S, lam)
        etf = edf_from_gram(XtWX_tf32, S, lam)
        row = {"n": 100000, "lam": lam, "gram_relerr": rel,
               "edf_fp64": e64, "edf_tf32": etf, "abs_err": abs(etf - e64)}
        RESULTS["B_tf32"].append(row)
        print(f"  lam={lam:.0e}  gram_relerr={rel:.1e}  EDF64={e64:.4f}  "
              f"EDF_tf32={etf:.4f}  err={abs(etf-e64):.3f}")


# ---------------------------------------------------------------- study C

def study_c():
    print("\n== C. augmented QR fp32 on CUDA (cuSOLVER) ==")
    p = 40
    S = second_diff_penalty(p)
    rng = np.random.default_rng(2)
    print(f"{'n':>8} {'lam':>8} {'cond_NE':>10} {'cond_aug':>10} "
          f"{'bNE32':>10} {'bQR32cuda':>10} | {'t_qr32':>9} {'t_gemm32':>9}")
    for n in (5000, 100000):
        X = near_rank_deficient(n, p)
        z = rng.standard_normal(n)
        for lam in (1e-4, 1e-6, 1e-8):
            A64 = X.T @ X + lam * S
            beta_ref = np.linalg.solve(A64, X.T @ z)
            cond_ne = float(np.linalg.cond(A64))
            evals, evecs = np.linalg.eigh(lam * S)
            B = (evecs * np.sqrt(np.clip(evals, 0, None))) @ evecs.T
            Aug = np.vstack([X, B])
            rhs = np.concatenate([z, np.zeros(p)])
            cond_aug = float(np.linalg.cond(Aug))
            # fp32 QR on CUDA
            Aug32 = torch.as_tensor(Aug, dtype=torch.float32, device=DEV)
            rhs32 = torch.as_tensor(rhs, dtype=torch.float32, device=DEV)
            sync(); t0 = time.perf_counter()
            Q, R = torch.linalg.qr(Aug32)
            qtb = Q.T @ rhs32
            sync(); t_qr = time.perf_counter() - t0
            beta_qr = np.linalg.solve(R.cpu().double().numpy(),
                                      qtb.cpu().double().numpy())
            # fp32 normal-equations on CUDA for contrast (+ timing baseline)
            X32 = torch.as_tensor(X, dtype=torch.float32, device=DEV)
            sync(); t0 = time.perf_counter()
            XtWX32 = X32.T @ X32
            sync(); t_ge = time.perf_counter() - t0
            A32 = XtWX32.cpu().double().numpy() + lam * S
            beta_ne = np.linalg.solve(A32, X.T @ z)
            e_qr = float(np.linalg.norm(beta_qr - beta_ref)
                         / np.linalg.norm(beta_ref))
            e_ne = float(np.linalg.norm(beta_ne - beta_ref)
                         / np.linalg.norm(beta_ref))
            row = {"n": n, "lam": lam, "cond_ne": cond_ne,
                   "cond_aug": cond_aug, "beta_relerr_ne32": e_ne,
                   "beta_relerr_qr32_cuda": e_qr,
                   "t_qr32_s": t_qr, "t_gemm32_s": t_ge}
            RESULTS["C_aug_qr"].append(row)
            print(f"{n:>8} {lam:>8.0e} {cond_ne:>10.2e} {cond_aug:>10.2e} "
                  f"{e_ne:>10.2e} {e_qr:>10.2e} | {t_qr*1e3:>8.2f}m "
                  f"{t_ge*1e3:>8.2f}m")


# ---------------------------------------------------------------- study D

def study_d():
    print("\n== D. cuSOLVER vs LAPACK fp64 solve on near-singular A ==")
    p = 40
    S = second_diff_penalty(p)
    print(f"{'n':>8} {'lam':>9} {'cond(A)':>11} {'EDF_lapack':>11} "
          f"{'EDF_cusolver':>13} {'abs_err':>9}")
    for n in (5000, 100000):
        X = near_rank_deficient(n, p)
        XtWX = X.T @ X                                     # fp64, host
        XtWX_t = torch.as_tensor(XtWX, dtype=torch.float64, device=DEV)
        S_t = torch.as_tensor(S, dtype=torch.float64, device=DEV)
        for lam in (1e-4, 1e-6, 1e-8, 1e-10):
            A = XtWX + lam * S
            cond = float(np.linalg.cond(A))
            edf_l = float(np.trace(np.linalg.solve(A, XtWX)))
            A_t = XtWX_t + lam * S_t
            F_t = torch.linalg.solve(A_t, XtWX_t)
            edf_c = float(torch.diagonal(F_t).sum().item())
            row = {"n": n, "lam": lam, "cond": cond, "edf_lapack": edf_l,
                   "edf_cusolver": edf_c, "abs_err": abs(edf_c - edf_l)}
            RESULTS["D_solver_divergence"].append(row)
            print(f"{n:>8} {lam:>9.0e} {cond:>11.2e} {edf_l:>11.4f} "
                  f"{edf_c:>13.4f} {abs(edf_c-edf_l):>9.4f}")


# ---------------------------------------------------------------- study E

def study_e():
    print("\n== E. batched multi-lambda (regime 3): B=50 weight vectors ==")
    n, p, B = 100000, 50, 50
    X, _, z, S, _ = make_problem(n, p)
    rng = np.random.default_rng(3)
    Wb = rng.uniform(0.5, 1.5, size=(B, n))
    lams = np.logspace(-6, 4, B)
    # CPU sequential
    t0 = time.perf_counter()
    edfs_cpu = []
    for b in range(B):
        XtW = X.T * Wb[b][np.newaxis, :]
        XtWX = XtW @ X
        A = XtWX + lams[b] * S
        edfs_cpu.append(float(np.trace(np.linalg.solve(A, XtWX))))
    t_cpu = time.perf_counter() - t0
    # CUDA batched fp64: (B,n,p) weighted copies -> bmm -> batched solve
    Xt = torch.as_tensor(X, dtype=torch.float64, device=DEV)
    Wt = torch.as_tensor(Wb, dtype=torch.float64, device=DEV)
    St = torch.as_tensor(S, dtype=torch.float64, device=DEV)
    lt = torch.as_tensor(lams, dtype=torch.float64, device=DEV)
    sync(); t0 = time.perf_counter()
    Xw = Xt.unsqueeze(0) * Wt.unsqueeze(-1)                # (B,n,p)
    XtWX_b = torch.matmul(Xt.T.unsqueeze(0), Xw)           # (B,p,p)
    A_b = XtWX_b + lt.view(-1, 1, 1) * St.unsqueeze(0)
    F_b = torch.linalg.solve(A_b, XtWX_b)
    edfs_gpu = torch.diagonal(F_b, dim1=-2, dim2=-1).sum(-1)
    sync(); t_gpu = time.perf_counter() - t0
    edfs_gpu = edfs_gpu.cpu().numpy()
    max_err = float(np.max(np.abs(edfs_gpu - np.array(edfs_cpu))))
    row = {"n": n, "p": p, "B": B, "t_cpu_sequential_s": t_cpu,
           "t_cuda_batched_fp64_s": t_gpu, "speedup": t_cpu / t_gpu,
           "max_edf_abs_err": max_err,
           "mem_note": "(B,n,p) fp64 weighted copy = "
                       f"{B*n*p*8/1e9:.1f} GB materialized"}
    RESULTS["E_batched_lambda"].append(row)
    print(f"  CPU sequential: {t_cpu:.3f}s   CUDA batched fp64: {t_gpu:.3f}s "
          f"  speedup {t_cpu/t_gpu:.2f}x   max EDF err {max_err:.2e}")


# ---------------------------------------------------------------- main

def main():
    cpu_model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    RESULTS["env"] = {
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "numpy": np.__version__,
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "tf32_default": False,
        "numpy_config_head": buf.getvalue()[:600],
    }
    print("env:", {k: v for k, v in RESULTS["env"].items()
                   if k != "numpy_config_head"})
    study_a()
    study_b()
    study_c()
    study_d()
    study_e()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
