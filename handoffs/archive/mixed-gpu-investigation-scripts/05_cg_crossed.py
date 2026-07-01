"""
Phase B.3: CROSSED via CG / Woodbury (rocks #2, #7) - sparse solve on GPU.

The crossed M = I + Lambda'Z'ZLambda (q x q, sparse) blows up under direct
sparse Cholesky because of fill-in. The genomics GPU tools (SAIGE-GPU, BOLT)
avoid this with CONJUGATE GRADIENT: the only kernel is the matvec
    M @ v = v + ZLam'(ZLam @ v)
computed straight from the sparse ZLam (O(nnz)), never forming/factoring M.
This matvec is the GPU-favorable primitive.

We measure:
  (1) direct splu factor+solve time (the fill-in cost),
  (2) CG solve time on CPU (does CG sidestep the fill-in blowup?),
  (3) the M@v matvec throughput, CPU vs CUDA (is the kernel GPU-favorable?),
  (4) the BLOCKER: CG yields the solve but NOT log|det M|, which the profiled
      deviance needs. We show forming/factoring M for the determinant is the
      cost CG was avoiding -> why genomics uses stochastic-trace AI-REML
      (a different, non-deterministic algorithm), not deviance minimization.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg, LinearOperator
import torch


def gen_crossed(n_a, n_b, n_obs, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_a, n_obs); b = rng.integers(0, n_b, n_obs)
    return a, b, n_a, n_b


def build_ZLam(a, b, n_a, n_b, theta):
    n = len(a)
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([a, n_a + b])
    vals = np.concatenate([np.full(n, theta[0]), np.full(n, theta[1])])
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n_a + n_b))


def main():
    print("=" * 88)
    print("CROSSED via CG/Woodbury: direct-factor vs CG-solve vs GPU matvec")
    print("=" * 88)
    theta = np.array([0.6, 0.9])
    has_cuda = torch.cuda.is_available()
    print(f"CUDA: {has_cuda}\n")
    print(f"{'design':>13} {'q':>7} {'nnz(M)':>9} | {'splu fac+slv':>13} {'CG solve(it)':>14} "
          f"{'matvec CPU':>11} {'matvec GPU':>11} | {'logdet via factor':>18}")

    for na, nb, nobs in [(1000, 500, 40000), (3000, 1500, 100000),
                         (5000, 3000, 200000), (10000, 5000, 400000)]:
        a, b, n_a, n_b = gen_crossed(na, nb, nobs, seed=na)
        ZLam = build_ZLam(a, b, n_a, n_b, theta)
        q = ZLam.shape[1]
        Zt = ZLam.T.tocsc()
        rng = np.random.default_rng(0)
        rhs = rng.standard_normal(q)

        # matvec operator (no M formed)
        def matvec(v):
            return v + Zt @ (ZLam @ v)

        # (1) direct: form M, splu factor + one solve
        t0 = time.perf_counter()
        M = (Zt @ ZLam + sp.identity(q, format="csc")).tocsc()
        try:
            lu = splu(M)
            _ = lu.solve(rhs)
            t_direct = time.perf_counter() - t0
            nnz = M.nnz
            # logdet cost = the factorization (already in t_direct); report fill
            fill = lu.L.nnz + lu.U.nnz
            logdet_note = f"{fill/1e6:.1f}M fill"
        except Exception as e:
            t_direct = float("nan"); nnz = M.nnz; logdet_note = "FAILED"

        # (2) CG solve via matvec
        it = [0]
        def cb(xk): it[0] += 1
        A = LinearOperator((q, q), matvec=matvec)
        t0 = time.perf_counter()
        _, info = cg(A, rhs, rtol=1e-8, maxiter=500, callback=cb)
        t_cg = time.perf_counter() - t0

        # (3) matvec throughput CPU (avg of 20)
        t0 = time.perf_counter()
        for _ in range(20):
            matvec(rhs)
        t_mv_cpu = (time.perf_counter() - t0) / 20 * 1000

        # GPU matvec via torch sparse
        if has_cuda:
            idx = torch.tensor(np.vstack(ZLam.nonzero()), dtype=torch.long, device="cuda")
            valc = torch.tensor(ZLam.data, dtype=torch.float32, device="cuda")
            ZLg = torch.sparse_coo_tensor(idx, valc, ZLam.shape).coalesce()
            ZTg = ZLg.transpose(0, 1).coalesce()
            vg = torch.tensor(rhs, dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            # warmup
            _ = vg + torch.sparse.mm(ZTg, torch.sparse.mm(ZLg, vg.unsqueeze(1)).to_dense() if False else (ZLg @ vg).unsqueeze(1)).squeeze(1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                mv = vg + (ZTg @ (ZLg @ vg))
            torch.cuda.synchronize()
            t_mv_gpu = (time.perf_counter() - t0) / 50 * 1000
            del ZLg, ZTg, vg; torch.cuda.empty_cache()
        else:
            t_mv_gpu = float("nan")

        print(f"{na:>5}x{nb:<6} {q:>7} {nnz:>9} | {t_direct*1000:>11.1f}ms "
              f"{t_cg*1000:>9.1f}ms({it[0]:>3}) {t_mv_cpu:>9.3f}ms {t_mv_gpu:>9.3f}ms | {logdet_note:>18}")

    print()
    print("KEY POINT: CG/GPU-matvec gives the SOLVE cheaply, but the profiled")
    print("deviance also needs log|det M|. The only ways to get it are (a) the")
    print("sparse factorization (the 'splu fac' column = the fill-in cost CG was")
    print("avoiding), or (b) a STOCHASTIC trace estimator on the REML gradient")
    print("(AI-REML / BOLT-style) -> a different, non-deterministic algorithm.")


if __name__ == "__main__":
    main()
