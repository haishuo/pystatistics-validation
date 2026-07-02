"""Near-rank-deficient basis (high-k spline analogue) + small lambda: the exact
cond~1e17 regime the gpu_pirls docstring flags. Show fp32 X'WX destroys EDF while
fp64 preserves it -> the CF-1 silent-wrong band, and why the decisive solve MUST
be fp64."""
import numpy as np

def near_rank_deficient(n, p=40, decay=0.55, seed=1):
    # Columns with geometrically decaying singular values -> the spectrum a
    # high-k cubic spline basis on clustered x actually has (many near-null dirs).
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    Vt, _ = np.linalg.qr(rng.standard_normal((p, p)))
    sv = decay ** np.arange(p)               # 1, .55, .3, ... -> tiny
    X = (U * sv) @ Vt
    return X

def edf_check(n, p=40, lam=1e-6, seed=1):
    X = near_rank_deficient(n, p, seed=seed)
    w = np.ones(n)
    # 2nd-difference penalty WITH its natural null space (constant+linear unpenalized)
    D = np.zeros((p-2, p))
    for i in range(p-2):
        D[i, i] = 1.0; D[i, i+1] = -2.0; D[i, i+2] = 1.0
    S = D.T @ D                               # rank p-2, null space dim 2
    XtWX64 = (X.T * w) @ X
    X32 = X.astype(np.float32)
    XtWX32 = ((X32.T * X32) @ np.eye(1, dtype=np.float32).sum()  # keep fp32
              ).astype(np.float64) if False else ((X32.T) @ X32).astype(np.float64)
    A64 = XtWX64 + lam * S
    A32 = XtWX32 + lam * S
    cond = np.linalg.cond(A64)
    # true EDF via fp64
    edf64 = np.trace(np.linalg.solve(A64, XtWX64))
    # fp32-gram EDF (fp64 solve on the fp32-corrupted gram -- promotion can't heal it)
    edf32 = np.trace(np.linalg.solve(A32, XtWX32))
    return cond, edf64, edf32, abs(edf32-edf64)

print(f"{'n':>8} {'lam':>8} {'cond(A)':>11} {'EDF_fp64':>10} {'EDF_fp32gram':>13} {'abs_err':>10}")
for lam in [1e-2, 1e-4, 1e-6, 1e-8]:
    for n in [5000, 100000]:
        cond, e64, e32, err = edf_check(n, lam=lam)
        print(f"{n:>8} {lam:>8.0e} {cond:>11.2e} {e64:>10.4f} {e32:>13.4f} {err:>10.3f}")
