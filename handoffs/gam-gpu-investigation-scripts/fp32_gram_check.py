"""Does forming X'WX in fp32 corrupt the penalized normal matrix at large n?
The GAM EDF/hat-trace depends on the SMALLEST eigenvalue of A = X'WX + lam*S,
which is ~1e-17 conditioned when lam is small. fp32 accumulation error in X'WX
scales ~ sqrt(n)*eps_fp32; we check whether that error swamps lam*S (the CF-1
exposure), and how it grows with n."""
import numpy as np

def check(n, p=50, lam=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    for j in range(1, p):
        X[:, j] = 0.6 * X[:, j] + 0.4 * X[:, j-1]   # spline-like collinearity
    w = rng.uniform(0.5, 1.5, n)
    S = np.zeros((p, p))
    for i in range(p):
        S[i, i] = 2.0
        if i+1 < p:
            S[i, i+1] = S[i+1, i] = -1.0
    # fp64 reference X'WX
    XtWX64 = (X.T * w) @ X
    # fp32 formation (what an fp32 GPU GEMM does: fp32 inputs, fp32 accumulate)
    X32 = X.astype(np.float32); w32 = w.astype(np.float32)
    XtWX32 = ((X32.T * w32) @ X32).astype(np.float64)
    rel_gram = np.linalg.norm(XtWX32 - XtWX64) / np.linalg.norm(XtWX64)
    A64 = XtWX64 + lam * S
    A32 = XtWX32 + lam * S
    cond = np.linalg.cond(A64)
    # EDF hat-trace, the quantity GAM reports
    edf64 = np.trace(np.linalg.solve(A64, XtWX64))
    edf32 = np.trace(np.linalg.solve(A32, XtWX32))
    edf_relerr = abs(edf32 - edf64) / abs(edf64)
    return rel_gram, cond, edf64, edf_relerr

print(f"{'n':>8} {'relerr(XtWX_fp32)':>18} {'cond(A)':>12} {'edf_fp64':>10} {'edf_relerr_fp32':>16}")
for n in [500, 5000, 20000, 100000, 500000, 2000000]:
    rg, cond, edf, ee = check(n)
    print(f"{n:>8} {rg:>18.2e} {cond:>12.2e} {edf:>10.3f} {ee:>16.2e}")
