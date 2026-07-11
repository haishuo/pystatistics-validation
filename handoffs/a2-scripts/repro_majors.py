"""Reproduce the two confirmed majors and verify the fixes."""
import warnings, numpy as np
from pystatistics.gam import gam, s
from pystatistics.regression.families import Binomial, Gaussian

# --- Major 1: gaussian-log hysteresis (seed 16, n=60, signal 5.0) ---
rng = np.random.default_rng(16)
n = 60
x1 = np.sort(rng.uniform(0, 1, n)); x2 = rng.uniform(0, 1, n)
f = 5.0 * (np.sin(2*np.pi*x1) + np.cos(2*np.pi*x2))
y = np.exp(0.6*f) * (1 + rng.normal(0, 0.3, n))
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    sol = gam(y, smooths=[s("x1",k=12,bs="cr"), s("x2",k=10,bs="cr")],
              smooth_data={"x1": x1, "x2": x2},
              family=Gaussian(link="log"), method="GCV")
warned = [str(w.message)[:60] for w in wlist]
print(f"M1 gaussian-log seed16: lambdas={np.asarray(sol.lambdas)}  GCV={sol.gcv:.4f}  "
      f"outer_conv={sol.outer_converged}")
print(f"   warnings: {warned if warned else 'none'}")
print(f"   VERDICT: {'FIXED (good optimum)' if sol.gcv < 30 else ('FIXED (honest non-convergence)' if not sol.outer_converged or warned else 'STILL SILENT-WRONG')}")

# --- Major 2: binomial-probit crash (seed 14, n=60, signal 2.5) ---
rng = np.random.default_rng(14)
x1 = np.sort(rng.uniform(0, 1, n)); x2 = rng.uniform(0, 1, n)
f = 2.5 * (np.sin(2*np.pi*x1) + np.cos(2*np.pi*x2))
p = 1/(1+np.exp(-f))
y2 = rng.binomial(1, p).astype(float)
try:
    with warnings.catch_warnings(record=True) as wlist2:
        warnings.simplefilter("always")
        sol2 = gam(y2, smooths=[s("x1",k=12,bs="cr"), s("x2",k=10,bs="cr")],
                   smooth_data={"x1": x1, "x2": x2},
                   family=Binomial(link="probit"), method="GCV")
    print(f"M2 probit seed14: COMPLETED  lambdas={np.asarray(sol2.lambdas)}  "
          f"outer_conv={sol2.outer_converged}  edf={sol2.total_edf:.3f}")
    print(f"   warnings: {[str(w.message)[:70] for w in wlist2] or 'none'}")
    print("   VERDICT: FIXED (no crash)")
except Exception as e:
    print(f"M2 probit seed14: STILL CRASHES  {type(e).__name__}: {e}")
