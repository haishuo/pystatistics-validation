"""Same 40 torture configs through the reconstructed 4.6.x FD path
(warm-started FD objective, final FRESH refit — exactly the old flow)."""
import warnings, numpy as np
from scipy.optimize import minimize
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._criteria import (gcv_score, ubre_score,
                                        initial_log_lambdas)
from pystatistics.gam._edf import influence_matrix, total_edf
from pystatistics.gam._smooth import s
from pystatistics.regression.families import Binomial, Gaussian

def crit_of(fit, n, fixed):
    edf = total_edf(influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank))
    return (ubre_score(fit.deviance, n, edf, 1.0) if fixed
            else gcv_score(fit.deviance, n, edf))

def fd_select_and_final(y, X, roots, fam, fixed):
    """4.6.x flow: warm FD search + FRESH final fit at user tol."""
    n = len(y)
    warm = {"mu": None}
    def obj(rho):
        try:
            fit = fit_fixed_lambda(y, X, roots, np.exp(rho), fam, 1e-12, 200,
                                   mu_start=warm["mu"])
        except Exception:
            return np.inf
        warm["mu"] = fit.mu
        return crit_of(fit, n, fixed)
    r0 = initial_log_lambdas(X, roots)
    ref = max(abs(obj(r0)), 1e-300)
    res = minimize(lambda r: obj(r)/ref, r0, method="L-BFGS-B",
                   bounds=[(v-15, v+15) for v in r0],
                   options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-4})
    # 4.6.x final fit: FRESH at user tol (1e-8)
    fit = fit_fixed_lambda(y, X, roots, np.exp(res.x), fam, 1e-8, 200)
    return crit_of(fit, n, fixed), bool(res.success)

def ref_search(y, X, roots, fam, fixed):
    n = len(y)
    def obj(rho):
        try:
            fit = fit_fixed_lambda(y, X, roots, np.exp(rho), fam, 1e-12, 400)
        except Exception:
            return np.inf
        return crit_of(fit, n, fixed)
    r0 = initial_log_lambdas(X, roots)
    ref = max(abs(obj(r0)), 1e-300)
    res = minimize(lambda r: obj(r)/ref, r0, method="L-BFGS-B",
                   bounds=[(v-15, v+15) for v in r0],
                   options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-4})
    return obj(res.x)

silent, ok, crashes = 0, 0, 0
n = 60
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for seed in range(14, 34):
        for fam_key in ("gausslog", "probit"):
            rng = np.random.default_rng(seed)
            x1 = np.sort(rng.uniform(0, 1, n)); x2 = rng.uniform(0, 1, n)
            sig = 5.0 if fam_key == "gausslog" else 2.5
            f = sig * (np.sin(2*np.pi*x1) + np.cos(2*np.pi*x2))
            if fam_key == "gausslog":
                y = np.exp(0.6*f/ (2.0 if sig==5.0 else 1.0) * 2.0) * (1 + rng.normal(0, 0.3, n))
                fam = Gaussian(link="log"); fixed = False
            else:
                y = rng.binomial(1, 1/(1+np.exp(-f))).astype(float)
                fam = Binomial(link="probit"); fixed = True
            X, built = build_design(np.ones((n,1)), {"x1": x1, "x2": x2},
                                    [s("x1",k=12,bs="cr"), s("x2",k=10,bs="cr")])
            roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
            try:
                crit, success = fd_select_and_final(y, X, roots, fam, fixed)
                refv = ref_search(y, X, roots, fam, fixed)
                deficit = (crit - refv) > 0.05 * max(abs(refv), 1e-3)
                if deficit and success:
                    silent += 1
                    print(f"FD-PATH SILENT DEFICIT seed={seed} {fam_key}: reported={crit:.4f} ref={refv:.4f}")
                else:
                    ok += 1
            except Exception as e:
                crashes += 1
                print(f"FD-PATH CRASH seed={seed} {fam_key}: {type(e).__name__}")
print(f"\nFD-baseline scan: ok={ok} SILENT={silent} crashes={crashes}")
