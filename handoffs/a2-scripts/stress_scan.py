"""Stressed-region scan: n=60 torture configs. For each, compare gam()'s
REPORTED criterion (of its reported fit) against an FD-reference search.
A 'silent deficit' = converged=True + no warning + reported criterion
materially worse than the reference. Expect ZERO after the branch fix."""
import warnings, numpy as np
from scipy.optimize import minimize
from pystatistics.gam import gam, s
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._criteria import (gcv_score, ubre_score,
                                        initial_log_lambdas)
from pystatistics.gam._edf import influence_matrix, total_edf
from pystatistics.regression.families import Binomial, Gaussian

def ref_search(y, X, roots, fam, fixed_disp):
    n = len(y)
    def obj(rho):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = fit_fixed_lambda(y, X, roots, np.exp(rho), fam, 1e-12, 400)
            except Exception:
                return np.inf
        edf = total_edf(influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank))
        return (ubre_score(fit.deviance, n, edf, 1.0) if fixed_disp
                else gcv_score(fit.deviance, n, edf))
    r0 = initial_log_lambdas(X, roots)
    ref = max(abs(obj(r0)), 1e-300)
    res = minimize(lambda r: obj(r)/ref, r0, method="L-BFGS-B",
                   bounds=[(v-15, v+15) for v in r0],
                   options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-4})
    return obj(res.x)

silent, honest, ok, crashes = 0, 0, 0, 0
n = 60
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
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                sol = gam(y, smooths=[s("x1",k=12,bs="cr"), s("x2",k=10,bs="cr")],
                          smooth_data={"x1": x1, "x2": x2}, family=fam, method="GCV")
            crit = sol.ubre if fixed else sol.gcv
            refv = ref_search(y, X, roots, fam, fixed)
            conv_warned = any("converge" in str(w.message) for w in wl)
            deficit = (crit - refv) > 0.05 * max(abs(refv), 1e-3)
            if deficit and sol.outer_converged and not conv_warned:
                silent += 1
                print(f"SILENT DEFICIT seed={seed} {fam_key}: reported={crit:.4f} ref={refv:.4f}")
            elif deficit:
                honest += 1
            else:
                ok += 1
        except Exception as e:
            crashes += 1
            print(f"CRASH seed={seed} {fam_key}: {type(e).__name__}")
print(f"\nscan: ok={ok} honest-flagged={honest} SILENT={silent} crashes={crashes}")
