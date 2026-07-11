"""Is the fixed path's seed-16 answer a genuine stationary point of the
FRESH criterion surface? And what does mgcv select on this exact data?"""
import warnings, numpy as np
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._criteria import gcv_score
from pystatistics.gam._edf import influence_matrix, total_edf
from pystatistics.gam._smooth import s
from pystatistics.regression.families import Gaussian

rng = np.random.default_rng(16)
n = 60
x1 = np.sort(rng.uniform(0, 1, n)); x2 = rng.uniform(0, 1, n)
f = 5.0 * (np.sin(2*np.pi*x1) + np.cos(2*np.pi*x2))
y = np.exp(0.6*f) * (1 + rng.normal(0, 0.3, n))
np.savetxt("m1_seed16.csv", np.column_stack([y, x1, x2]), delimiter=",",
           header="y,x1,x2", comments="")
fam = Gaussian(link="log")
X, built = build_design(np.ones((n,1)), {"x1": x1, "x2": x2},
                        [s("x1",k=12,bs="cr"), s("x2",k=10,bs="cr")])
roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])

def fresh_gcv(lam):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_fixed_lambda(y, X, roots, np.asarray(lam), fam, 1e-12, 400)
    edf = total_edf(influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank))
    return gcv_score(fit.deviance, n, edf)

for name, lam in [("fixed-path answer", [1.53488801, 0.01108705]),
                  ("FD-reference", None)]:
    if lam is None:
        continue
    rho = np.log(np.asarray(lam))
    g = np.empty(2)
    for j in range(2):
        rp, rm = rho.copy(), rho.copy()
        rp[j] += 1e-4; rm[j] -= 1e-4
        g[j] = (fresh_gcv(np.exp(rp)) - fresh_gcv(np.exp(rm))) / 2e-4
    print(f"{name}: fresh GCV={fresh_gcv(lam):.4f}  fresh-FD grad={g}")
