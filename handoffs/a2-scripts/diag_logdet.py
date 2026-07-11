"""Is mgcv's REML determinant the Newton-weight one? Test: does
0.5*(log|A_newton| - log|A_fisher|) equal the 0.03359 score delta?"""
import warnings, numpy as np
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots, stack_penalty
from pystatistics.gam._smooth import s
from pystatistics.gam._gradient import _penalty_terms
from pystatistics.gam._gradient_glm import _eta_derivatives
from pystatistics.regression.families import Binomial

d = np.genfromtxt("fs_binomial.csv", delimiter=",", names=True)
y = d["y"].astype(float)
fam = Binomial(link="probit")
X, built = build_design(np.ones((len(y),1)), {"x1": d["x1"], "x2": d["x2"]},
                        [s("x1",k=10,bs="cr"), s("x2",k=8,bs="cr")])
roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
lam = np.array([40.00448, 23.14561])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = fit_fixed_lambda(y, X, roots, lam, fam, 1e-13, 400)
_, s_lam = _penalty_terms(roots, lam, X.shape[1])
u, omega, w_newton = _eta_derivatives(fam, y, fit.eta)
A_f = (X * fit.w[:, None]).T @ X + s_lam
A_n = (X * w_newton[:, None]).T @ X + s_lam
sf = np.linalg.slogdet(A_f)
sn = np.linalg.slogdet(A_n)
print("sign_f", sf[0], "logdet_f", sf[1])
print("sign_n", sn[0], "logdet_n", sn[1])
print("0.5*(logdet_f - logdet_n) =", 0.5*(sf[1]-sn[1]))
print("observed score delta py - mgcv =", 224.53873327 - 224.50514541)
