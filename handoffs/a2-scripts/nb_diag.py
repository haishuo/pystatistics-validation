import warnings, numpy as np
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._smooth import s
from pystatistics.gam._gradient import _penalty_terms
from pystatistics.gam._gradient_glm import _eta_derivatives
from pystatistics.gam._criteria import reml_score
from pystatistics.regression.families import NegativeBinomial

d = np.genfromtxt("fs_nb.csv", delimiter=",", names=True)
y = d["y"].astype(float)
fam = NegativeBinomial(theta=3.0)
X, built = build_design(np.ones((len(y),1)), {"x1": d["x1"], "x2": d["x2"]},
                        [s("x1",k=10,bs="cr"), s("x2",k=8,bs="cr")])
roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
lam = np.array([45.114822, 43.085251])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = fit_fixed_lambda(y, X, roots, lam, fam, 1e-13, 400)
score_py = reml_score(fit, y, fam, roots, lam)
_, s_lam = _penalty_terms(roots, lam, X.shape[1])
u, omega, w_newton = _eta_derivatives(fam, y, fit.eta)
A_f = (X * fit.w[:, None]).T @ X + s_lam
A_n = (X * w_newton[:, None]).T @ X + s_lam
half_delta = 0.5*(np.linalg.slogdet(A_f)[1] - np.linalg.slogdet(A_n)[1])
print(f"py nb @fixed sp: reml={score_py:.8f}  0.5*(logdetF-logdetN)={half_delta:.7f}")
