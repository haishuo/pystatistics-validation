"""Separation-grade probit REML: where does the Newton Hessian go non-PD?"""
import warnings, numpy as np
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._criteria import initial_log_lambdas
from pystatistics.gam._gradient import _penalty_terms
from pystatistics.gam._smooth import s
from pystatistics.regression.families import Binomial

rng = np.random.default_rng(3)
n = 200
x1 = np.sort(rng.uniform(0, 1, n))
y = (x1 > 0.5).astype(float)
flip = rng.choice(n, 4, replace=False)
y[flip] = 1.0 - y[flip]
np.savetxt("sep_probit.csv", np.column_stack([y, x1]), delimiter=",",
           header="y,x1", comments="")
fam = Binomial(link="probit")
X, built = build_design(np.ones((n,1)), {"x1": x1}, [s("x1",k=10,bs="cr")])
roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])

def u_of(e):
    mu = fam.link.linkinv(e); me = fam.link.mu_eta(e)
    return (y - mu) * me / np.maximum(fam.variance(mu), 1e-300)

rho0 = initial_log_lambdas(X, roots)
for off in (0.0, -4.0, -8.0, -12.0, 4.0):
    lam = np.exp(rho0 + off)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_fixed_lambda(y, X, roots, lam, fam, 1e-12, 400)
    _, s_lam = _penalty_terms(roots, lam, X.shape[1])
    h = 1e-5 * np.maximum(np.abs(fit.eta), 1.0)
    w_n = -(u_of(fit.eta + h) - u_of(fit.eta - h)) / (2*h)
    kept = np.asarray(fit.piv[:fit.rank])
    Xk = X[:, kept]
    A_n = (Xk * w_n[:, None]).T @ Xk + s_lam[np.ix_(kept, kept)]
    sign, ld = np.linalg.slogdet(A_n)
    ev_min = np.linalg.eigvalsh(A_n).min()
    print(f"off={off:+5.1f}: negNewtonW={int((w_n<0).sum()):3d}  sign={sign:+.0f}  "
          f"eig_min={ev_min:.3e}  conv={fit.converged}")
