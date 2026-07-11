"""Rank-deficient (concurvity) gradient check: s(x1) + s(x2) with x2 == x1."""
import warnings
import numpy as np
from proto_grad import (glm_gradient, fd_gradient, TOL_INNER, MAXIT, RNG)
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._criteria import initial_log_lambdas
from pystatistics.gam._smooth import s
from pystatistics.regression.families import resolve_family

n = 300
x1 = np.sort(RNG.uniform(0, 1, n))
x2 = x1.copy()                      # exact concurvity -> rank deficiency
f = 1.4 * np.sin(2 * np.pi * x1)
y = RNG.poisson(np.exp(f - f.mean() + 1.0)).astype(float)
fam = resolve_family("poisson")
X_aug, built = build_design(np.ones((n, 1)), {"x1": x1, "x2": x2},
                            [s("x1", k=10, bs="cr"), s("x2", k=10, bs="cr")])
roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
rho0 = initial_log_lambdas(X_aug, roots)
for method in ("UBRE", "REML"):
    for off in (0.0, 2.0):
        rho = rho0 + off
        lam = np.exp(rho)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_fixed_lambda(y, X_aug, roots, lam, fam, TOL_INNER, MAXIT)
            ga = glm_gradient(fit, roots, lam, y, X_aug, fam, method)
            gf = fd_gradient(rho, y, X_aug, roots, fam, method)
        rel = np.max(np.abs(ga - gf) / np.maximum(np.abs(gf), 1e-8))
        print(f"rankdef {method} off={off:+.1f}: rank={fit.rank}/{X_aug.shape[1]}"
              f"  rel={rel:.2e}  ga={np.round(ga,6)}  gf={np.round(gf,6)}")
