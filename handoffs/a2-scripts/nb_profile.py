"""Is the profiled-UBRE criterion itself degenerate in theta?"""
import warnings, numpy as np
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._criteria import select_lambdas, ubre_score
from pystatistics.gam._edf import influence_matrix, total_edf
from pystatistics.gam._smooth import s
from pystatistics.regression.families import NegativeBinomial

rng = np.random.default_rng(17)
n = 300
x1 = np.sort(rng.uniform(0.0, 1.0, n))
f = 1.4 * np.sin(2 * np.pi * x1)
mu = np.exp(f - f.mean() + 1.0)
y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
X, built = build_design(np.ones((n,1)), {"x1": x1}, [s("x1",k=10,bs="cr")])
roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
for theta in (0.04, 0.5, 1.0, 3.0, 10.0):
    fam = NegativeBinomial(theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lam, conv = select_lambdas(y, X, roots, fam, "GCV", 1e-8, 200)
        fit = fit_fixed_lambda(y, X, roots, lam, fam, 1e-12, 200)
    edf = total_edf(influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank))
    print(f"theta={theta:6.2f}: UBRE={ubre_score(fit.deviance, n, edf, 1.0):.6f}  dev={fit.deviance:.2f}  edf={edf:.3f}")
