"""Library-level verification of the Newton-REML fix."""
import warnings, numpy as np
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._criteria import reml_score
from pystatistics.gam._smooth import s
from pystatistics.regression.families import (Binomial, NegativeBinomial,
                                              resolve_family)

def score_at(csv, fam, lam, k2=8):
    d = np.genfromtxt(csv, delimiter=",", names=True)
    y = d["y"].astype(float)
    X, built = build_design(np.ones((len(y),1)), {"x1": d["x1"], "x2": d["x2"]},
                            [s("x1",k=10,bs="cr"), s("x2",k=k2,bs="cr")])
    roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_fixed_lambda(y, X, roots, np.asarray(lam), fam, 1e-13, 400)
    return reml_score(fit, y, X, fam, roots, np.asarray(lam))

# probit @ mgcv sp: mgcv reml = 224.50514541
v = score_at("fs_binomial.csv", Binomial(link="probit"), [40.00448, 23.14561])
print(f"probit  fixed-sp REML: py={v:.8f}  mgcv=224.50514541  diff={v-224.50514541:+.2e}")
# nb @ mgcv sp: mgcv reml = 1072.52450991
v = score_at("fs_nb.csv", NegativeBinomial(theta=3.0), [45.114822, 43.085251])
print(f"nb      fixed-sp REML: py={v:.8f}  mgcv=1072.52450991  diff={v-1072.52450991:+.2e}")
# CANONICAL regression: poisson @ mgcv sp (from the very first experiment:
# mgcv REML score 750.7404787651 at sp=36.739120367202709 on data.csv)
d = np.genfromtxt("data.csv", delimiter=",", names=True)
y = d["y"].astype(float)
X, built = build_design(np.ones((len(y),1)), {"x": d["x"]}, [s("x",k=10,bs="cr")])
roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
fam = resolve_family("poisson")
fit = fit_fixed_lambda(y, X, roots, np.array([36.739120367202709]), fam, 1e-13, 400)
v = reml_score(fit, y, X, fam, roots, np.array([36.739120367202709]))
print(f"poisson fixed-sp REML: py={v:.10f}  mgcv=750.7404787651  diff={v-750.7404787651:+.2e}  (canonical, must stay ~1e-10)")
