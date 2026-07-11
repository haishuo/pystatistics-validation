"""R10: separation-grade probit REML e2e vs mgcv (mgcv: sp=1.6584,
edf(smooth)=5.7003, reml=24.524750)."""
import warnings, numpy as np
from pystatistics.gam import gam, s
from pystatistics.regression.families import Binomial
d = np.genfromtxt("sep_probit.csv", delimiter=",", names=True)
y = d["y"].astype(float)
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    sol = gam(y, smooths=[s("x1", k=10, bs="cr")], smooth_data={"x1": d["x1"]},
              family=Binomial(link="probit"), method="REML")
msgs = sorted({str(w.message)[:55] for w in wl})
print(f"py: sp={float(np.asarray(sol.lambdas)[0]):.5f} (mgcv 1.6584)  "
      f"smooth_edf={sol.total_edf-1:.4f} (mgcv 5.7003)  "
      f"reml={sol.reml_score:.6f} (mgcv 24.524750)  conv={sol.outer_converged}")
print("warnings:", msgs)
