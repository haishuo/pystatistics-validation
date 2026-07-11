import warnings, numpy as np
from pystatistics.gam import gam, s
from pystatistics.regression.families import Binomial
d = np.genfromtxt("fs_binomial.csv", delimiter=",", names=True)
y = d["y"].astype(float)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sol = gam(y, smooths=[s("x1",k=10,bs="cr"), s("x2",k=8,bs="cr")],
              smooth_data={"x1": d["x1"], "x2": d["x2"]},
              family=Binomial(link="probit"), method="REML",
              sp=[40.00448, 23.14561])
fit_r = np.genfromtxt("probit_t1_fitted.csv", delimiter=",", skip_header=1)
print(f"py probit @fixed sp: edf={sol.total_edf:.8f} dev={sol.deviance:.8f} "
      f"reml={sol.reml_score:.8f}")
print("fitted max abs diff vs mgcv:", float(np.max(np.abs(sol.fitted_values - fit_r))))
