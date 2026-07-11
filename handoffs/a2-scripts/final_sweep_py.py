"""Final A2 R6 sweep: py free selection vs mgcv on identical data."""
import warnings, numpy as np
from pystatistics.gam import gam, s
from pystatistics.regression.families import Binomial, GammaFamily, Gaussian, NegativeBinomial

# mgcv references from final_sweep_mgcv.txt
REF = {
 ("poisson","REML"):        (14.321526, [54.179813,64.658228], 512.577863, None),
 ("poisson","GCV"):         (12.846850, [132.24633,140.23351], 514.408219, None),
 ("binomial","REML"):       (11.438415, [12.451084,7.3316055], 410.184683, None),
 ("binomial","GCV"):        (11.670757, [22.357007,2.6107527], 409.112524, None),
 ("binomial-probit","REML"):(11.343721, [40.00448,23.14561],   412.691145, None),
 ("binomial-probit","GCV"): (11.552354, [77.037508,7.8342859], 411.499333, None),
 ("Gamma-log","GCV"):       (12.680147, [97.41351,10.353024],  120.617990, None),
 ("gaussian-log","GCV"):    (15.211208, [18.001415,6.3893523], 6.853031,   None),
 ("nb-fixed","REML"):       (12.986883, [45.114822,43.085251], 507.629311, 3.0),
 ("nb-est","REML"):         (13.087811, [44.894335,43.143841], 526.744705, 3.297),
}
FAM = {
 "poisson": "poisson", "binomial": "binomial",
 "binomial-probit": Binomial(link="probit"), "Gamma-log": GammaFamily(link="log"),
 "gaussian-log": Gaussian(link="log"), "nb-fixed": NegativeBinomial(theta=3.0),
 "nb-est": "nb",
}
print(f"{'case':<22} {'edf_gap':>10} {'sp_rel':>9} {'fitted_max':>10} {'theta':>8}")
worst = 0.0
for (key, meth), (edf_r, sp_r, dev_r, th_r) in REF.items():
    d = np.genfromtxt(f"fs_{key}.csv", delimiter=",", names=True)
    y = d["y"].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = gam(y, smooths=[s("x1",k=10,bs="cr"), s("x2",k=8,bs="cr")],
                  smooth_data={"x1": d["x1"], "x2": d["x2"]},
                  family=FAM[key], method=meth)
    edf_gap = sol.total_edf - edf_r
    sp_rel = np.max(np.abs(np.asarray(sol.lambdas) - sp_r) / np.abs(sp_r))
    fit_r = np.genfromtxt(f"fs_{key}_{'GCV.Cp' if meth=='GCV' else meth}_fitted.csv",
                          delimiter=",", skip_header=1)
    fitted_max = float(np.max(np.abs(np.asarray(sol.fitted_values) - fit_r)))
    fitted_rel = fitted_max / float(np.max(np.abs(fit_r)))
    th = sol._result.info.get("nb_theta") if key == "nb-est" else None
    th_s = f"{th:.4f}" if th else "-"
    print(f"{key+'/'+meth:<22} {edf_gap:>+10.6f} {sp_rel:>9.2e} {fitted_rel:>10.2e} {th_s:>8}")
    worst = max(worst, abs(edf_gap))
print("worst |edf gap|:", worst)
