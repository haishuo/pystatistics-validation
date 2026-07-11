"""FD-baseline (reconstructed 4.6.x path) vs analytic path, same problems."""
import numpy as np, time
from scipy.optimize import minimize
import pystatistics.gam._criteria as crit
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import make_penalty_roots
from pystatistics.gam._smooth import s
from pystatistics.gam._edf import influence_matrix, total_edf
from pystatistics.regression.families import resolve_family

d = np.genfromtxt("bench.csv", delimiter=",", names=True)
y = d["y"].astype(float)
fam = resolve_family("poisson")

def problem(m):
    data = {f"x{j}": d[f"x{j}"] for j in range(1, m + 1)}
    sm = [s(f"x{j}", k=10, bs="cr") for j in range(1, m + 1)]
    X_aug, built = build_design(np.ones((len(y), 1)), data, sm)
    roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
    return X_aug, roots

def fd_select(y, X, roots, family, method, tol, max_iter):
    """The 4.6.x finite-difference path, reconstructed verbatim."""
    n = y.shape[0]
    warm = {"mu": None}
    tol_inner = min(tol, 1e-12)
    calls = {"n": 0}
    def objective(log_lam):
        calls["n"] += 1
        lam = np.exp(np.asarray(log_lam, dtype=np.float64))
        fit = crit.fit_fixed_lambda(y, X, roots, lam, family, tol_inner, max_iter,
                                    mu_start=warm["mu"])
        warm["mu"] = fit.mu
        h = influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank)
        edf = total_edf(h)
        if method == "REML":
            return crit.reml_score(fit, y, family, roots, lam)
        if family.dispersion_is_fixed:
            return crit.ubre_score(fit.deviance, n, edf, scale=1.0)
        return crit.gcv_score(fit.deviance, n, edf)
    log_lam0 = crit.initial_log_lambdas(X, roots)
    bounds = [(lo - 15.0, lo + 15.0) for lo in log_lam0]
    ref = max(abs(objective(log_lam0)), 1e-300)
    result = minimize(lambda r: objective(r) / ref, log_lam0, method="L-BFGS-B",
                      bounds=bounds,
                      options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-4})
    return np.exp(result.x), calls["n"]

print(f"{'m':>2} {'meth':>5} {'fd_time':>8} {'fd_fits':>7} {'an_time':>8} {'an_fits':>7} {'speedup':>7}")
for m in (1, 2, 3, 4, 5, 6):
    X_aug, roots = problem(m)
    for meth in ("REML", "GCV"):
        # FD baseline (best of 3)
        ts_fd, nf = [], None
        for _ in range(3):
            t = time.perf_counter()
            lam_fd, nf = fd_select(y, X_aug, roots, fam, meth, 1e-8, 200)
            ts_fd.append(time.perf_counter() - t)
        # analytic (instrument fit count)
        calls = {"n": 0}
        real = crit.fit_fixed_lambda
        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)
        crit.fit_fixed_lambda = counting
        ts_an = []
        for _ in range(3):
            calls["n"] = 0
            t = time.perf_counter()
            lam_an, conv = crit.select_lambdas(y, X_aug, roots, fam, meth, 1e-8, 200)
            ts_an.append(time.perf_counter() - t)
        crit.fit_fixed_lambda = real
        rel_lam = np.max(np.abs(lam_an - lam_fd) / np.abs(lam_fd))
        print(f"{m:>2} {meth:>5} {min(ts_fd):8.4f} {nf:>7} {min(ts_an):8.4f} {calls['n']:>7} "
              f"{min(ts_fd)/min(ts_an):6.2f}x  lam_rel_diff={rel_lam:.1e}")
