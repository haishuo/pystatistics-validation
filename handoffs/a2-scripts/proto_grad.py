"""Prototype: Wood-2011 implicit-derivative sp-gradient for GLM-family GAMs.

Verifies the analytic gradient of UBRE / GCV / fixed-dispersion REML w.r.t.
rho = log(lambda) against central finite differences of the ACTUAL criterion
evaluated through fit_fixed_lambda, across canonical and non-canonical links.
"""
import numpy as np

from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._edf import (
    influence_matrix, posterior_covariance, total_edf, logdet_penalized,
)
from pystatistics.gam._criteria import (
    reml_score, ubre_score, gcv_score, initial_log_lambdas,
)
from pystatistics.gam._gradient import _penalty_terms
from pystatistics.gam._smooth import s
from pystatistics.regression.families import resolve_family

RNG = np.random.default_rng(7)
TOL_INNER = 1e-13
MAXIT = 400


def make_problem(family_name, link, n=350, m=2):
    x1 = np.sort(RNG.uniform(0, 1, n))
    x2 = RNG.uniform(0, 1, n)
    f = 1.4 * np.sin(2 * np.pi * x1) + (np.cos(2 * np.pi * x2) if m >= 2 else 0.0)
    if link:
        from pystatistics.regression.families import (
            Binomial, GammaFamily, Gaussian, Poisson,
        )
        cls = {"binomial": Binomial, "Gamma": GammaFamily,
               "gaussian": Gaussian, "poisson": Poisson}[family_name]
        fam = cls(link=link)
    elif family_name == "negative.binomial":
        from pystatistics.regression.families import NegativeBinomial
        fam = NegativeBinomial(theta=3.0)
    else:
        fam = resolve_family(family_name)
    if family_name == "poisson":
        y = RNG.poisson(np.exp(f - f.mean() + 1.0)).astype(float)
    elif family_name == "binomial":
        eta = 2.0 * f
        p = 1.0 / (1.0 + np.exp(-eta))
        y = RNG.binomial(1, p).astype(float)
    elif family_name == "Gamma":
        mu = np.exp(0.8 * f + 1.0)
        y = RNG.gamma(shape=4.0, scale=mu / 4.0)
    elif family_name == "negative.binomial":
        mu = np.exp(f - f.mean() + 1.0)
        theta = 3.0
        y = RNG.negative_binomial(theta, theta / (theta + mu)).astype(float)
    elif family_name == "gaussian":
        # positive response for log-link gaussian
        y = np.exp(0.6 * f) * (1.0 + RNG.normal(0, 0.08, n))
    else:
        raise ValueError(family_name)
    smooths = [s("x1", k=10, bs="cr")] + ([s("x2", k=8, bs="cr")] if m >= 2 else [])
    sd = {"x1": x1, "x2": x2}
    X_aug, built = build_design(np.ones((n, 1)),
                                {k: sd[k] for k in ("x1", "x2")[:m]}, smooths)
    blocks = [b.block for b in built]
    roots = make_penalty_roots([b.S_block for b in built], blocks)
    return y, X_aug, roots, fam


# --------------------------------------------------------------------------
# Analytic gradient (prototype)
# --------------------------------------------------------------------------

def _fd_eta_derivs(fam, y, eta, h_scale=1e-5):
    """Central-difference d(fisher weight)/d eta and Newton weight -du/deta.

    w(eta)  = mu_eta(eta)^2 / V(linkinv(eta))
    u(eta)  = (y - linkinv(eta)) * mu_eta(eta) / V(linkinv(eta))
    """
    h = h_scale * np.maximum(np.abs(eta), 1.0)

    def w_of(e):
        mu = fam.link.linkinv(e)
        me = fam.link.mu_eta(e)
        return me * me / np.maximum(fam.variance(mu), 1e-300)

    def u_of(e):
        mu = fam.link.linkinv(e)
        me = fam.link.mu_eta(e)
        return (y - mu) * me / np.maximum(fam.variance(mu), 1e-300)

    omega = (w_of(eta + h) - w_of(eta - h)) / (2.0 * h)
    w_newton = -(u_of(eta + h) - u_of(eta - h)) / (2.0 * h)
    return omega, w_newton


def glm_gradient(fit, roots, lambdas, y, X, fam, method, use_newton=True):
    """d criterion / d rho for the GLM path. method in {REML, UBRE, GCV}."""
    n = y.shape[0]
    p = fit.R.shape[0]
    terms, s_lam = _penalty_terms(roots, lambdas, p)
    a_inv = posterior_covariance(fit.R, fit.piv, fit.rank, 1.0)
    H = influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank)
    beta, eta, mu, w = fit.beta, fit.eta, fit.mu, fit.w
    edf = total_edf(H)

    me = fam.link.mu_eta(eta)
    u = (y - mu) * me / np.maximum(fam.variance(mu), 1e-300)
    omega, w_newton = _fd_eta_derivs(fam, y, eta)

    # dbeta/drho_j = -lambda_j * Atilde^{-1} S_j beta  (Newton weights)
    if use_newton:
        XtWtX = (X * w_newton[:, None]).T @ X
        A_t = XtWtX + s_lam
        # rank-aware: solve on kept coords
        kept = np.asarray(fit.piv[: fit.rank])
        A_kk = A_t[np.ix_(kept, kept)]
        def ainv_t(v):
            out = np.zeros_like(v)
            out[kept] = np.linalg.solve(A_kk, v[kept])
            return out
    else:
        def ainv_t(v):
            return a_inv @ v

    # n-space diagonals via M = X A^{-1} (original order)
    M = X @ a_inv                      # (n, p)
    a_diag = np.einsum("ij,ij->i", M, X)              # (X A^-1 X')_ii
    s_diag = np.einsum("ij,jk,ik->i", M, s_lam, M)    # (X A^-1 S_lam A^-1 X')_ii

    grads = np.empty(len(terms))
    K = fit.R_x.T @ fit.R_x  # X'WX
    for j, (lam, sj, rank_j) in enumerate(terms):
        sj_beta = sj @ beta
        dbeta = -lam * ainv_t(sj_beta)
        deta = X @ dbeta
        dD = -2.0 * float(u @ deta)
        # d edf: -lam tr(Ainv Sj H) + sum_i omega_i deta_i s_diag_i
        tr_term = -lam * float(np.einsum("ab,ba->", a_inv @ sj, H))
        dedf = tr_term + float(np.sum(omega * deta * s_diag))
        if method == "UBRE":
            grads[j] = dD / n + 2.0 * dedf / n
        elif method == "GCV":
            tau = n - edf
            grads[j] = n * dD / tau**2 + 2.0 * n * fit.deviance * dedf / tau**3
        elif method == "REML":
            # fixed dispersion phi=1
            b_sj_b = float(beta @ sj_beta)
            tr_ainv_sj = float(np.einsum("ab,ba->", a_inv, sj))
            dlogdetA = lam * tr_ainv_sj + float(np.sum(omega * deta * a_diag))
            grads[j] = 0.5 * lam * b_sj_b + 0.5 * dlogdetA - 0.5 * rank_j
        else:
            raise ValueError(method)
    return grads


# --------------------------------------------------------------------------
# FD reference of the actual criterion
# --------------------------------------------------------------------------

def criterion_at(rho, y, X, roots, fam, method):
    lam = np.exp(rho)
    fit = fit_fixed_lambda(y, X, roots, lam, fam, TOL_INNER, MAXIT)
    n = y.shape[0]
    if method == "REML":
        return reml_score(fit, y, fam, roots, lam)
    H = influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank)
    edf = total_edf(H)
    if method == "UBRE":
        return ubre_score(fit.deviance, n, edf, 1.0)
    return gcv_score(fit.deviance, n, edf)


def fd_gradient(rho, y, X, roots, fam, method, eps=1e-5):
    g = np.empty_like(rho)
    for j in range(len(rho)):
        rp, rm = rho.copy(), rho.copy()
        rp[j] += eps
        rm[j] -= eps
        g[j] = (criterion_at(rp, y, X, roots, fam, method)
                - criterion_at(rm, y, X, roots, fam, method)) / (2 * eps)
    return g


def check(family_name, link, method, m=2, rho_offsets=(0.0, -3.0, 2.5)):
    y, X, roots, fam = make_problem(family_name, link, m=m)
    rho0 = initial_log_lambdas(X, roots)
    print(f"--- {family_name}/{link or 'default'} {method} m={m} ---")
    for off in rho_offsets:
        rho = rho0 + off
        lam = np.exp(rho)
        fit = fit_fixed_lambda(y, X, roots, lam, fam, TOL_INNER, MAXIT)
        # verify the fixed point holds (score residual small)
        me_ = fam.link.mu_eta(fit.eta)
        u_ = (y - fit.mu) * me_ / np.maximum(fam.variance(fit.mu), 1e-300)
        _, s_lam_ = _penalty_terms(roots, lam, X.shape[1])
        score_res = np.linalg.norm(X.T @ u_ - s_lam_ @ fit.beta)
        if not fit.converged or score_res > 1e-6:
            print(f"  off={off:+.1f}: WARNING conv={fit.converged} "
                  f"n_iter={fit.n_iter} score_res={score_res:.2e}")
        ga = glm_gradient(fit, roots, lam, y, X, fam, method)
        gf = fd_gradient(rho, y, X, roots, fam, method)
        ga_fisher = glm_gradient(fit, roots, lam, y, X, fam, method,
                                 use_newton=False)
        denom = np.maximum(np.abs(gf), 1e-10)
        rel = np.max(np.abs(ga - gf) / denom)
        rel_f = np.max(np.abs(ga_fisher - gf) / denom)
        print(f"  off={off:+.1f}: newton_rel={rel:.2e}  fisher_rel={rel_f:.2e}"
              f"   fd={np.array2string(gf, precision=4)}")


def check_separation():
    """Binomial with quasi-separation: mu clamps active."""
    n = 200
    x1 = np.sort(RNG.uniform(0, 1, n))
    y = (x1 > 0.5).astype(float)          # perfectly separable in x1
    y[RNG.choice(n, 4, replace=False)] = 1 - y[RNG.choice(n, 4, replace=False)]
    fam = resolve_family("binomial")
    X_aug, built = build_design(np.ones((n, 1)), {"x1": x1},
                                [s("x1", k=10, bs="cr")])
    roots = make_penalty_roots([b.S_block for b in built],
                               [b.block for b in built])
    rho0 = initial_log_lambdas(X_aug, roots)
    print("--- binomial near-separation UBRE ---")
    import warnings
    for off in (0.0, -4.0):
        rho = rho0 + off
        lam = np.exp(rho)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_fixed_lambda(y, X_aug, roots, lam, fam, TOL_INNER, MAXIT)
            ga = glm_gradient(fit, roots, lam, y, X_aug, fam, "UBRE")
            gf = fd_gradient(rho, y, X_aug, roots, fam, "UBRE")
        rel = np.max(np.abs(ga - gf) / np.maximum(np.abs(gf), 1e-10))
        print(f"  off={off:+.1f}: newton_rel={rel:.2e}  fd={gf}")


if __name__ == "__main__":
    check("poisson", None, "UBRE")
    check("poisson", None, "REML")
    check("binomial", None, "UBRE")
    check("binomial", None, "REML")
    check("binomial", "probit", "UBRE")     # non-canonical
    check("binomial", "probit", "REML")
    check("Gamma", "log", "GCV")            # non-canonical, free dispersion
    check("Gamma", None, "GCV")             # canonical inverse
    check("gaussian", "log", "GCV")         # non-canonical gaussian
    check("negative.binomial", None, "REML")  # NB fixed theta
    check("negative.binomial", None, "UBRE")
    check("poisson", None, "UBRE", m=1)     # single smooth
    check("poisson", None, "REML", m=2, rho_offsets=(-8.0, 8.0))  # extremes
    check_separation()
