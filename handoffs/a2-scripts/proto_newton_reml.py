"""Prototype: Newton-determinant REML score at fixed sp vs mgcv's reported
score, for probit and nb (the two proven-divergent cases). Also verify the
Newton-REML *gradient* (with omega_newton via second central difference)
against criterion-FD of the NEW score."""
import warnings, numpy as np
from pystatistics.gam._basis import build_design
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._criteria import reml_score, initial_log_lambdas
from pystatistics.gam._gradient import _penalty_terms
from pystatistics.gam._smooth import s
from pystatistics.regression.families import Binomial, NegativeBinomial

_LOG2PI = float(np.log(2.0 * np.pi))

def eta_derivs(fam, y, eta, h1=1e-5, h2=1e-4):
    ha = h1 * np.maximum(np.abs(eta), 1.0)
    hb = h2 * np.maximum(np.abs(eta), 1.0)
    def u_of(e):
        mu = fam.link.linkinv(e)
        me = fam.link.mu_eta(e)
        return (y - mu) * me / np.maximum(fam.variance(mu), 1e-300)
    u0 = u_of(eta)
    w_newton = -(u_of(eta + ha) - u_of(eta - ha)) / (2.0 * ha)
    omega_newton = -(u_of(eta + hb) - 2.0*u0 + u_of(eta - hb)) / (hb*hb)
    return u0, w_newton, omega_newton

def newton_reml(fit, y, X, fam, roots, lam):
    """REML with log|X'WnX+S_lam| (Newton) instead of Fisher."""
    n = y.shape[0]
    p = fit.rank
    rank_s = sum(r.rank for r in roots)
    m_p = max(p - rank_s, 0)
    terms, s_lam = _penalty_terms(roots, lam, X.shape[1])
    u, w_n, _ = eta_derivs(fam, y, fit.eta)
    kept = np.asarray(fit.piv[:fit.rank])
    Xk = X[:, kept]
    A_n = (Xk * w_n[:, None]).T @ Xk + s_lam[np.ix_(kept, kept)]
    sign, logdet_a = np.linalg.slogdet(A_n)
    assert sign > 0, "non-PD Newton Hessian"
    logdet_s = float(sum(r.rank*np.log(l) + r.logdet_pos
                         for r, l in zip(roots, lam)))
    wt = np.ones(n)
    neg_ll = -fam.log_likelihood(y, fit.mu, wt, 1.0)
    return float(neg_ll + fit.penalty/2.0 + (logdet_a - logdet_s)/2.0
                 - (m_p/2.0)*_LOG2PI)

def newton_reml_grad(fit, y, X, fam, roots, lam):
    """Gradient of the Newton-determinant REML."""
    p_full = X.shape[1]
    terms, s_lam = _penalty_terms(roots, lam, p_full)
    u, w_n, om_n = eta_derivs(fam, y, fit.eta)
    kept = np.asarray(fit.piv[:fit.rank])
    Xk = X[:, kept]
    A_n = (Xk * w_n[:, None]).T @ Xk + s_lam[np.ix_(kept, kept)]
    A_n_inv = np.linalg.inv(A_n)
    beta = fit.beta
    Mk = Xk @ A_n_inv                    # (n, rank)
    a_diag = np.einsum("ij,ij->i", Mk, Xk)   # (X A_n^{-1} X')_ii
    grad = np.empty(len(terms))
    for j, (lj, sj, rank_j) in enumerate(terms):
        sj_beta = sj @ beta
        dbeta_k = -lj * (A_n_inv @ sj_beta[kept])
        deta = Xk @ dbeta_k
        b_sj_b = float(beta @ sj_beta)
        tr_term = lj * float(np.einsum("ab,ba->", A_n_inv, sj[np.ix_(kept,kept)]))
        dlogdet = tr_term + float(np.sum(om_n * deta * a_diag))
        grad[j] = 0.5*lj*b_sj_b + 0.5*dlogdet - 0.5*rank_j
    return grad

# ---- probit @ mgcv's selected sp ----
d = np.genfromtxt("fs_binomial.csv", delimiter=",", names=True)
y = d["y"].astype(float)
fam = Binomial(link="probit")
X, built = build_design(np.ones((len(y),1)), {"x1": d["x1"], "x2": d["x2"]},
                        [s("x1",k=10,bs="cr"), s("x2",k=8,bs="cr")])
roots = make_penalty_roots([b.S_block for b in built], [b.block for b in built])
lam = np.array([40.00448, 23.14561])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = fit_fixed_lambda(y, X, roots, lam, fam, 1e-13, 400)
v_new = newton_reml(fit, y, X, fam, roots, lam)
print(f"probit: newton_reml={v_new:.8f}  mgcv=224.50514541  diff={v_new-224.50514541:+.2e}")

# gradient vs FD of the NEW criterion
def crit(rho):
    l = np.exp(rho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fit_fixed_lambda(y, X, roots, l, fam, 1e-13, 400)
    return newton_reml(f, y, X, fam, roots, l)
rho0 = initial_log_lambdas(X, roots)
for off in (0.0, -3.0, 2.5):
    rho = rho0 + off
    l = np.exp(rho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fit_fixed_lambda(y, X, roots, l, fam, 1e-13, 400)
    ga = newton_reml_grad(f, y, X, fam, roots, l)
    gf = np.empty(2)
    for j in range(2):
        rp, rm = rho.copy(), rho.copy()
        rp[j] += 1e-5; rm[j] -= 1e-5
        gf[j] = (crit(rp) - crit(rm)) / 2e-5
    rel = np.max(np.abs(ga-gf)/np.maximum(np.abs(gf), 1e-8))
    print(f"  probit grad off={off:+.1f}: rel={rel:.2e}  ga={np.round(ga,5)} gf={np.round(gf,5)}")

# ---- nb @ mgcv's selected sp ----
d2 = np.genfromtxt("fs_nb.csv", delimiter=",", names=True)
y2 = d2["y"].astype(float)
fam2 = NegativeBinomial(theta=3.0)
X2, built2 = build_design(np.ones((len(y2),1)), {"x1": d2["x1"], "x2": d2["x2"]},
                          [s("x1",k=10,bs="cr"), s("x2",k=8,bs="cr")])
roots2 = make_penalty_roots([b.S_block for b in built2], [b.block for b in built2])
lam2 = np.array([45.114822, 43.085251])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit2 = fit_fixed_lambda(y2, X2, roots2, lam2, fam2, 1e-13, 400)
v2 = newton_reml(fit2, y2, X2, fam2, roots2, lam2)
print(f"nb:     newton_reml={v2:.8f}  mgcv=1072.52450991  diff={v2-1072.52450991:+.2e}")
