"""Independent standard-cauchit ordinal MLE — a reference for polr(link='cauchit').

One job: fit the cumulative-link (proportional-odds) model with the Cauchy CDF as
the inverse link, by a neutral-start multi-restart optimizer, WITHOUT using
MASS::polr or pystatistics as the anchor. This exists because MASS::polr's cauchit
is an unreliable reference — its `optim` under-converges on the heavy-tailed Cauchy
likelihood and reports a strictly worse optimum than the true MLE (see run_newlinks
for the recorded evidence). A generic scipy optimizer on the standard cauchit
likelihood, restarted from several neutral starts, recovers the true MLE and is the
reference pystatistics is validated against.

Model (matching MASS::polr / pystatistics polr sign convention):
    P(Y <= j | x) = Fc(alpha_j - x'beta),   Fc(z) = 1/2 + atan(z)/pi   (Cauchy CDF)
    P(Y = j | x)  = Fc(alpha_j - eta) - Fc(alpha_{j-1} - eta)
with ordered thresholds alpha_1 < ... < alpha_{K-1}, alpha_0 = -inf, alpha_K = +inf.

# NON-DETERMINISTIC source isolated: the multi-restart perturbations use a SEEDED
# RNG (default seed=0), so the fit is reproducible; no other randomness is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def _cauchy_cdf(z: NDArray[np.floating]) -> NDArray[np.floating]:
    return 0.5 + np.arctan(z) / np.pi


def _unpack(params: NDArray[np.floating], K: int) -> tuple[NDArray, NDArray]:
    """(t1, log-gaps, beta) -> (ordered thresholds, beta), enforcing ordering."""
    t1 = params[0]
    gaps = np.exp(params[1:K - 1])
    thr = np.concatenate([[t1], t1 + np.cumsum(gaps)]) if K > 2 else np.array([t1])
    return thr, params[K - 1:]


def _neg_loglik(params: NDArray[np.floating], y: NDArray[np.integer],
                X: NDArray[np.floating], K: int) -> float:
    thr, beta = _unpack(params, K)
    eta = X @ beta
    n = len(y)
    F = np.zeros((n, K + 1))
    F[:, K] = 1.0
    for j in range(1, K):
        F[:, j] = _cauchy_cdf(thr[j - 1] - eta)
    probs = np.clip(F[np.arange(n), y + 1] - F[np.arange(n), y], 1e-12, 1.0)
    return float(-np.sum(np.log(probs)))


def cauchit_ordinal_mle(y: NDArray[np.integer], X: NDArray[np.floating],
                        n_levels: int, seed: int = 0) -> dict[str, Any]:
    """Independent cauchit-ordinal MLE. Returns loglik, coef (beta), thr (alpha).

    Input contract (fail-loud): y are integer codes 0..K-1 with every level present;
    X is finite (n, p) with no intercept column (the thresholds are the intercepts).
    """
    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)
    K = int(n_levels)
    if not np.isfinite(X).all():
        raise ValueError("cauchit_ordinal_mle: X has non-finite entries")
    present = set(np.unique(y).tolist())
    if present != set(range(K)):
        raise ValueError(f"cauchit_ordinal_mle: y levels {sorted(present)} != 0..{K - 1}")
    p = X.shape[1]
    rng = np.random.default_rng(seed)

    # Data-driven neutral start: invert the cauchit at beta=0 through the empirical
    # cumulative class frequencies (independent of any fitted engine).
    cum = np.cumsum(np.bincount(y, minlength=K) / len(y))[:K - 1]
    t = np.tan(np.pi * (np.clip(cum, 1e-4, 1 - 1e-4) - 0.5))
    log_gaps = np.log(np.diff(t) + 1e-6) if K > 2 else np.array([])
    starts = [np.concatenate([[t[0]], log_gaps, np.zeros(p)]),
              np.concatenate([[-0.5], np.zeros(K - 2), np.zeros(p)]),
              np.concatenate([[0.0], np.zeros(K - 2), np.zeros(p)])]
    starts += [np.concatenate([[rng.normal()], rng.normal(size=K - 2),
                               rng.normal(size=p) * 0.3]) for _ in range(6)]

    best = None
    for s in starts:
        r = minimize(_neg_loglik, s, args=(y, X, K), method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-10,
                              "maxiter": 60000, "maxfev": 60000})
        r = minimize(_neg_loglik, r.x, args=(y, X, K), method="BFGS",
                     options={"gtol": 1e-8, "maxiter": 5000})
        if best is None or r.fun < best.fun:
            best = r
    thr, beta = _unpack(best.x, K)
    return {"loglik": float(-best.fun), "coef": beta, "thr": thr}
