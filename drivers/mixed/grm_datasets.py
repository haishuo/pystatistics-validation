"""Deterministic simulated designs for the GRM / low-rank mixed model (grm_lmm).

One job: generate genomic-style `(y, X, W)` designs with a KNOWN heritability, so
`grm_lmm` (K = W Wᵀ / M) can be compared against R `rrBLUP::mixed.solve` (the
canonical GBLUP/GRM reference) and against its own fp64 optimum. Synthetic +
deterministic (seeded), generated in-driver and handed to R via ephemeral CSVs —
parameterised stress designs are not stored in the central HDF5 store (the
multivariate/LMM-scaling precedent; R17).

Model: y = Xβ + g + e, with the genetic value g = W a / √M, a ~ N(0, σ_g² I_M),
so Cov(g) = σ_g² · W Wᵀ / M = σ_g² K — exactly the covariance grm_lmm fits. The
residual e ~ N(0, σ_e²). Heritability h² = σ_g² / (σ_g² + σ_e²).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GRMDataset:
    key: str
    y: NDArray            # (n,)
    X: NDArray            # (n, p) fixed design (col 0 = intercept)
    W: NDArray            # (n, M) low-rank factor; K = W Wᵀ / M
    K: NDArray            # (n, n) GRM = W Wᵀ / M (handed to rrBLUP)
    fixed_names: list[str]
    true_beta: NDArray
    true_h2: float
    reml: bool = True
    why: str = ""

    @property
    def n(self) -> int: return int(self.y.shape[0])
    @property
    def p(self) -> int: return int(self.X.shape[1])
    @property
    def M(self) -> int: return int(self.W.shape[1])


def make_grm_conditioned(*, n: int, M: int, h2: float, cond: float, seed: int,
                         p_fixed: int = 2, key: str | None = None,
                         reml: bool = True) -> GRMDataset:
    """Like :func:`make_grm` but with W's spectrum stretched to a target condition
    number ``cond`` — the CF-1 boundary sweep design. As ``cond`` grows past the
    fp32-safe threshold, the fp32 Gram Cholesky must either stay accurate or REFUSE
    loud (never silently wrong). Deterministic given the seed."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, M))
    U, _s, Vt = np.linalg.svd(A, full_matrices=False)
    k = _s.shape[0]
    # geometric spectrum spanning [1, cond]
    svals = np.geomspace(cond, 1.0, k)
    W = (U * svals) @ Vt
    K = W @ W.T / M
    sigma_g2 = float(h2); sigma_e2 = float(1.0 - h2)
    a = rng.normal(0.0, np.sqrt(sigma_g2), size=M)
    g = W @ a / np.sqrt(M)
    if p_fixed > 1:
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p_fixed - 1))])
    else:
        X = np.ones((n, 1))
    beta = np.array([1.0, 0.5, -0.75, 0.3][:p_fixed], dtype=float)
    y = X @ beta + g + rng.normal(0.0, np.sqrt(sigma_e2), size=n)
    names = ["(Intercept)"] + [f"x{i}" for i in range(1, p_fixed)]
    return GRMDataset(
        key=key or f"grm_cond{cond:.0e}_n{n}_M{M}",
        y=y, X=X, W=W, K=K, fixed_names=names,
        true_beta=beta, true_h2=float(h2), reml=reml,
        why=f"CF-1 boundary design: cond(W)≈{cond:.0e}, n={n}, M={M}")


def make_grm(*, n: int, M: int, h2: float, seed: int, p_fixed: int = 2,
             standardize: bool = True, key: str | None = None,
             reml: bool = True) -> GRMDataset:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, M))
    if standardize:                       # genomic-marker style column scaling
        W = (W - W.mean(0)) / W.std(0)
    K = W @ W.T / M
    sigma_g2 = float(h2)
    sigma_e2 = float(1.0 - h2)
    a = rng.normal(0.0, np.sqrt(sigma_g2), size=M)
    g = W @ a / np.sqrt(M)                # Cov(g) = sigma_g2 * K
    beta = np.array([1.0, 0.5, -0.75, 0.3][:p_fixed], dtype=float)
    if p_fixed > 1:
        Xcov = rng.standard_normal((n, p_fixed - 1))
        X = np.column_stack([np.ones(n), Xcov])
    else:
        X = np.ones((n, 1))
    e = rng.normal(0.0, np.sqrt(sigma_e2), size=n)
    y = X @ beta + g + e
    names = ["(Intercept)"] + [f"x{i}" for i in range(1, p_fixed)]
    return GRMDataset(
        key=key or f"grm_n{n}_M{M}_h{int(h2*100)}",
        y=y, X=X, W=W, K=K, fixed_names=names,
        true_beta=beta, true_h2=float(h2), reml=reml,
        why=f"simulated GRM: n={n}, M={M} markers, true h²={h2}")
