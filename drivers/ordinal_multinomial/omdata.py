"""Curate the ordinal (polr) and multinomial (multinom) validation designs.

One job: turn the centralized HDF5 datasets (R17) into the exact numeric
(y, X) design both engines fit — and deterministically synthesize the red-team
hard cases (separation, unbalanced / rare / collapsed categories, scaling
sweeps) that are NOT fixed reference data.

Design-matrix conventions, matched to each library's contract:

- ``polr``: X carries NO intercept column (the ordered thresholds are the
  category-specific intercepts, exactly as ``MASS::polr`` drops the intercept).
- ``multinom``: the caller supplies the intercept, so X's first column is an
  explicit column of ones (matching ``nnet::multinom``'s formula intercept).

Factor covariates are expanded to treatment-contrast dummies (drop first level)
in Python, and the SAME numeric matrix is handed to R via the CSV the driver
dumps — so agreement is to fp64 round-off of identical numbers, never
contaminated by a design mismatch (the mixed/multivariate precedent). The store
is float32; continuous columns are promoted to float64 on load.

Source of truth is the centralized store (R17): ``housing.h5`` (ordinal),
``fgl.h5`` / ``iris_multinom.h5`` / ``multinom_synth.h5`` (multinomial) live in
the central store under the ``pystatistics`` namespace, reached via
``DATASETS_ROOT`` (resolution lives in ``drivers/_shared/store_io.py``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from store_io import DEFAULT_NAMESPACE, store_root  # noqa: E402


def _store_dir() -> Path:
    """Directory holding this project's curated datasets.

    Store resolution lives in ``drivers/_shared/store_io.py`` — the single place
    that knows where the store is. Do not reintroduce a local search path here.
    """
    return store_root() / DEFAULT_NAMESPACE


def _load_raw(stem: str) -> tuple[NDArray, list[str], list[str], list[dict]]:
    """Return (values fp64 (n,p), raw_names, data_types, category_maps)."""
    path = _store_dir() / f"{stem}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"{stem}.h5 not in store {path.parent}")
    with h5py.File(path, "r") as f:
        vals = f["data/values"][:].astype(np.float64)
        names = [x.decode() for x in f["columns/raw_names"][:]]
        dtypes = [x.decode() for x in f["columns/data_types"][:]]
        cmaps = [json.loads(x.decode()) for x in f["columns/category_maps"][:]]
    return vals, names, dtypes, cmaps


def _dummies(codes: NDArray[np.integer], k: int) -> NDArray[np.float64]:
    """Treatment-contrast dummies for a k-level factor: drop level 0, keep 1..k-1."""
    return np.column_stack([(codes == j).astype(np.float64) for j in range(1, k)])


# ----------------------------------------------------------------------------
# Ordinal (polr) designs
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OrdinalDesign:
    key: str
    why: str
    y: NDArray            # integer codes 0..K-1
    X: NDArray            # (n, p) NO intercept
    names: list[str]      # length p
    n_levels: int
    r_ordered: bool = True   # response is an ordered factor in R


def load_housing() -> OrdinalDesign:
    """MASS::housing (expanded to individual rows): Sat ~ Infl + Type + Cont.

    Treatment-contrast dummies for the three factor covariates — the canonical
    ``MASS::polr`` example, exercising a pure-factor contrast-coded design.
    """
    vals, names, _, _ = _load_raw("housing")
    col = {n: i for i, n in enumerate(names)}
    y = vals[:, col["Sat"]].astype(int)
    infl = vals[:, col["Infl"]].astype(int)
    typ = vals[:, col["Type"]].astype(int)
    cont = vals[:, col["Cont"]].astype(int)
    X = np.column_stack([_dummies(infl, 3), _dummies(typ, 4), _dummies(cont, 2)])
    xnames = ["Infl.Medium", "Infl.High",
              "Type.Apartment", "Type.Atrium", "Type.Terrace", "Cont.High"]
    return OrdinalDesign("housing", "MASS::polr canonical example (pure factor "
                         "contrast design, weighted table expanded to rows)",
                         y, X, xnames, n_levels=3)


def synth_ordinal(n: int = 1200, seed: int = 20260705) -> OrdinalDesign:
    """Deterministic well-conditioned proportional-odds DGP (continuous + factor).

    A 4-level ordered response from a proportional-odds model with two continuous
    covariates and one 3-level factor (contrast-coded) — the clean coefficient-tier
    anchor and a continuous+factor mix distinct from housing's pure-factor design.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    g = rng.integers(0, 3, n)
    beta = np.array([1.0, -0.7, 0.5, -0.9])           # x1, x2, g==1, g==2
    Xd = np.column_stack([x1, x2, (g == 1).astype(float), (g == 2).astype(float)])
    eta = Xd @ beta
    # ordered thresholds on the logit (cumulative) scale
    zeta = np.array([-1.2, 0.1, 1.4])
    u = rng.logistic(size=n)                           # latent logistic
    latent = eta + u
    y = np.searchsorted(zeta, latent)                 # 0..3
    return OrdinalDesign("synth_ordinal",
                         "well-conditioned proportional-odds DGP (2 continuous + "
                         "3-level factor), clean coefficient-tier anchor",
                         y.astype(int), Xd,
                         ["x1", "x2", "g.1", "g.2"], n_levels=4)


def sep_ordinal_complete(seed: int = 11) -> OrdinalDesign:
    """Complete separation for polr: a covariate perfectly orders the response.

    A single continuous covariate whose sign deterministically fixes the category
    drives the slope to +inf — the unpenalized MLE does not exist. R's
    ``MASS::polr`` runs ``optim`` to a huge coefficient (often with a warning /
    non-finite Hessian); pystatistics' score-guarded Newton polish rejects the
    non-PD observed information with ``ConvergenceError``.
    """
    rng = np.random.default_rng(seed)
    n = 150
    x = np.sort(rng.standard_normal(n))
    # Perfectly separable ordered response: low x -> 0, mid -> 1, high -> 2.
    y = np.zeros(n, dtype=int)
    y[x > np.quantile(x, 1 / 3)] = 1
    y[x > np.quantile(x, 2 / 3)] = 2
    X = x.reshape(-1, 1)
    return OrdinalDesign("sep_ordinal_complete",
                         "complete separation (covariate perfectly orders y); "
                         "unpenalized MLE at infinity",
                         y, X, ["x"], n_levels=3)


def collapsed_ordinal() -> OrdinalDesign:
    """An ordinal response with an interior category that has very few cases.

    Not a gap (all levels present) but a near-empty middle category — stresses the
    threshold estimation where a cut point is barely identified.
    """
    rng = np.random.default_rng(7)
    n = 400
    x = rng.standard_normal(n)
    eta = 1.1 * x
    u = rng.logistic(size=n)
    latent = eta + u
    # thresholds placed so the middle category is rare
    zeta = np.array([-0.2, 0.25])
    y = np.searchsorted(zeta, latent)
    return OrdinalDesign("collapsed_ordinal",
                         "rare interior category (barely-identified middle cut)",
                         y.astype(int), x.reshape(-1, 1), ["x"], n_levels=3)


# ----------------------------------------------------------------------------
# Multinomial (multinom) designs
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MultinomDesign:
    key: str
    why: str
    y: NDArray            # integer codes 0..K-1  (pystatistics reference = K-1)
    X: NDArray            # (n, 1+p) WITH intercept column 0
    names: list[str]      # length 1+p incl '(Intercept)'
    n_classes: int
    # R factor level order so nnet's FIRST-level baseline == pystatistics'
    # LAST-code reference: [K-1, 0, 1, ..., K-2].
    r_levels: list[int]

    @property
    def r_level_str(self) -> str:
        return ",".join(str(v) for v in self.r_levels)


def _with_intercept(Xf: NDArray) -> NDArray:
    return np.column_stack([np.ones(len(Xf)), Xf])


def _r_levels_for(K: int) -> list[int]:
    return [K - 1] + list(range(K - 1))


def load_multinom_synth() -> MultinomDesign:
    """Deterministic well-conditioned 3-class softmax (clean coefficient tier)."""
    vals, names, _, _ = _load_raw("multinom_synth")
    col = {n: i for i, n in enumerate(names)}
    y = vals[:, col["y"]].astype(int)
    Xf = np.column_stack([vals[:, col[c]] for c in ("x1", "x2", "x3")])
    K = int(y.max()) + 1
    return MultinomDesign("multinom_synth",
                          "well-conditioned balanced 3-class softmax DGP; clean "
                          "coefficient-tier anchor",
                          y, _with_intercept(Xf),
                          ["(Intercept)", "x1", "x2", "x3"], K, _r_levels_for(K))


def load_iris_multinom() -> MultinomDesign:
    """Fisher iris 3-class: real-data QUASI-SEPARATION (setosa linearly separable)."""
    vals, names, _, _ = _load_raw("iris_multinom")
    col = {n: i for i, n in enumerate(names)}
    y = vals[:, col["Species"]].astype(int)
    feats = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    Xf = np.column_stack([vals[:, col[c]] for c in feats])
    K = int(y.max()) + 1
    return MultinomDesign("iris_multinom",
                          "real 3-class; setosa linearly separable => quasi-"
                          "separation (loglik/probs match, coefficients non-"
                          "identified in both engines)",
                          y, _with_intercept(Xf),
                          ["(Intercept)"] + feats, K, _r_levels_for(K))


def load_fgl(standardize: bool = True) -> MultinomDesign:
    """MASS::fgl 6-class forensic glass: many categories + small classes.

    Standardized covariates (helps both optimizers on the badly-scaled oxides);
    still quasi-separation-prone through the small Con/Tabl classes.
    """
    vals, names, _, _ = _load_raw("fgl")
    col = {n: i for i, n in enumerate(names)}
    y = vals[:, col["type"]].astype(int)
    feats = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    Xf = np.column_stack([vals[:, col[c]] for c in feats])
    if standardize:
        Xf = (Xf - Xf.mean(0)) / Xf.std(0)
    K = int(y.max()) + 1
    return MultinomDesign("fgl",
                          "real 6-class (many categories) with small classes "
                          "(Con n=13, Tabl n=9) => quasi-separation-prone",
                          y, _with_intercept(Xf),
                          ["(Intercept)"] + feats, K, _r_levels_for(K))


def synth_multinom(n: int = 1500, K: int = 3, p: int = 3,
                   seed: int = 20260705) -> MultinomDesign:
    """Deterministic well-conditioned K-class softmax with moderate coefficients."""
    rng = np.random.default_rng(seed)
    Xf = rng.standard_normal((n, p))
    B = 0.6 * rng.standard_normal((K - 1, p))
    inter = 0.3 * rng.standard_normal(K - 1)
    eta = np.column_stack([inter[j] + Xf @ B[j] for j in range(K - 1)]
                          + [np.zeros(n)])
    P = np.exp(eta - eta.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    u = rng.uniform(size=n)
    y = (u[:, None] > np.cumsum(P, 1)).sum(1).astype(int)
    return MultinomDesign(f"synth_multinom_K{K}_p{p}",
                          f"well-conditioned {K}-class softmax (n={n}, p={p})",
                          y, _with_intercept(Xf),
                          ["(Intercept)"] + [f"x{i+1}" for i in range(p)],
                          K, _r_levels_for(K))


def sep_multinom_complete(seed: int = 3) -> MultinomDesign:
    """Complete separation for multinom: one class perfectly predicted by a covariate.

    Both nnet and multinom report 'convergence' but the separating coefficient runs
    off to infinity (bounded only by the optimizer). The validation records that the
    log-likelihood agrees while the coefficients are non-identified — matching R's
    own behaviour (nnet does not fail loud on separation).
    """
    rng = np.random.default_rng(seed)
    n = 180
    x = rng.standard_normal(n)
    # class 2 iff x large; classes 0/1 split among the rest by a second covariate.
    z = rng.standard_normal(n)
    y = np.where(x > 0.8, 2, np.where(z > 0, 1, 0))
    Xf = np.column_stack([x, z])
    K = 3
    return MultinomDesign("sep_multinom_complete",
                          "complete separation (class 2 iff x>0.8); MLE at infinity",
                          y, _with_intercept(Xf),
                          ["(Intercept)", "x", "z"], K, _r_levels_for(K))


def unbalanced_multinom(seed: int = 5) -> MultinomDesign:
    """Severely unbalanced classes (a rare category with a handful of cases)."""
    rng = np.random.default_rng(seed)
    n = 900
    Xf = rng.standard_normal((n, 2))
    # heavily skewed marginal: class 2 very rare
    eta = np.column_stack([0.5 + 0.8 * Xf[:, 0], -3.0 + 0.5 * Xf[:, 1],
                           np.zeros(n)])
    P = np.exp(eta - eta.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    u = rng.uniform(size=n)
    y = (u[:, None] > np.cumsum(P, 1)).sum(1).astype(int)
    K = 3
    return MultinomDesign("unbalanced_multinom",
                          "severely unbalanced classes (rare middle category)",
                          y, _with_intercept(Xf),
                          ["(Intercept)", "x1", "x2"], K, _r_levels_for(K))
