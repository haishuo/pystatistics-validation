"""Simulated MVN MLE benchmark problems that mirror the retired survey inputs.

One job: generate, deterministically from a seed, NaN-coded data matrices whose
*computational* profile matches the registration-gated survey problems the
mvnmle benchmarks previously used (WVS wave 7 and the GSS cumulative file, both
excluded from published evidence by JSS's open-data policy). The estimator's
cost is driven by (n, p), the number and shape of distinct missingness
patterns, and the conditioning of the covariance — so those are what the two
profiles reproduce; the values themselves are synthetic and fully
redistributable.

Two profiles, mirroring the two missingness regimes we benchmarked:

  * ``simw`` — item-nonresponse dominated (WVS-like): n = 97,220. Missingness
    comes from per-(group, item) "not fielded" blocks (country-wave omissions)
    plus row-correlated item nonresponse; pattern counts grow steeply with p.
  * ``simg`` — planned-missingness dominated (GSS-like): n = 75,699. Rows fall
    into year-blocks and ballots; an item exists only from its first-fielded
    year and (below a coverage threshold) skips one of three ballots. Complete
    cases vanish for large p and patterns are dominated by (year x ballot)
    cells.

Per-item observed rates are anchored to the retired problems: the mean observed
rate of the first p columns matches, by construction, the measured mean of the
real curated problem at each benchmarked p (anchors measured 2026-08-26 from
the last survey builds; see ``_COVERAGE_SEGMENTS``). Items are nested — item i
is identical in the p=50 and p=100 problems — mirroring how the curation
pipeline's column selection largely nests as p grows.

The covariance is a factor model (a few global factors + a per-module factor +
idiosyncratic noise) with one near-duplicate pair per started block of 25 items
(latent r up to 0.985, just under the curation screen's 0.99 cap) so the
correlation spectrum's conditioning lands in the O(10^2)-O(10^3) range measured
on the real problems. The latent Gaussian draws are then **discretized into
skewed Likert-type ordinal items** (2-10 categories, Dirichlet-random category
masses, duplicate pairs share thresholds) — the retired problems were ordinal
survey items, and fitting a Gaussian MLE to discretized/skewed data is what
makes the optimizer do realistic work. On exactly-Gaussian draws the optimum
sits next to the initialization and every fit converges in a handful of
iterations, which understates the end-to-end cost the benchmarks exist to
measure (observed: 3-7 iterations vs the retired problems' 19-164).

Everything is deterministic given ``seed`` (PCG64 streams keyed on
``(seed, p)``; default seeds are fixed per profile so two machines generate
byte-identical problems), and the returned object quacks like
:class:`survey_io.MVNProblem` so drivers treat simulated and store-backed
problems uniformly (see ``problem_source``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Default seeds, fixed so every machine regenerates identical problems.
PROFILE_SEEDS = {"simw": 20260826, "simg": 20260827}

#: Row counts of the retired survey problems these profiles mirror.
PROFILE_N = {"simw": 97_220, "simg": 75_699}

#: Piecewise-constant per-item observed-rate targets, as ``(end_index, mean)``
#: segments over the absolute item index. Derived by inverting the measured
#: mean observed rates of the retired curated problems at each benchmarked p
#: (prefix means; WVS anchors at p = 5/15/50/100, GSS at p = 15/50/100), so the
#: mean coverage of the first p simulated items reproduces the real problem's
#: at every anchor. Measured 2026-08-26.
_COVERAGE_SEGMENTS = {
    "simw": ((5, 0.914), (15, 0.9215), (50, 0.8904), (100, 0.8706)),
    "simg": ((15, 0.839), (50, 0.726), (100, 0.693)),
}

#: Fraction of an item's missingness deficit realized structurally (group
#: blocks for simw; design-cell rotation for simg) rather than by item
#: nonresponse. Tuned so pattern counts land near the retired problems'. For
#: simg the share is depth-graded (see ``_simg_structural_share``): core items
#: are nearly purely structural (GSS p=15 shows only 89 patterns), deeper items
#: carry a few percent of row-level nonresponse (p=50 shows 3,778).
_STRUCTURAL_SHARE = {"simw": 0.45}


def _simg_structural_share(index: NDArray[np.intp]) -> NDArray[np.float64]:
    # Core items are almost purely structural; row-level nonresponse ramps up
    # sharply past index 50 (rare/sensitive items), which is what multiplies
    # the pattern count ~9x between p=50 and p=100 in the real file.
    idx = index.astype(np.float64)
    return 0.99 - 0.07 * np.minimum(1.0, idx / 50.0) - 0.25 * np.maximum(0.0, (idx - 50.0) / 50.0)


@dataclass(frozen=True)
class SimProblem:
    """A simulated MVN benchmark problem (interface-compatible with MVNProblem)."""

    dataset_name: str
    source: str
    X: NDArray[np.float32]              # (n_kept, p) with NaN for missing
    column_indices: NDArray[np.intp]
    column_names: list[str]
    n_rows_original: int
    overall_missing_frac: float
    seed: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


def _item_coverage(profile: str, p: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Target observed rate per item, from the anchored segments plus jitter."""
    segments = _COVERAGE_SEGMENTS[profile]
    if p > segments[-1][0]:
        raise ValueError(
            f"{profile}: p={p} exceeds the profiled range (max {segments[-1][0]}); "
            f"the coverage anchors only cover the benchmarked grid")
    cov = np.empty(p)
    start = 0
    for end, mean in segments:
        if start >= p:
            break
        stop = min(end, p)
        cov[start:stop] = mean
        start = stop
    cov += rng.uniform(-0.02, 0.02, size=p)
    return np.clip(cov, 0.35, 0.97)


def _factor_correlation(p: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """A unit-diagonal covariance with survey-like correlation spread."""
    k = max(2, round(p / 12))
    load = rng.normal(0.0, 0.32, size=(p, k))
    # Each item loads mainly on one global factor — survey items cluster by topic.
    main = rng.integers(0, k, size=p)
    load[np.arange(p), main] += rng.uniform(0.35, 0.6, size=p) * rng.choice([-1.0, 1.0], size=p)

    n_modules = max(1, p // 5)
    module = np.arange(p) % n_modules
    mod_load = np.zeros((p, n_modules))
    mod_load[np.arange(p), module] = rng.uniform(0.25, 0.45, size=p)

    sigma = load @ load.T + mod_load @ mod_load.T + np.diag(rng.uniform(0.4, 1.0, size=p))
    d = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(d, d)

    # Near-duplicate pairs: overwrite row j as an r-correlated copy of row i,
    # matching the just-under-the-screen pairs the curated problems retain
    # (the screen caps |r| at 0.99; the real problems' worst pairs sit at
    # ~0.94-0.99 and push the pairwise-correlation conditioning into the
    # O(10^3) range, which the fp32 precision claims are calibrated against).
    for m in range(max(1, p // 25)):
        r = 0.99 if m == 0 else 0.95
        i, j = 2 * m, 2 * m + 1  # adjacent low indices; deterministic
        corr[j, :] = r * corr[i, :]
        corr[:, j] = corr[j, :]
        corr[j, j] = 1.0
        corr[i, j] = corr[j, i] = r
    # Symmetrize and lift tiny eigenvalues so Cholesky is well-defined.
    corr = (corr + corr.T) / 2.0
    w, v = np.linalg.eigh(corr)
    w = np.clip(w, 1e-4, None)
    corr = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def _draw_values(n: int, corr: NDArray[np.float64],
                 rng: np.random.Generator) -> NDArray[np.float32]:
    p = corr.shape[0]
    chol = np.linalg.cholesky(corr)
    x = np.empty((n, p), dtype=np.float32)
    # Chunked so p=100 at n~100k stays comfortably in memory.
    step = 16_384
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        z = rng.standard_normal((hi - lo, p))
        x[lo:hi] = (z @ chol.T).astype(np.float32)
    return x


def _item_scales(p: int, rng: np.random.Generator) -> list[NDArray[np.float64]]:
    """Latent thresholds per item: a Likert-type scale with random skew.

    Category count is weighted toward the 4-7 point scales surveys actually
    use; category masses are Dirichlet-random (so some items are symmetric,
    some heavily skewed) with a 2% floor so no category is empty.
    """
    from statistics import NormalDist
    inv = NormalDist().inv_cdf
    cats = rng.choice([2, 3, 4, 5, 6, 7, 10],
                      p=[0.05, 0.10, 0.20, 0.30, 0.15, 0.15, 0.05], size=p)
    cats[0] = cats[1] = 10   # the tightest near-duplicate pair sits on a fine
    #                          scale, so discretization attenuates it least and
    #                          the observed |r| stays just under the 0.99 screen
    scales = []
    for k in cats:
        mass = rng.dirichlet(np.full(k, 1.5))
        mass = np.maximum(mass, 0.02)
        mass /= mass.sum()
        cum = np.cumsum(mass)[:-1]
        scales.append(np.array([inv(c) for c in cum]))
    return scales


def _discretize(x: NDArray[np.float32], scales: list[NDArray[np.float64]],
                dupe_pairs: int) -> NDArray[np.float32]:
    """Cut each latent column into integer codes 1..k (in place, returned).

    Each near-duplicate pair shares one scale: attenuating the two members
    through different thresholds would break the just-under-the-screen
    correlation the pair exists to provide.
    """
    for m in range(dupe_pairs):
        scales[2 * m + 1] = scales[2 * m]
    for j, thr in enumerate(scales):
        x[:, j] = np.searchsorted(thr, x[:, j]).astype(np.float32) + 1.0
    # Derived scale scores: one item per started block of 25 is the rounded
    # mean of its three predecessors — surveys carry such composite indices,
    # and they are the multi-way near-dependencies (invisible to the pairwise
    # |r| <= 0.99 screen) that push the real problems' correlation
    # conditioning into the O(10^3) range. Rounding keeps the dependence
    # near-exact rather than exact, so the MLE stays well-defined.
    p = x.shape[1]
    for j in range(12, p, 25):
        x[:, j] = np.round(x[:, j - 3:j].mean(axis=1))
    return x


def _nonresponse_mask(n: int, rates: NDArray[np.float64],
                      rng: np.random.Generator, *,
                      clean_frac: float) -> NDArray[np.bool_]:
    """Item nonresponse: a clean-respondent share plus row-correlated skipping.

    ``rates`` are per-item *unconditional* target rates; active rows carry the
    whole load, scaled by a lognormal per-row propensity (mean 1), which is what
    concentrates misses in a subset of rows the way real respondents do.
    """
    p = rates.size
    active = rng.random(n) >= clean_frac
    row_mult = np.where(active, np.exp(rng.normal(0.0, 0.4, size=n)) / (1.0 - clean_frac), 0.0)
    prob = np.clip(row_mult[:, None] * rates[None, :], 0.0, 0.75)
    return rng.random((n, p)) < prob


def _mask_simw(n: int, p: int, coverage: NDArray[np.float64],
               rng: np.random.Generator) -> NDArray[np.bool_]:
    """WVS-like: (group x item) not-fielded blocks + row-correlated nonresponse."""
    n_groups = 49
    group = rng.integers(0, n_groups, size=n)
    deficit = 1.0 - coverage

    # Structural share: each item is not fielded in a seeded subset of groups
    # whose total row mass ~ the structural deficit.
    struct = _STRUCTURAL_SHARE["simw"] * deficit
    n_blocked = np.round(struct * n_groups).astype(int)          # groups to block, per item
    miss = np.zeros((n, p), dtype=bool)
    for j in range(p):
        if n_blocked[j] > 0:
            blocked = rng.choice(n_groups, size=n_blocked[j], replace=False)
            miss[np.isin(group, blocked), j] = True

    realized = miss.mean(axis=0)
    residual = np.clip(deficit - realized, 0.0, None)
    miss |= _nonresponse_mask(n, residual, rng, clean_frac=0.7)
    return miss, group.astype(np.float64) / (n_groups - 1)


#: simg only: weight on the early-year bias when ranking (year x ballot) cells
#: for dropping. 0 = items drop independent random cells (no shared structure),
#: 1 = every item drops the same early years (rows there become all-missing).
_SIMG_YEAR_BIAS = 0.5


def _mask_simg(n: int, p: int, coverage: NDArray[np.float64],
               rng: np.random.Generator) -> NDArray[np.bool_]:
    """GSS-like: per-item (year x ballot) rotation + light item nonresponse.

    Each item is fielded on a subset of the (year, ballot) design cells; the
    dropped cells are chosen per item by ranking cells on a mix of early-year
    bias (items enter the survey late) and item-specific noise (rotation), and
    dropping from the front until the item's structural deficit is covered.
    Shared early-cell drops give correlated missingness (bounded pattern
    counts); the noise keeps the drops from coinciding exactly (no all-missing
    rows, vanishing complete cases at large p).
    """
    n_years, n_ballots = 30, 3
    # Uneven year sizes (later waves bigger), fixed by the seed.
    sizes = rng.dirichlet(np.linspace(1.0, 3.0, n_years)) * n
    year = np.repeat(np.arange(n_years), np.maximum(1, sizes.astype(int)))[:n]
    if year.size < n:  # rounding shortfall
        year = np.concatenate([year, np.full(n - year.size, n_years - 1)])
    ballot = rng.integers(0, n_ballots, size=n)

    cell = year * n_ballots + ballot                     # (n,) design-cell id
    n_cells = n_years * n_ballots
    cell_mass = np.bincount(cell, minlength=n_cells) / n
    cell_year = np.arange(n_cells) // n_ballots / (n_years - 1)

    deficit = 1.0 - coverage
    struct = _simg_structural_share(np.arange(p, dtype=np.intp)) * deficit
    cell_ballot = np.arange(n_cells) % n_ballots
    miss = np.zeros((n, p), dtype=bool)
    for j in range(p):
        score = _SIMG_YEAR_BIAS * cell_year + (1.0 - _SIMG_YEAR_BIAS) * rng.random(n_cells)
        if struct[j] >= 0.2:
            # Deep item: rotated off one ballot in the years it is fielded, so
            # its structural drops reach late-year rows too. Without this, rows
            # in never-dropped late cells stay complete — the real cumulative
            # file has none at p >= 50.
            rot = rng.integers(0, n_ballots)
            score -= 0.35 * (cell_ballot == rot)
        order = np.argsort(score)                         # early-biased, item-jittered
        cum = np.cumsum(cell_mass[order])
        n_drop = int(np.searchsorted(cum, struct[j])) + 1 if struct[j] > 0 else 0
        if n_drop:
            miss[np.isin(cell, order[:n_drop]), j] = True

    realized = miss.mean(axis=0)
    residual = np.clip(deficit - realized, 0.0, None)
    miss |= _nonresponse_mask(n, residual, rng, clean_frac=0.75)
    return miss, year.astype(np.float64) / (n_years - 1)


#: Scale of the group-level latent mean shifts (country effects for simw,
#: era drift for simg). The retired problems pool heterogeneous populations —
#: WVS pools ~49 countries, the GSS cumulative file pools five decades — so
#: pattern-wise moments genuinely conflict and the joint MLE has to reconcile
#: them, which is a large part of why those fits took 19-164 iterations.
#: Without drift the simulated optimum sits near the moment initialization.
_DRIFT_SCALE = {"simw": 0.35, "simg": 0.55}


def _group_drift(profile: str, base_seed: int, design: NDArray[np.float64],
                 p: int) -> NDArray[np.float32]:
    """Per-item latent mean shift as a function of the row's design position.

    The shifts share one group-level factor across items — countries differ
    along common cultural dimensions and eras drift along common trends, they
    do not scatter independently per item — plus a per-item idiosyncratic
    part. For simw the shared factor is an arbitrary per-country vector; for
    simg it is the era axis itself (a common linear trend), with a per-item
    trend of its own and a small per-year wobble. Per-item streams are keyed
    without p so items stay nested across problem sizes, and each
    near-duplicate pair shares one drift column (variant wordings of one
    question move together; independent drift would decorrelate the pair the
    conditioning story depends on).
    """
    scale = _DRIFT_SCALE[profile]
    levels = np.unique(design)
    n_levels = len(levels)
    level_of = np.searchsorted(levels, design)
    shared_rng = np.random.Generator(np.random.PCG64([base_seed, 5]))
    if profile == "simw":
        shared = shared_rng.normal(0.0, 1.0, size=n_levels)
    else:
        shared = (levels - 0.5) * 2.0          # the era axis, range ~[-1, 1]
    cols = []
    for i in range(p):
        g = np.random.Generator(np.random.PCG64([base_seed, 4, i]))
        w = g.uniform(0.3, 0.9) * g.choice([-1.0, 1.0])
        if profile == "simw":
            idio = g.normal(0.0, 1.0, size=n_levels)
        else:
            idio = (g.normal(0.0, 0.6) * (levels - 0.5) * 2.0
                    + g.normal(0.0, 0.15, size=n_levels))
        cols.append(scale * (w * shared + np.sqrt(1.0 - w * w) * idio)[level_of])
    for m in range(max(1, p // 25)):
        cols[2 * m + 1] = cols[2 * m]
    return np.column_stack(cols).astype(np.float32)


_MASKS = {"simw": _mask_simw, "simg": _mask_simg}


def make_sim_problem(profile: str, p: int, *, seed: int | None = None) -> SimProblem:
    """Generate the ``(profile, p)`` benchmark problem.

    Deterministic: streams are keyed on ``(seed or PROFILE_SEEDS[profile], p)``
    so each (profile, p) cell is reproducible in isolation. Item properties
    (coverage targets, covariance) use dedicated substreams seeded without p,
    so items are nested across p.
    """
    if profile not in _MASKS:
        raise ValueError(f"unknown sim profile {profile!r}; expected one of {sorted(_MASKS)}")
    if p < 2:
        raise ValueError(f"p must be >= 2, got {p}")
    base_seed = PROFILE_SEEDS[profile] if seed is None else seed
    item_rng = np.random.Generator(np.random.PCG64([base_seed, 1]))   # p-independent
    mask_rng = np.random.Generator(np.random.PCG64([base_seed, 2, p]))
    value_rng = np.random.Generator(np.random.PCG64([base_seed, 3, p]))
    n = PROFILE_N[profile]

    coverage = _item_coverage(profile, p, item_rng)
    corr = _factor_correlation(p, item_rng)
    scales = _item_scales(p, item_rng)
    x = _draw_values(n, corr, value_rng)
    miss, design = _MASKS[profile](n, p, coverage, mask_rng)
    x += _group_drift(profile, base_seed, design, p)
    x = _discretize(x, scales, max(1, p // 25))

    # No column may end up effectively unobserved; that would be a degenerate
    # problem no curation pipeline would emit. Fail loud rather than fit garbage.
    obs_rate = 1.0 - miss.mean(axis=0)
    if obs_rate.min() < 0.05:
        raise RuntimeError(
            f"{profile} p={p}: column observed rate fell to {obs_rate.min():.3f}; "
            f"the design constants no longer match the documented profile")

    x[miss] = np.nan
    keep = ~np.all(miss, axis=1)          # drop all-missing rows, like curation does
    x = x[keep]

    return SimProblem(
        dataset_name=profile,
        source=f"synthetic ({profile} profile, seed={base_seed}, mvn_sim v1)",
        X=x,
        column_indices=np.arange(p, dtype=np.intp),
        column_names=[f"{profile}_v{i:03d}" for i in range(p)],
        n_rows_original=n,
        overall_missing_frac=float(np.isnan(x).mean()),
        seed=base_seed,
    )
