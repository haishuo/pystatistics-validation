# mixed F3 fix — handback (intended version 4.5.1)

Targeted robustness fix for **finding F3** (extreme variance-ratio non-convergence
regression in the 4.5.0 structured LMM solver). Implemented in `../pystatistics`.
**Not published** — the validation session publishes 4.5.1 and re-validates/blesses.

## 1. The bug, reproduced and root-caused

Reproduced `hc3_extreme_variance_ratio()` (G=15, ni=6, between sd 10, residual sd
0.03, ICC≈0.99999) through `lmm(y, X, groups={'g': g})`:

- 4.5.0: `converged=False`, `n_iter=15`, L-BFGS-B `Message: ABNORMAL:`,
  between-group variance **107.5** (R/lme4 130.3, ~17% low), logLik 105.302
  (R 105.471) — confirmed.

Root cause (deeper than "flat deviance"): at ICC→1 the optimal relative RE scale
θ is very large — here θ*≈481, and out in the sd=0.01 tail θ*≈800–1300. L-BFGS-B's
gradient is a **forward finite difference with an absolute step** (~1.5e-8); at
θ≈1e3 that is a ~1e-11 *relative* step, swamped by round-off, so the gradient is
garbage. That produces **two** failure modes, not one:

- **Loud** (sd=0.03, the F3 case): the line search can't make progress on the
  noisy gradient → `ABNORMAL` → `converged=False`.
- **Silent** (found while gating sd=0.01): L-BFGS-B thinks the gradient is ~0 and
  reports **success at a non-stationary point** — e.g. θ=599 (dev −311.2) when the
  true optimum is θ=1311 (dev −330.8), a between-group variance 61 vs 183. This is
  worse than F3 (silently wrong) and the gate's sd=0.01 column would have caught
  it, so the fix had to address both.

lme4's derivative-free bobyqa (trust-region) is immune to the ill-scaled gradient
and converges in this regime.

## 2. The fix

New module `pystatistics/mixed/_optimizer.py` (θ-optimization extracted from
`solvers.py`, which was already at the 400-LoC soft limit; Rule 3/4). Public
entry `optimize_theta(ctx, starts, bounds, lb, max_iter, tol)`:

1. Run L-BFGS-B from each candidate start (unchanged primary path), keep the best.
2. **Fast path** — if L-BFGS-B succeeded *and* a cheap scale-aware stationarity
   probe (`_is_stationary`, 2·dim deviance evals, relative+absolute step
   `1e-3·|θ_i| + 1e-4`) finds no lower neighbour → return that result unchanged.
3. Otherwise (loud non-convergence **or** the probe flags a silent premature
   stop) → restart a **bounded Nelder-Mead** simplex (bobyqa-analogue,
   derivative-free) from the best θ, and **adopt it only if it strictly lowers
   the deviance** by more than round-off (`_FALLBACK_MIN_IMPROVEMENT = 1e-6`).

Adopt-only-if-strictly-lower means the fallback can only move a fit *toward* the
global optimum, never regress one L-BFGS-B already solved. On every well-behaved
design the probe passes and NM never runs, so those fits are **byte-for-behaviour
identical** to 4.5.0 (verified: NM fired 0× on sleepstudy-like, random-intercept,
and intercept-only G=2000). No new dependency (scipy `minimize`/`Bounds`). GLMM
path (`glmm()`) untouched.

## 3. Gate results (local, `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$PWD`)

**Gate 1 — hc3 exact case now converges and matches R:**
- `converged=True`, `n_iter=34`, `is_singular=False`, no warnings.
- between-group variance **130.30500** vs true global / R 130.305 → rel **1.7e-6**
  (was 0.17). logLik **105.470943** vs R 105.47094 → rel **6.1e-13**.

**Gate 2 — residual-sd × seed sweep (sd ∈ {0.01,0.03,0.1,0.3,1.0} × 5 seeds, and
re-checked with 8 seeds each = 40 fits):**
- **0 non-convergences.**
- Every fit matches the true global optimum of the profiled deviance (exhaustive
  1-D scalar search, the optimizer-independent reference lme4 also targets): max
  between-group-variance rel-err **6.8e-4** (was up to 0.67 from silent premature
  stops). All within the optimizer tier.

**Gate 3 — no regression elsewhere:**
- Full mixed/LMM suite green; **whole `tests/` suite: 3034 passed, 110 skipped, 0
  failures**.
- Textbook CPU-vs-R correctness (sleepstudy/penicillin/dyestuff/dyestuff2/pastes,
  REML+ML) and F1/F2a/F2b behaviour intact — NM never fires on them, results
  byte-identical. dyestuff2 + HC1 still flag `is_singular`.
- Scaling unchanged: intercept-only G=2000 still ~20 ms (only added cost on
  well-behaved fits is the 2·dim stationarity probe), near-linear, beating lmer.

**Gate 4 — GLMM suite still green** (shared code untouched).

## 4. Files changed

- `pystatistics/mixed/_optimizer.py` — **new** (46 LoC), the robust θ optimizer.
- `pystatistics/mixed/solvers.py` — `lmm()` optimization block now calls
  `optimize_theta`; multi-start list built for both branches; θ-optimization code
  moved out. Now 388 code LoC (was 406, over soft limit before this change).
- `tests/mixed/test_lmm_extreme_ratio.py` — **new** regression tests: the exact
  hc3 case must converge and match lme4's variance component, plus the full
  sd×seed sweep must converge and hit the global optimum (26 tests).
- `.release/UNRELEASED.md` — one `## Changes` bullet (see below).

## 5. Staged `.release/UNRELEASED.md` content (verbatim, intended version 4.5.1)

> - Fixed `lmm()` failing to converge (and, in a related case, converging to a
>   silently wrong optimum) in the extreme variance-ratio regime — when the
>   intraclass correlation approaches 1 (residual variance orders of magnitude
>   below the between-group variance). The optimal relative random-effects scale θ
>   is then very large (O(1e2)–O(1e3)) and the profiled REML/ML deviance is flat
>   and ill-scaled, where the gradient-based L-BFGS-B optimizer either terminated
>   its line search abnormally (returning `converged=False`) or stopped and
>   reported success at a non-stationary point (because its absolute-step
>   finite-difference gradient is a negligible *relative* step at large θ). On a
>   near-perfect-ICC random-intercept fit (G=15, residual sd 0.03) this returned a
>   between-group variance of 107.9 vs lme4's 130.3 (~17% low). The θ optimizer
>   (new module `pystatistics/mixed/_optimizer.py`) now runs a bounded
>   derivative-free Nelder-Mead fallback — engaged only when L-BFGS-B did not
>   converge or a cheap scale-aware stationarity probe flags a premature stop, and
>   adopted only when it strictly lowers the deviance — so it converges to the true
>   global optimum across the extreme-ratio tail (now matches lme4 to ~1e-6
>   relative on the variance component and ~1e-13 on the log-likelihood) while
>   leaving all well-converged fits byte-for-behaviour identical.

**Intended version: 4.5.1.** Public changelog draft — adjust per README/CHANGELOG
discipline (Rule 9). Did NOT run `release.py`, did NOT bump versions, did NOT
publish — those are yours.

## 6. Notes for re-validation

- The gate reference above uses an **exhaustive 1-D scalar search** as the "true
  global optimum" proxy (θ is scalar for a random intercept). When you re-run
  against live R, confirm lme4/bobyqa lands on the same optimum at sd=0.01 — the
  fix drives pystatistics to the *global* min, which is the MLE; if R's bobyqa
  ever stops early there, pystatistics would be the more-correct engine, not a
  mismatch to explain away.
- Only `lmm()`'s optimizer changed. `grm_lmm()`, GLMM, SEs (F2a), the structured
  solve (F2b), and singular detection (F1) are all untouched.
