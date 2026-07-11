# A2 — gam GLM-family analytic smoothing-parameter gradient: implementation + validation evidence

**Task (close-out brief / finding H4):** the Gaussian-identity analytic sp-gradient was done at
4.6.1; Poisson/binomial (UBRE), free-dispersion GCV and non-Gaussian REML still used a
finite-difference `(2m+1)`-inner-fit gradient. Implement the Wood (2011) implicit-derivative
analytic gradient for the GLM-family criteria. Guarantee-3 performance close, MUST-FINISH.
Ships in the 4.7.0 bundle (implemented in the working tree; formal artifact freeze + report
refresh happens at the bundled 4.7.0 re-validation per R5).

## What was built (library repo, uncommitted 4.7.0 working tree)

- **New `pystatistics/gam/_gradient_glm.py`** — the full Wood-2011 implicit-derivative gradient:
  `dβ/dρ_j = −λ_j (X'W̃X + S_λ)^{-1} S_j β` through the P-IRLS fixed point with **full NEWTON
  weights** `W̃ = diag(−du_i/dη_i)`, plus the weight-dependence terms in `d edf/dρ` and
  `d log|A|/dρ` via `ω = dw_Fisher/dη`. `u`, `ω`, `W̃` come from deterministic central
  differences (step `1e-5·max(|η|,1)`) of the family's own `linkinv/mu_eta/variance` — works for
  every family/link pair without growing the Family/Link contract (approved design decision;
  numpy only, NO torch per `a3-torch-loses-on-cpu`).
- **`_criteria.select_lambdas`**: every supported family/method now drives L-BFGS-B with
  `jac=True` (Gaussian-identity closed form or the new GLM form); the finite-difference
  objective path is removed. Warm-started inner fits retained; the winning P-IRLS branch's
  converged mean is returned so `gam()`'s final fit continues it (see review finding 1 below).

## Adversarial review (21-agent workflow: 5 lenses → per-finding refuters)

Review of the A2 diff ran 5 independent lenses (math-vs-Wood-2011, numerics, wiring,
constitution, test-coverage) with adversarial verification of every finding. 13 findings
refuted or fixed-in-flight (incl. an end-to-end mgcv-pinned `gam()` regression test, cloglog/
sqrt links in the gradient matrix, tolerance-floor robustification, exceptions-contract tests).
**Two majors CONFIRMED with reproductions, both fixed and regression-tested:**

1. **Multimodal inner fits + warm-start hysteresis (was silent-wrong).** gaussian-log n=60
   torture case: the warm-chained search converges on the DEEP P-IRLS branch (GCV 4.43 —
   matching `mgcv` (GCV 4.84) on identical data), but `gam()`'s final FRESH refit landed on a
   shallow branch (GCV 38.2), silently reported with `converged=True`. mgcv never refits from
   scratch — its reported fit is the search's own continuation. Fix: `select_lambdas` resolves
   the branch at the accepted λ (warm-continued vs fresh, better criterion wins) and returns
   the winner's converged mean; `gam()`'s final fit warm-starts from it. Post-fix the case
   reports GCV 4.43, converged. Healthy panel unchanged (differences ≤1e-9).
2. **Reachable crash: singular Newton system.** binomial-probit n=60: mgcv's own optimum sits
   where the Newton eigenvalues collapse (~4e-8); the fail-loud `ConvergenceError` propagated
   uncaught out of public `gam()` on a fit mgcv completes (and the removed FD path degraded
   gracefully). Fix: warned Fisher-weight fallback for that evaluation (always defined, exact
   for canonical links, only steers the optimizer). Post-fix the case completes. Plus: a
   diverging inner fit at a TRIAL λ during the search is now a soft `+inf` barrier (line
   search backtracks, mgcv-newton-style) instead of an aborted selection; divergence at the
   starting values or the final fit still fails loud.

**Torture-panel closure evidence (40 configs: n=60, k=12+10, gaussian-log signal-5 /
binomial-probit signal-2.5, seeds 14–33; reported criterion vs a fresh-evaluation FD reference
search, deficit threshold 5%):** post-fix analytic path — 36 ok, 1 honestly flagged
(`outer_converged=False`/warned), 3 local-basin deficits, **0 crashes**. The reconstructed
4.6.x FD baseline on the SAME panel: 3 silent deficits (its seed-16 at GCV 37.8 vs our 4.43)
and the same local-minimum behavior. Verdict: the remaining 3 are the pre-existing
optimizer-tier local-optimum behavior of any quasi-Newton search on a genuinely multimodal
criterion (mgcv's Newton basin-hops on such surfaces too — Wood documents GCV multimodality);
A2 is at baseline parity on them, strictly better on the worst case, and crash-free.

## Fixes gathered while building A2 (same unreleased bundle)

- **Fix (found by the new A2 test coverage, VA-3 scope, same unreleased bundle):**
  `gam(family='nb', method='GCV')` with *estimated* theta silently returned a degenerate
  dispersion — profiling the UBRE score over theta is structurally degenerate (NB deviance
  shrinks monotonically as θ→0 with no counterweight in UBRE; on θ=3 data the profile ran
  UBRE −0.02 @θ=3 → −0.90 @θ=0.04, and the fit returned θ̂=0.039). mgcv's GCV-era `nb()`
  uses a different, unimplemented theta estimator (its GCV θ̂=1.27 on the same data — also
  far from 3; the REML route is the sound one, θ̂=3.297). Now REFUSED loudly with a message
  directing to `method='REML'` or a fixed `NegativeBinomial(theta=…)`; VA-3's UNRELEASED
  entry amended.
- **Fix (gathered in passing):** `ConvergenceError` required a positional `iterations`, but the
  4.6.0 gam P-IRLS fail-loud sites construct it message-only — a genuinely diverging GAM raised
  `TypeError` instead of the documented `ConvergenceError`. `iterations` is now optional; the 27
  call sites that pass it are unchanged.
- **Tests:** `tests/gam/test_sp_gradient.py` (38 tests) — analytic-vs-criterion-FD agreement
  (~1e-7 rel; asserted <1e-4) across poisson(log, sqrt) / binomial(logit, probit, cloglog) /
  Gamma(log, inverse) / nb(θ=3) / gaussian-log × UBRE/GCV/REML; λ extremes (ρ₀±8) and
  anisotropic ρ; rank-deficient concurvity; near-separation binomial (μ clamps active);
  bit-determinism; singular-Newton warned Fisher-fallback; multimodal-branch regression;
  probit crash-region completion; end-to-end `gam()` pins on mgcv-selected EDF/sp
  (poisson/binomial × REML/GCV); selection lands on the FD-search optimum; inner-fit count no
  longer scales as `2m+1`; nb+GCV refusal. Plus exceptions-contract tests in
  `tests/core/test_exceptions.py`; probit-REML e2e selection pin + fixed-sp REML score pin
  vs mgcv + non-PD-Hessian fallback (Newton-determinant fix). Full library suite: 3647
  passed.

## Why Newton weights (silent-wrong risk killed)

A Fisher-weight shortcut in the implicit solve (reusing the fit's `A⁻¹`) is exact ONLY for
canonical links. Measured against criterion-FD: probit UBRE off up to **8.9e-2** relative,
Gamma-log GCV up to 4e-2 — enough to silently shift selected smoothness. The Newton form agrees
to ~1e-8–7e-7 everywhere (canonical links: Fisher = Newton, both ~1e-9).

## Validation evidence (working tree vs mgcv 1.9.3, identical CSV data both engines)

**Free selection, n=500, 2 smooths (cr k=10 + cr k=8), one dataset per family:**

| case | EDF gap (py − mgcv) | sp rel diff | fitted rel max | note |
|---|---|---|---|---|
| poisson REML | +1.05e-4 | 1.1e-4 | 1.3e-6 | |
| poisson UBRE | +3.4e-5 | 2.2e-5 | 4.4e-7 | |
| binomial REML | +1e-6 | 2.1e-7 | 8.8e-9 | |
| binomial UBRE | +1e-6 | 1.0e-6 | 3.9e-8 | |
| binomial-probit UBRE | +1.7e-5 | 1.0e-5 | 7.7e-7 | |
| binomial-probit REML | **+7.6e-3** | 4.4e-3 | 2.3e-4 | criterion nuance, see below |
| Gamma-log GCV | −2.1e-4 | 2.2e-4 | 7.5e-6 | |
| gaussian-log GCV | +2.2e-5 | 6.3e-5 | 2.2e-7 | |
| nb θ=3 REML | +1.4e-3 | 3.0e-3 | 1.1e-4 | criterion nuance, see below |
| nb θ estimated REML | +1.3e-3 | 2.8e-3 | 1.0e-4 | θ̂ 3.2971 vs mgcv 3.29700 |

Plus a 30-dataset canonical-link sweep (poisson/binomial × REML, n∈{120..600}, incl. sparse
low-count and single-trial flat surfaces): worst EDF gap 1.5e-4, most ≤ 2e-5 — tighter than the
finite-difference path it replaces (whose worst was ~2e-4).

**Perf (H4 close), n=2000, poisson, m=1..6 cr smooths, best-of-reps, same machine (final
numbers incl. the two branch-resolution fits added by the review hardening):**

| m | REML speedup vs FD path | UBRE speedup | inner fits FD → analytic (REML) |
|---|---|---|---|
| 1 | 2.0× | 1.1× | 25 → 10 |
| 2 | 3.2× | 1.5× | 46 → 11 |
| 3 | 2.6× | 2.0× | 37 → 11 |
| 4 | 4.9× | 2.4× | 91 → 14 |
| 5 | 9.0× | 2.7× | 163 → 14 |
| 6 | 4.2× | 3.3× | 71 → 13 |

Selected λ agrees with the FD search to ≤ 8.5e-5 relative in every cell — same optimum, cheaper
route. vs mgcv itself the multi-smooth gap shrinks from ~5–12× (FD era) to ~1.5–3× (REML) /
~2–5× (UBRE/GCV) — the `2m+1` multiplier is gone; the residual is inner-fit iteration counts
and compiled-C constant factors, no longer scaling with m.

## Two-tier-gap resolution (the question the brief flagged)

**(a) — optimizer imprecision, and it no longer exists at canonical links.** The reported
~0.2-EDF / few-percent-fitted gap on a plain poisson REML gam did **not reproduce** anywhere in
a 30-dataset sweep: at mgcv's exact sp the engines agree to machine precision (EDF 8.46510112
both, deviance to 1e-8), and free selection agrees to ≤ 1.5e-4 EDF. Not a REML/UBRE formulation
difference at canonical links (score verified to 6e-11 on poisson at fixed sp).

**BUT the investigation surfaced a real, precisely-diagnosed criterion nuance at NON-canonical
links (new finding, pre-existing, NOT introduced or altered by A2):** `reml_score` computes the
Laplace determinant `log|X'WX + S_λ|` with **Fisher** weights; mgcv uses the **full-Newton**
Hessian weights. Proven exactly on BOTH non-canonical cases: at mgcv's selected sp,
probit `0.5·(log|A_Fisher| − log|A_Newton|) = 0.0335846` vs observed score delta `0.0335879`;
nb-log `−0.0420104` vs `−0.0420104` (seven digits — the deltas have opposite signs, exactly as
the weight orderings predict). At canonical links W̃ = W and the difference vanishes (hence the
historical 6e-11 poisson match). Consequences: probit/nb-log REML *free selection* sat ~1e-3–8e-3
EDF from mgcv (fitted ≤ 2.3e-4 rel); fixed-sp fits were identical (EDF to 1e-6, fitted to 5e-7).

**Status: FIXED (user-approved, same 4.7.0 bundle).** Fixed-dispersion non-canonical REML now
computes `log|X'W̃X + S_λ|` with Newton weights (Cholesky on the pivoted rank block — the PD
test is a Cholesky attempt, NOT a slogdet sign, which misses even-dimensional negative-definite
matrices), and `reml_gradient_glm` differentiates the same determinant (`dW̃/dη` by a second
central difference at the eps^(1/4) step). Canonical links keep the exact QR path bit-for-bit
(poisson fixed-sp REML still matches mgcv to 3.7e-11). Post-fix evidence:
- Fixed-sp REML score vs mgcv: probit **1.3e-8** (was 0.034), nb **2.0e-8** (was 0.042).
- Free selection (sweep table above re-run): probit REML EDF gap **+0.000000** (was +7.6e-3),
  sp rel 4.9e-7; nb-fixed **+3.0e-5** (was +1.4e-3); nb-est **−5e-6**, θ̂ 3.2973 vs mgcv
  3.29700. Whole-sweep worst |EDF gap| now **2.1e-4** (a GCV case, not REML).
- R10 separation hard case (quasi-separable probit REML): py sp 1.65835 vs mgcv 1.6584, smooth
  EDF 4.7003 = mgcv's 4.7003, REML score identical to 6 dp, same R-style separation warning.
- Non-PD Newton Hessian → warned Fisher-determinant fallback, score and gradient making the
  same deterministic decision (unit-tested via injected negative weights; never observed on
  real data — separation-grade probit stays PD across the full λ range).
- `reml_score` now takes `X` (internal API change, all call sites updated); e2e probit-REML
  selection pin + fixed-sp score pin vs mgcv added to the suite. Scripts:
  `proto_newton_reml.py`, `verify_newton_fix.py`, `sep_probit.{py,R}`, `sep_probit_e2e.py`
  in `a2-scripts/`.

## To-do at the bundled 4.7.0 re-validation (R5/R6, per convention meta.json is untouched now)

`subsystems/gam/meta.json` updates owed when the 4.7.0 artifacts are frozen:
- fidelity item "Smoothing-parameter selection": drop "the finite-difference step is sized
  above the inner-P-IRLS noise floor" — selection is analytic-gradient for ALL families now.
- `reference.what` + `tier2_free_selection_optimizer_tier`: the tier-2 bound is no longer
  "mgcv's exact-derivative Newton vs pystatistics' finite-difference L-BFGS-B" — both engines
  use exact derivatives; canonical-link tier-2 tightens to ~1e-4 EDF (re-measure on the frozen
  artifacts); non-canonical REML now matches mgcv too (Newton determinant fix above), so the
  optimizer-tier wording can tighten across the board.
- `cpu_speed_g3` + limitation "Multi-smooth GENERALIZED GAMs": GLM families no longer keep the
  FD path; refresh the multi-smooth ratios from the frozen perf artifacts.
- Finding H4: mark fully resolved (Gaussian at 4.6.1, GLM families at 4.7.0).
- Add the Fisher/Newton REML determinant finding as a finding entry, status FIXED at 4.7.0
  (see the Status block above for the evidence to cite).

Scripts: `handoffs/a2-scripts/` — proto_grad.py / proto_rankdef.py (gradient-vs-criterion-FD
verification incl. the Fisher-shortcut disproof), final_sweep.R + final_sweep_py.py (the R6
sweep above), bench_data.R + bench_fd_vs_an.py (perf), diag_logdet.py + nb_diag.py +
probit_tier1.{R,py} (the Fisher/Newton determinant proof), nb_profile.py (the nb+GCV theta
degeneracy proof), repro_majors.py + check_stationary.py + stress_scan{,_fd}.py (the adversarial-review majors: repro, mgcv comparison, 40-config torture panel vs the FD baseline). Run py scripts with `PYTHONPATH=<pystatistics repo>`; R scripts first
(they write the shared CSVs).
