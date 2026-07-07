# Final whole-library completeness audit — path to a single pystatistics bless

**Audit chip deliverable.** Read-only on library code. Produces a triaged ledger of every
documented limitation / gap / deferral across the entire public surface, so we know exactly
what must be closed to earn one whole-library bless. Current core: **pystatistics 4.6.12** (PyPI).

---

## 1. Scope & method

**Read:** `RIGOR.md` (R2, R7, R10–R18 incl. the R18 bless precondition), `ROADMAP.md`
(module table §78–88 + Open work §92–152), all 11 report-units under `reports/`, the library
`CHANGELOG.md`, and `git log` over `../pystatistics/pystatistics/{mvnmle,mice,<corpus>}/`.
Limitations from the 9 corpus reports were harvested via three parallel read-only passes;
mvnmle and mice were read directly (prime suspects). No library code was edited, no releases
cut, no drivers run. Git was used only to establish version currency; R was not needed.

**The 11 report-units and version currency vs 4.6.12:**

| # | Report-unit | Blessed ver | Report file | Module code last changed | Currency verdict |
|---|---|---|---|---|---|
| 1 | regression | 4.3.2 | `reports/regression-v4.3.2.md` | 2026-06-29 (4.3.2) | Current-in-code; nominal version drift only. Red-teamed under modern regime. |
| 2 | survival | 4.3.3 | `reports/survival-v4.3.3.md` | 2026-06-29 (4.3.3) | Current-in-code; nominal drift. Red-teamed. |
| 3 | multivariate | 4.4.1 | `reports/multivariate-v4.4.1.md` | 2026-06-30 (4.4.1) | Current-in-code; nominal drift. Red-teamed. |
| 4 | mixed | 4.5.7 | `reports/mixed-v4.5.7.md` | 2026-07-01 (4.5.7) | Current-in-code; nominal drift. Red-teamed. |
| 5 | gam | 4.6.1 | `reports/gam-v4.6.1.md` | 2026-07-02 (4.6.1) | Current-in-code; nominal drift. Red-teamed. |
| 6 | timeseries | 4.6.6 | `reports/timeseries-v4.6.6.md` | **2026-07-06 (4.6.12)** | **Code changed after bless** — A7-1 arima/auto_arima GPU fail-loud fix landed at 4.6.12; covered by `reports/a7-torchfree-certification-v4.6.12.md` but the 4.6.6 report predates the new fail-loud behavior. Minor re-bless owed. |
| 7 | montecarlo | 4.6.8 | `reports/montecarlo-v4.6.8.md` | 2026-07-05 (4.6.8) | Current-in-code; nominal drift. Red-teamed. |
| 8 | ordinal+multinomial | 4.6.10 | `reports/ordinal-multinomial-v4.6.10.md` | 2026-07-05 (4.6.10) | Current-in-code; nominal drift. Red-teamed. |
| 9 | anova+descriptive+hypothesis | 4.6.11 | `reports/anova-descriptive-hypothesis-v4.6.11.md` | 2026-07-06 (4.6.11) | Current-in-code; nominal drift. Red-teamed. |
| 10 | **mvnmle** | **3.18.0** | `reports/mvnmle-v3.18.0.md` | **2026-06-27 (1bb1501)** | **STALE — pre-red-team regime.** Numerically unchanged since bless, but API renamed by the 4.0 release and never subjected to R10/R12/R13/R15/R18. See §6. |
| 11 | **mice** | **3.16.3 / 3.18.0** | `reports/mice-v3.16.3.md`, `reports/mice-v3.18.0.md` | **2026-06-27 (1bb1501)** | **STALE — pre-red-team regime.** Same as mvnmle. Report still documents the removed `use_fp64` kwarg. See §6. |

**Key currency finding.** For the 9 corpus modules, each module's *own source* last changed at
(or before) the version that blessed it — so their nominal version staleness is an artifact of
*other* modules advancing the shared version line, not of their code drifting. They were all
validated under the modern RIGOR regime (R10 red-team, R12/R13 fp32 gates, R15 default,
R18 bless precondition). **One exception: timeseries** — its code changed at 4.6.12 (the A7-1
arima GPU fail-loud fix), so its 4.6.6 report is code-stale; the A7 certification report covers the
fix but a timeseries re-bless at 4.6.12 is owed (minor, additive fail-loud).

---

## 2. Per-module findings table (triage summary)

Triage key: **A** = fixable gap, must be CLOSED before whole-library bless (R18: "documented ≠
fixed"). **B** = permanent documented limitation with an accepted justification. **VERIFY** =
suspected undocumented gap to confirm (not yet A or B).

| Module | Stale? | Confirmed limitations (A / B) | VERIFY candidates |
|---|---|---|---|
| regression | nominal | 6× B (fp32 gates, Metal-no-fp64, timer artifact), Gamma-dispersion convention (B) | GLM families beyond 5; non-canonical links; profile CIs; anova/drop1; diagnostics; contrasts |
| survival | nominal | **Stratified Cox/KM (A)**, Breslow-not-validated (A-coverage), prior-art trawl (A-noncode); CPU-only, Harrell-C, MPS-torch-ver, discrete-time py-loop = B | time-varying Cox; left-truncation KM; robust/cluster SE; cox.zph/residuals; G-rho unvalidated |
| multivariate | nominal | 5× B (MPS randomized-only, randomized-approx, fp32 gate, consumer-fp64 slow, FA CPU-only); F3 small-n FA = B (R2) | FA factor scores; promax validation; prcomp tol truncation |
| mixed | nominal | **G7 offset/weights/agg-binomial (A)**, **G6 is_singular surface (A)**; free-dispersion fail-loud, no-GPU, F4 slowdown, autodiff-deferred = B | GLMM crossed/nested multi-factor RE; `||` uncorrelated RE; nAGQ>1 |
| gam | nominal | **H4 GLM-family analytic gradient (A, deferral)**, **`bs='cc'` cyclic (A)**; approx F-stat, AIC edf2, tp knot-cap, REML-Gamma fail-loud, GPU-removed = B | **tensor/`te()`/multivariate `s(x,z)` smooths (highest)**; `by=` smooths; other bases (`ps`,`re`,…) fail-loud proof; NB/Tweedie families implemented? |
| timeseries | **code-stale (4.6.12)** | ETS-loglik convention, ADF MacKinnon-p, ndiffs boundary, Whittle-batch scope, NA fail-loud, small-K GPU = B | **`xreg`/drift in arima/auto_arima (high)**; damped-trend `Ad` validated?; stepwise=FALSE/approx toggles; PI-value validation |
| montecarlo | nominal | **BCa non-ordinary jackknife divergence (A/VERIFY)**; GPU-statistic restriction, small-n skew, no-fp64, tiny-n R2 = B | boot.ci `h`/`hinv`/`index` transforms; antithetic sim; multi-group/stratified permutation |
| ordinal+multinomial | nominal | **polr `loglog`/`cauchit` links unimplemented (A)**; no-formula-predict, separation fail-loud, small-n GPU, optimizer-tier, MPS-host-fp64 = B | `weights`; multinom `decay>0`; LR-test/profile CIs; t-vs-z convention |
| anova+descriptive+hypothesis | nominal | none open (F1–F5 fixed at 4.6.11); cor/quantile n=1e6 slowdown, no-GPU = B | effect sizes (eta²/d); non-Tukey post-hoc; cor.test inference; chisq simulate.p; rm mixed designs |
| **mvnmle** | **STALE** | Missing-at-random, GPU-fp32, non-degenerate-cov, R-stops-at-p=50 = B **on their face** — but never R10/R12/R13/R15/R18-tested; large CPU-vs-fp32-GPU loglik gaps (Δ up to ~146) never gated | whole modern-regime surface (see §6) |
| **mice** | **STALE** | MAR, fp32-GPU, polr-separation-ridge, fully-missing-reject, high-cardinality = B on their face — never modern-regime tested; report documents removed `use_fp64` API | whole modern-regime surface (see §6) |

---

## 3. CLOSE list (bucket A) — ordered by importance

Per R18, "it's documented" is not a resting state. Each item below is either fixed to bless, or
must be re-argued as a true permanent B with an accepted one-sentence justification.

**A0 — mvnmle & mice modern-regime re-validation (BIGGEST ITEM).**
*What closing requires:* a full first-time-rigor pass at 4.6.12 for each (R10 red-team, R12/R13
fp32 no-silent-wrong gate, R15 default-invocation, R4 API audit, R18 ledger). Both have fp32
GPU paths that estimate/invert a covariance and were blessed before any of these rules existed;
mvnmle's own report tables show CPU-vs-fp32-GPU `|Δloglik|` up to ~146 (`mvnmle-v3.18.0:§5`
CPU-vs-MPS table, gss p=25/20/50) with **no no-silent-wrong gate proven** — the exact CF-1 /
R12 exposure class. Source is git-confirmed numerically unchanged since bless, so this is a
*rigor-bar* re-validation, not a code-change re-validation. **This is the gating work for a
whole-library bless.** (Detail in §6.)

**A1 — survival: stratified Cox / stratified KM.**
*Source:* `survival-v4.3.3:§6` — the report itself refuses to call it an edge case: *"stratified
Cox PH is real, commonly-needed functionality for biostatisticians … a known missing feature, not
merely an edge case."* Currently `strata=` raises `NotImplementedFeatureError`.
*What closing requires:* implement stratified partial likelihood + stratified KM, validate vs
`survival::coxph(...+strata())`; **or** a deliberate, documented product decision to defer with an
accepted scope justification. Fail-loud today, so not a showstopper — but the single most
substantive self-flagged feature gap in the corpus.

**A2 — gam: H4 GLM-family analytic smoothing-parameter gradient (confirmed MUST-FINISH).**
*Source:* `gam-v4.6.1:§6` + Finding H4 + ROADMAP Open work §112–116. Gaussian path done at 4.6.1;
Poisson/binomial (UBRE) and non-Gaussian REML still use finite-difference `(2m+1)` inner fits.
*What closing requires:* the Wood-2011 implicit-derivative gradient for GLM families; re-validate
multi-smooth non-Gaussian speed vs mgcv. Correctness is unaffected (estimates match mgcv on the
two-tier contract) — this is a Guarantee-3 performance close, already decided.

**A3 — gam: `bs='cc'` cyclic basis (and a fail-loud-on-unknown-`bs` proof for other bases).**
*Source:* `gam-v4.6.1:§4 fidelity_a6`, §5 G2 table — only `cr`/`tp` implemented; `cc` raises.
*What closing requires:* implement cyclic cubic (a mainstream mgcv smooth for seasonal/periodic
data) **or** a prominent scope statement; plus confirm every *other* `bs=` (`ps`,`re`,`ds`,`gp`,…)
fails loud rather than silently accepting an unknown basis. Interacts with VERIFY-gam tensor
smooths (§5) — resolve together.

**A4 — mixed G7: glmm offset / prior-weights / aggregated-binomial.**
*Source:* `mixed-v4.5.7:§6` + Finding G7; ROADMAP §142–145. `glmm()` takes a single unit-weight
response and fails loud on `cbind(k,n-k)`, weights, offset.
*What closing requires:* add at least **offset** (Poisson exposure) and prior-weights, or a
prominent declared scope carve-out. Offset/grouped-binomial (`cbpp`) are canonical glmer usage; a
whole-library reviewer will expect them. Fail-loud today → not a showstopper.

**A5 — ordinal: polr `loglog` and `cauchit` links.**
*Source:* `ordinal-multinomial-v4.6.10:§6` + Finding F-B — 3 of MASS::polr's 5 links implemented;
`loglog`/`cauchit` fail loud ("Unknown link").
*What closing requires:* implement the two links (the machinery mirrors the existing three) or
accept as B — but "we implement 3 of R's 5 links" is a weaker B than the hardware/fail-loud items.

**A6 — mixed G6: glmm `is_singular` surface.**
*Source:* `mixed-v4.5.7:§6` + Finding G6; ROADMAP §142–144. Numbers are correct (match glmer's
boundary MLE); only the diagnostic flag/warning is absent, while `lmm()` already exposes it.
*What closing requires:* surface the flag on `GLMMSolution` (cheap, additive; lmm machinery exists).
Low severity — not silent-wrong — but a trivial-to-close asymmetry a reviewer notices.

**A7 — montecarlo: BCa acceleration on non-ordinary bootstrap.**
*Source:* `montecarlo-v4.6.8:§6` — on balanced/parametric/stratified bootstrap BCa falls back to
delete-1 jackknife acceleration that *"differs slightly from R's regression default in the tails."*
*What closing requires:* either match R's regression `empinf` on these paths, or confirm+argue the
divergence is bounded and accept as B. Borderline A/B — resolve the magnitude before deciding.

**A8 — survival: validate Breslow ties (coverage) + owed prior-art trawl (non-code).**
*Source:* `survival-v4.3.3:§6` (Breslow implemented but not tabulated) and Finding R13 / ROADMAP
§146–148 (prior-art trawl owed for the "first discrete-time survival on GPU at scale" novelty
claim before publication). Cheap: one head-to-head row for Breslow; a literature trawl for the
claim. Neither is a code defect; both are open owed items.

**A9 — timeseries: re-bless at 4.6.12.**
*Source:* the A7-1 fix (arima/auto_arima now fail loud on unsupported GPU backend, `CHANGELOG 4.6.12`)
changed timeseries behavior after its 4.6.6 bless. Covered by `a7-torchfree-certification-v4.6.12.md`
but a refreshed timeseries report at 4.6.12 closes the code-staleness. Additive fail-loud, trivial.

---

## 4. PERMANENT limitations (bucket B) — accepted justifications

These are deliberate, argued scope choices. Each carries a one-sentence justification a reviewer
accepts. (Grouped; full per-item quotes in the harvest — citations inline.)

**Hardware / precision (all B):**
- Metal (MPS) has no fp64 and no on-device SVD/eigh → gpu_fp64 is CUDA-only; MPS PCA is
  randomized-only; fp32-GPU covariance needs CPU for full precision (`regression:§6`,
  `multivariate:§6`, `mvnmle:§6`, `mice:§6`, `ordinal:§6`). *Justification:* hardware facts of
  Metal, handled by host-fp64 gates or fail-loud, never silent CPU substitution (A6).
- Consumer-NVIDIA fp64 ≈ 1/64 fp32 throughput → gpu_fp64 PCA is a correctness option, not a speed
  path (`multivariate:§6`). *Justification:* honest positioning of a hardware throughput fact.

**Deliberate fail-loud numerical gates (all B):**
- Conservative fp32 accept/refuse gates that false-refuse a force-recoverable band rather than ever
  false-accept (`regression:§6`, `multivariate:§6`). *Justification:* a numerical gate must bias to
  safe-side; `force=True` recovers the band; never accepts a wrong fit (R12).
- GPU-OLS Cholesky refuses cond > 1e6 unless `force=True` (`regression:§6`). *Justification:* fp32
  squares the condition number; the exact fp64 QR/CPU path is the documented fallback.

**CPU-only-by-construction (all B):**
- coxph partial likelihood (`survival:§6`); factor_analysis (`multivariate:§6`); lmm/glmm
  (`mixed:§6`); gam (`gam:§6`, GPU removed at 4.6.0 for proven silent-wrongness, H1);
  anova/descriptive/hypothesis (`anova…:§5`). *Justification:* each is sequential or closed-form
  with no amortizable FLOPs; a manufactured GPU path would add silent-precision risk for zero gain,
  matching all prior art (lme4/mgcv/MixedModels.jl stay CPU-only).

**Documented convention differences (all B, disclosed and estimator-invariant-anchored):**
- Gamma SE via ML dispersion vs R's Pearson/df (`regression:§4 gamma_link`, ~1e-2);
  Harrell-C vs Therneau tie-aware concordance (`survival:§4 coxph_concordance`, ~1e-4);
  gam AIC classical-df vs mgcv edf2 (`gam:§6`); ETS full-Gaussian vs concentrated pseudo-loglik
  (`timeseries:§6`, exact deterministic offset, selection unaffected); ADF MacKinnon vs tseries
  coarse-table p (`timeseries:§6`, pystatistics is the *more* accurate); optimizer-tier agreement
  for polr/multinom (`ordinal:§6`). *Justification:* each is a named, valid alternative convention;
  the estimator-invariant quantities (loglik, fitted probs, W, epsilons) match tightly, and the
  divergence is disclosed on the accessor.

**Deliberate fidelity fail-loud / scope (all B):**
- Non-finite (NA) input raises everywhere vs R's `na.action` (`timeseries:§6`); glmm refuses
  free-dispersion gaussian/gamma (`mixed:§6`, G3); gam REML refuses free-dispersion non-Gaussian →
  use GCV (`gam:§6`); multinom fails loud on complete separation (`ordinal:§6`, stricter than nnet);
  montecarlo GPU refuses a non-declared/non-vectorizable statistic (`montecarlo:§6`, M1/M6);
  arima_batch is Whittle-only non-seasonal, fails loud on unsupported specs (`timeseries:§6`).
  *Justification:* refusing loudly beats silently solving a different problem (Guarantee 2).

**Documented R2 performance costs (all B, bounded, self-reversing or off critical path):**
- Small-n FA vs factanal ~0.65× at n=500 (`multivariate:§6`, F3 — data-dependent iteration count,
  leads 1.7–3.3× at n≥2000); glmm ~1.5–3× vs glmer constant factor (`mixed:§6` — buys in-fit SEs +
  robustness, no complexity gap); random-slope LMM ~2× vs lmerTest (`mixed:§6`, F4 residual —
  in-fit Satterthwaite df); tiny-n montecarlo CPU lag (`montecarlo:§4 g3` — fixed per-replicate
  Python overhead, self-reverses by n~1000, 3.4× faster at n=10000); cor:pearson/quantile ~3× at
  n=1e6 (`anova…:§5` — constant-factor numpy overhead, both sub-35ms); small-K arima_batch GPU
  dispatch overhead (`timeseries:§6`); discrete-time person-period py-loop (~1.7% of end-to-end,
  `survival:§6`). *Justification:* each is a named constant factor or off the critical path — no
  complexity-class gap (R1); the user is buying more output or robustness (R2).

**Deferred optimizations that do not affect correctness (B):**
- mixed autodiff θ-gradient (`mixed:§6` — benchmarked and rejected on CPU, roadmap-only).
  (gam H4 GLM-family gradient is listed as **A2** above because ROADMAP marks it MUST-FINISH; if the
  program instead accepts it as permanent, its B justification is "estimates correct and unchanged;
  only multi-smooth non-Gaussian speed affected.")

**Statistical properties, not defects (B):**
- Bootstrap small-n skew under-coverage (`montecarlo:§6`, M3 — reproduced by R too); non-identified
  coefficients under separation in multinom/polr (`ordinal:§6` — MLE genuinely at infinity).

**mvnmle/mice on-their-face limitations (B *pending A0 re-validation*):**
- MAR assumption, non-degenerate-covariance requirement, R-stops-at-p≈50 (`mvnmle:§6`); MAR,
  polr-separation-ridge, fully-missing-column reject, high-cardinality polyreg (`mice:§6`).
  *Justification:* each is a standard, defensible scope statement — **but** these were written under
  the pre-red-team regime and must be re-confirmed (not re-discovered) during the A0 pass before they
  count as accepted B.

---

## 5. VERIFY list — suspected undocumented gaps to confirm (not yet A/B)

These are R default capabilities a skeptic would expect that the reports never mention. Each needs
a yes/no + fail-loud check against the actual library surface before triage. Ordered by risk to a
whole-library bless.

**Highest risk (large scope holes if real):**
- **gam tensor / multivariate smooths** — `te()`, `ti()`, `s(x,z)` never mentioned; the report only
  ever discusses univariate `f_j(x_j)`. Core mgcv functionality. If absent, must be named explicitly.
- **timeseries `xreg` / drift** — regression-with-ARIMA-errors and `include.drift` never mentioned
  in `arima`/`auto_arima`; standard forecast usage.
- **gam families** — the §5 fail-loud error string lists `nb`/`negative.binomial` as *valid* family
  names, yet §1/§3 never validate NB against mgcv. Implemented-but-untested, or listed-but-absent?
- **mixed GLMM crossed/nested multi-factor RE + `||` uncorrelated covariance** — LMM hard-cases
  exercise nested/crossed; GLMM tables appear single-grouping-factor only.

**Medium risk (common defaults, likely deliberate scope but unstated):**
- regression: GLM families beyond the 5 validated (inverse.gaussian, quasi*); non-canonical links
  (probit/cloglog/cauchit/sqrt); profile-likelihood CIs (`confint.glm`); anova/drop1 deviance tests;
  diagnostics (hat, Cook's D, residual types); non-treatment contrasts.
- survival: time-varying `(start,stop)` Cox + `tt()`; left-truncation KM; robust/cluster-robust SE
  (`robust=TRUE, cluster=`); `cox.zph` + Schoenfeld/martingale residuals; G-rho weighted log-rank
  is implemented but unvalidated.
- ordinal/multinomial: `weights`; multinom `decay>0` (validated only at decay=0); LR-tests /
  profile CIs; t-vs-z + no-p-value MASS convention.
- montecarlo: boot.ci `h`/`hinv`/`index` variance-stabilizing transforms; antithetic sim;
  multi-group / stratified / blocked permutation (report is strictly two-group).
- anova/descriptive/hypothesis: effect sizes (eta²/partial-eta²/omega²/Cohen's d); non-Tukey
  post-hoc (pairwise.t.test, Games-Howell); `cor.test` inference conventions (Fisher-z, AS 89);
  chisq `simulate.p.value`; rm mixed between/within + multi-within-factor designs.
- gam: `by=` factor-smooth / varying-coefficient smooths; other `bs=` fail-loud proof.
- multivariate: FA factor scores (`scores=`); promax oblique-rotation validation + `Phi`;
  prcomp `tol` rank truncation.
- timeseries: damped-trend `Ad` in a validated ZZZ case; `stepwise=FALSE`/`approximation` toggles;
  prediction-interval *value* validation (tables show point-forecast pass, PI columns unclear).

---

## 6. mvnmle & mice verdict

**Verdict: RE-VALIDATE both at 4.6.12 under the modern RIGOR regime. Do NOT justify as
"unchanged."**

**Evidence the *numbers* are unchanged (so this is a rigor-bar, not a code-change, re-validation):**
`git log` over `../pystatistics/pystatistics/mvnmle/` and `.../mice/` shows the last commits
touching either are **1c1c752** (the 4.0 consistency release) and **1bb1501** (Phase 8 audit),
both dated 2026-06-27 — nothing since. The 4.0 commit message states *"No statistical/numerical
change."* So the estimators the 3.18.0/3.16.3 reports blessed are byte-for-byte the current ones.

**Why "unchanged" does NOT let them ride, three independent reasons:**

1. **Public API drift (R4 fails today).** The 4.0 consistency release renamed the entire public
   surface: `use_fp64` removed → `backend=` device+precision encoding; `maxit`→`max_iter`;
   `m`→`n_imputations`; results → `*Solution`; exceptions → `ValidationError(ValueError)`. Both
   reports still describe the *old* API — e.g. `mice-v3.18.0:§2` documents *"fp64 available on CUDA
   via use_fp64"*, a kwarg that no longer exists publicly at 4.6.12 (it survives only as an internal
   backend field). A reader following either report against 4.6.12 hits renamed parameters. An R4
   constitutional audit against the current constitution fails on both.

2. **Never subjected to R10/R12/R13/R15 (the rules that caught every recent showstopper).** Both
   were blessed before the red-team, the fp32 no-silent-wrong gate, regime-conditional guarantees,
   and default-invocation validation existed. Both ship **fp32 GPU paths that estimate/invert a
   covariance** — precisely the class where CF-1 materialized as a silent-wrong R16 showstopper in
   *ordinal/multinomial* (fp32 `X'WX` inversion → negative-variance SEs) and *gam* (fp32 Gram → negative
   EDF). mvnmle's own report tables show CPU-vs-fp32-GPU `|Δloglik|` of **113–146** at p=20–50
   (`mvnmle-v3.18.0:§5`, CPU-vs-MPS scaling table, gss rows) with **no principled accept/refuse gate
   proven** — the report treats it as "fp32 tolerance," which is exactly the R12 gap (a loose gate
   that may accept a biased fp32 fit). mice's GPU fp32 imputation (`mice:§4 gpu_vs_cpu`) has the same
   unproven-gate status. This is a live, unaudited correctness exposure, not a cosmetic staleness.

3. **R15 default + R10 hard cases never run.** Neither report validates the naive-default call or
   the adversarial regime (near-degenerate covariance for mvnmle; (quasi-)separation under chained
   equations for mice — the latter's ridge stabilizer is described but never stress-tested to the
   accept/refuse boundary, `mice:§6`).

**Scope of the re-validation (per module, at 4.6.12):** R4 API audit against current CONVENTIONS;
R10 red-team grid matching R's failures/warnings; **R12/R13 fp32 no-silent-wrong gate on the GPU
covariance path (the priority item — CF-1 class)**; R15 default-invocation; R18 findings ledger.
Numbers are git-confirmed stable, so the CPU head-to-head vs R mvnmle / R mice should reproduce —
but the GPU gate, the hard cases, and the API surface are genuinely unvalidated at the modern bar.

---

## 7. Adjacent, not blocking the bless

- **survival prior-art trawl** — arXiv/PyPI/GitHub trawl for the "first discrete-time survival on
  GPU at scale" novelty claim (`survival-v4.3.3` Finding R13; ROADMAP §146–148). A *publication*
  gate, not a library gap. (Also listed as A8 because it is an open owed item, but it blocks a
  paper, not the bless.)
- **survival 4.3.3 changelog wording tidy** — the illustrative convergence examples ("all-censored →
  n_iter=0", "separated → converged=False") describe narrower code paths than the validated
  behaviour (ROADMAP §149–152). Cosmetic doc-wording, not a defect.

---

## 8. Recommended path to a single whole-library bless

**Target version: 4.7.0** (a feature+consistency minor that closes the A-list), with mvnmle/mice
re-validation possibly forcing patch bumps under R16 if their GPU-gate audit surfaces a silent-wrong.

**Order (dependency- and risk-first):**

1. **A0 — mvnmle & mice modern-regime re-validation at 4.6.12** *(do first; highest correctness
   risk).* If the fp32 GPU covariance gate proves silent-wrong (CF-1 class), it is an R16 showstopper
   → stop-fix-release-restart, which sets the version floor for everything else. If clean, they
   re-bless additively.
2. **A1 (stratified Cox/KM)** and **A4 (glmm offset/weights)** — the two substantive missing-feature
   gaps a whole-library reviewer will demand; both currently fail loud (safe) so they gate the bless,
   not correctness.
3. **A3 + A5 (gam `cc` / polr links)** and the **§5 VERIFY sweep** — resolve the "does R do X by
   default that we silently lack?" questions (tensor smooths, xreg, NB family, GLMM multi-factor)
   into explicit yes/no + fail-loud; promote confirmed real gaps to A, close or scope each.
4. **A2 (gam H4 GLM-family gradient)** — the confirmed MUST-FINISH performance close.
5. **A6 (glmm is_singular), A7 (montecarlo BCa non-ordinary), A8 (Breslow row + prior-art trawl),
   A9 (timeseries 4.6.12 re-bless)** — the cheap/additive tail; bundle under R18.
6. Fold all corpus B-items into a single library-wide "accepted limitations" appendix so the bless
   is a defensible ledger, and re-confirm each mvnmle/mice B during A0.

**The one-line gate:** the library is blessable when A0 is clean-or-fixed and every A-item is
either closed or re-argued as an accepted B — nothing rests in "documented" (R18).

---

## Locked decisions (coordinator, 2026-07-07)

**Governing principle (user, this session) — the parity bar for "done":**
> The **math must never differ from R**. The **interface may** differ — omit an R
> capability, add something R lacks, or change a convention — **only** with a GOOD
> reason that is **explained and documented**. Absence by oversight (undocumented,
> unjustified) is not permitted. See memory `pystatistics-raison-detre`.

Consequences for this ledger:
- **A3 (gam `bs='cc'` cyclic), A5 (polr `loglog`/`cauchit` links), A7 (BCa non-ordinary
  bootstrap) → CLOSE**, not documented-subset. They are math/capability, so full
  R-parity is required.
- **The VERIFY sweep is now a hard gate, not a nicety.** Any R default capability found
  silently absent with no justified documented rationale is a **mandatory new A-item**
  (implement it), OR must be converted into a **B with a written justification** — it may
  not remain an undocumented gap. Running it next (decision #1, agreed).
- Every B-item in the final ledger must carry its GOOD-reason justification; a B without
  one is really an A.

**Sequence agreed:** VERIFY sweep (lock scope) → A0 mvnmle/mice re-validation (R16 risk +
version floor) → bundle remaining closes into **4.7.0** → single whole-library bless.

---

## VERIFY sweep results

Read-only capability-parity sweep resolving §5 and the priority-target list against the actual
4.6.12 library surface (source at `../pystatistics`), with R spot-checks (`Rscript`) where math
parity was in question. **Headline safety finding: every gap below FAILS LOUD** (TypeError on an
absent kwarg, `NotImplementedFeatureError`, `ValidationError "Unknown ..."`, or a fail-loud family
guard). **No silent-wrong capability was found — there is no new R16 showstopper in scope.** The
distinction that matters for the bless is therefore purely *documented-justification-present (B)*
vs *undocumented (A)*. `CONVENTIONS.md` was checked and carries **no** capability-scope
declarations (it governs naming/backend only), so almost none of these gaps have a constitutional
justification; the only in-tree documentation is per-function docstrings and the reports.

### Priority targets

| # | Target | Verdict | Evidence | Existing justification | Action |
|---|---|---|---|---|---|
| 1 | gam `te()`/`ti()`/`s(x,z)` tensor & multivariate smooths | **ABSENT-A (fail-loud)** | `gam/_smooth.py` `s(var_name: str,...)` takes a single variable; `hasattr(gam,'te')==False`, no `ti`. Only univariate `f_j(x_j)`. | none (report/CONVENTIONS silent) | **A** — implement or write a prominent scope carve-out. Core mgcv surface. |
| 1b | gam bases beyond `cr`/`tp` (`cc`,`ps`,`bs`,`re`,`ds`,`gp`) | **ABSENT-A (fail-loud)** | `_VALID_BASIS_TYPES=frozenset({'cr','tp'})`; any other `bs=` raises `ValidationError`. Fail-loud proof for unknown `bs` **holds**. | `cc` already tracked as **A3** | A3 covers `cc`; the fail-loud-on-unknown-`bs` requirement is satisfied. |
| 1c | gam `by=` factor-smooth / varying-coefficient | **ABSENT-A (fail-loud)** | `s()` has no `by` parameter → unknown kwarg is a TypeError. | none | **A** — implement or scope-document. |
| 1d | gam `nb`/`negative.binomial` family | **ABSENT-A (fail-loud, unusable as mgcv `nb()`)** | Family name IS accepted (`_FAMILY_CLASSES['nb']`) and PIRLS handles `negative.binomial` (`gam/_pirls.py:217`), BUT theta is not jointly estimated — `gam(...,family='nb')` raises `ValidationError "Cannot compute deviance without theta"`. mgcv `nb()` estimates theta by default (R spot-check: theta≈2.95, edf≈7.7). | none | **A** — either estimate theta (mgcv parity) or document that only fixed-theta NB is offered. |
| 2 | timeseries `xreg` / `include.drift` (regression w/ ARIMA errors, drift) | **ABSENT-A (fail-loud, documented-as-TODO)** | `arima()`/`auto_arima()` have no `xreg` param → TypeError. Docstring `_arima_fit.py` Notes: *"Not yet implemented: `fixed` (parameter masking) and `xreg` — including drift terms, so models R reports 'with drift' cannot be requested."* | docstring documents the **absence** but gives **no reason** ("not yet implemented" = TODO) | **A** — a TODO is not a justification under the bar. Implement xreg+drift, or convert the docstring note into a GOOD documented reason. auto_arima silently cannot select drift models R would. |
| 2b | timeseries `fixed=` parameter masking | **ABSENT-A (fail-loud, documented-as-TODO)** | same docstring note; no `fixed` param. | TODO only | **A** — same as 2. |
| 2c | timeseries `stepwise=FALSE` / exhaustive search | **PRESENT** | `auto_arima(stepwise=...)`; `stepwise=False` → exhaustive grid (`_arima_order.py:648`). | n/a | none. |
| 3 | mixed GLMM crossed / nested multi-factor RE | **PRESENT (matches glmer)** | `glmm(...,groups={'g1':..,'g2':..})` runs; R glmer `(1\|g1)+(1\|g2)` spot-check: fixef (−0.2382, 1.0157) vs py (−0.2382, 1.0157); var g1 0.42149/0.42151, g2 0.003985/0.003989. Nested expressible via interaction group id. | n/a | **none — clean.** Resolves the highest §5 "large scope hole" risk. |
| 3b | mixed `\|\|` uncorrelated RE covariance | **ABSENT-A (silent-capability, low sev)** | `random_effects: dict[str,list[str]]` always builds a full Cholesky Λ (correlated); no diagonal/`\|\|` option (`_random_effects.py`). Correlated random slopes ARE present (corr in `var_components`). | none | **A (minor)** — add a diagonal-covariance flag or document. Not silent-wrong (correlated is a valid, just different, model). |
| 4 | ordinal polr `loglog`/`cauchit` links | **ABSENT-A (fail-loud)** | `polr(link='loglog'/'cauchit')` → `ValidationError "Unknown link ... Valid: cloglog, logistic, probit"`. Exactly 3 of MASS's 5. | tracked as **A5** | A5 confirmed; 3/5 present, other 2 fail loud. |
| 5 | survival stratified Cox / stratified KM | **ABSENT-A (fail-loud)** | `coxph(strata=...)` and `kaplan_meier(strata=...)` accept the kwarg but raise `NotImplementedFeatureError` (verified). The KM `strata` plumbing exists in `SurvivalDesign` but the solver stub still raises. | tracked as **A1** | A1 confirmed absent + fail-loud. |
| 5b | survival `ties=` Efron/Breslow/exact | **PARTIAL** | `coxph(ties=...)` accepts `efron`(default)/`breslow` (`solvers.py:245`); anything else fails loud. R's `exact` ties ABSENT. Breslow present but untabulated. | Breslow-coverage = **A8**; `exact` undocumented | A8 covers Breslow row; **`exact` ties → new A (minor)** or documented scope. |

### Medium-risk defaults (lighter pass — all fail-loud, all undocumented unless noted)

| Target | Verdict | Evidence | Action |
|---|---|---|---|
| regression non-canonical links `cloglog`/`cauchit`/`sqrt` | **ABSENT-A (fail-loud)** | `_LINK_CLASSES` = identity/logit/log/inverse/probit only. `fit()` has no `link=` kwarg (→TypeError); probit reachable only via a `Family` instance. cloglog/cauchit/sqrt unimplemented. | **A** — implement or scope-doc; binomial-cloglog is a common default. |
| regression families `inverse.gaussian`, `quasi*` | **ABSENT-A (fail-loud)** | not in `_FAMILY_CLASSES`; `ValidationError "Unknown family"`. (`inverse.gaussian` is named in `_pirls.py` but unreachable via the family registry.) | **A** — implement or scope-doc. quasi-families are mainstream `glm()`. |
| regression profile-likelihood CIs (`confint.glm`) | **ABSENT-A** | no profile-CI accessor; only Wald. | **A** — implement or document Wald-only. |
| regression `anova`/`drop1` deviance tests | **ABSENT-A** | no such entry points. | **A** — implement or document. |
| regression diagnostics (hat/Cook's D/residual types) | **ABSENT-A** | no leverage/Cook's/residual-type accessors. | **A** — implement or document. |
| regression non-treatment contrasts | **ABSENT-A** | treatment coding only. | **A** — implement or document. |
| montecarlo boot.ci `h`/`hinv` variance-stabilizing transforms | **ABSENT-A** | `_ci.py` computes all 5 interval TYPES (normal/basic/perc/bca/stud) but no monotone-transform hooks. | **A (minor)** — implement or document. |
| montecarlo antithetic / stratified-blocked permutation | **ABSENT-A** | report + code are strictly two-group / ordinary sim. | **A (minor)** — document scope. |
| multivariate FA factor scores (`scores=`) | **ABSENT-A** | `factor_analysis()` has no `scores=` param (R factanal default "none", but regression/Bartlett are standard). | **A (minor)** — implement or document. |
| multivariate `prcomp` `tol` rank truncation | **ABSENT-A** | `pca()` has `n_components` but no `tol=`. | **A (minor)** — document (n_components is equivalent-enough). |
| multivariate promax oblique rotation + `Phi` | **PRESENT** | `_rotation.py` varimax + promax, "matches R stats::varimax/promax". | validate `Phi` in a report row (cheap). |
| anova effect sizes η²/partial-η² | **PRESENT** | `AnovaSolution.eta_squared`, `.partial_eta_squared`. | none. |
| anova post-hoc Tukey / Bonferroni | **PRESENT**; omega²/Cohen's d/Games-Howell **ABSENT-A (minor)** | `_posthoc.py` tukey_hsd + bonferroni_pairwise. | document the omissions or add. |
| hypothesis chisq `simulate.p.value` | **PRESENT** | `_chisq_test.py` Monte-Carlo p + Yates. | none. |
| survival time-varying `(start,stop)` Cox, left-truncation KM, robust/cluster SE, `cox.zph`/residuals | **ABSENT (fail-loud / not exposed)** | no start-stop or `tt()`; no robust SE kwarg; no zph. | **A (bundle)** — these + stratified (A1) are the survival feature cluster; scope or implement. |

### New CLOSE items promoted to bucket A (undocumented capability gaps)

Under the locked bar ("absence by oversight is not permitted"), each of the following is either
implemented for 4.7.0 or given a written GOOD justification (→B). **None is silent-wrong; all fail
loud, so none forces an R16 restart.** Grouped by weight:

**Substantive (a reviewer will demand these):**
- **VA-1 gam tensor/multivariate smooths** `te()`/`ti()`/`s(x,z)` — core mgcv surface, entirely absent, undocumented.
- **VA-2 gam `by=` smooths** — factor-smooth / varying-coefficient interactions, absent, undocumented.
- **VA-3 gam usable `nb()` family** — theta not estimated; only fixed-theta reachable, undocumented.
- **VA-4 timeseries `xreg`+drift and `fixed=`** — documented only as "not yet implemented" (a TODO, not a justification); auto_arima cannot select R's drift models.
- **VA-5 regression non-canonical links (`cloglog`/`cauchit`/`sqrt`) and families (`inverse.gaussian`/`quasi*`)** — mainstream `glm()` defaults, absent, undocumented.

**Medium / low (cheap to implement or to justify as B):**
- **VA-6 regression profile CIs, `anova`/`drop1`, diagnostics (Cook's/hat/residual types), non-treatment contrasts.**
- **VA-7 mixed `||` uncorrelated RE covariance.**
- **VA-8 survival `exact` ties; time-varying/start-stop Cox; left-truncation KM; robust/cluster SE; `cox.zph`** (bundle with A1).
- **VA-9 montecarlo `h`/`hinv` transforms; antithetic; stratified/blocked permutation.**
- **VA-10 multivariate FA factor scores; prcomp `tol`.**
- **VA-11 anova omega²/Cohen's d; Games-Howell post-hoc.**

### Confirmed PRESENT (came back clean)

- **mixed GLMM crossed/nested multi-factor RE** — numerically matches glmer (the biggest §5 risk, cleared).
- **timeseries `stepwise=FALSE`** exhaustive search.
- **survival `ties='breslow'`** (present; validation row still owed = A8).
- **multivariate varimax + promax** rotations.
- **anova η²/partial-η², Tukey + Bonferroni post-hoc; hypothesis chisq `simulate.p.value`.**
- gam **fail-loud-on-unknown-`bs`** proof (satisfies the A3 sub-requirement).

### Confirmed B (justified) — none newly earned here

No capability gap in this sweep carries a GOOD documented justification today. `CONVENTIONS.md`
has no capability-scope section, and the only in-tree note (timeseries `xreg` "not yet
implemented") is a TODO, not a justification. **Every A-item above therefore needs either
implementation or a newly-written one-sentence justification before it can rest as B** — the bar
forbids the current undocumented-absence state.

---

## Consolidated close plan — parity rule applied (coordinator, 2026-07-07)

Rule applied (ratified this session): **IMPLEMENT** anything a working statistician routinely
reaches for in the method's core use (textbook-standard, or a common/default option in the R
reference fn); **JUSTIFY→B** only if genuinely specialist/niche, R-farms-it-to-a-dedicated-package,
or superseded by a better path we already offer; **tie-break → implement** ("effort" is never a
good reason to omit); **never leave named-but-broken.** Every item below is IMPLEMENT or JUSTIFY-B;
nothing rests undocumented.

### Workstream 0 — gating re-validation / doc (not implement-vs-justify; do FIRST)
- **A0** mvnmle + mice modern-regime re-validation at 4.6.12 (R10/R12/R13/R15/R4/R18). Sets the
  version floor; own separate-session chip. *(No silent-wrong found by the VERIFY sweep, but the
  fp32 covariance gate is still unproven at the modern bar — that's A0's job.)*
- **A9** timeseries re-bless at 4.6.12 (code moved via the A7-1 fix). Trivial.
- **A8** survival Breslow-ties validation row (cheap) + owed prior-art trawl (non-code, pre-paper).

### IMPLEMENT — mandatory (grouped by weight → proposed release tier)

**Tier A (large; each ≈ its own release):**
| id | item | module | est. size |
|---|---|---|---|
| VA-1 | tensor / multivariate smooths `te()`/`ti()`/`s(x,z)` | gam | LARGE (core mgcv) |
| A1+VA-8 | survival feature cluster: stratified Cox/KM + start-stop/time-varying Cox + left-truncation KM + `cox.zph` + robust/cluster SE | survival | LARGE |
| VA-4 | `xreg` + drift (regression w/ ARIMA errors, "with drift") | timeseries | MEDIUM-LARGE |

**Tier B (medium; one coordinated feature release, e.g. 4.7.0):**
| id | item | module | est. size |
|---|---|---|---|
| VA-2 | `by=` factor-smooth / varying-coefficient | gam | MEDIUM |
| VA-3 | usable `nb()` — estimate theta (mgcv parity) | gam | MEDIUM |
| A3 | `bs='cc'` cyclic basis (+ `ps` P-splines) | gam | MEDIUM |
| A2 | H4 GLM-family analytic sp-gradient (perf close) | gam | MEDIUM |
| VA-5 | general `link=` (cloglog/cauchit/sqrt) + families (inverse.gaussian, quasipoisson/quasibinomial) | regression | MEDIUM |
| A4 | glmm offset / prior-weights / aggregated-binomial | mixed | MEDIUM |
| VA-6a | `anova`/`drop1` deviance tests + diagnostics (Cook's/hat/residual types) | regression | MEDIUM |

**Tier C (cheap; fold into the nearest bundle or an R18 patch):**
| id | item | module | est. size |
|---|---|---|---|
| A5 | polr `loglog`/`cauchit` links | ordinal | SMALL |
| A6 | glmm `is_singular` surface | mixed | SMALL |
| VA-7 | glmm `\|\|` uncorrelated RE covariance | mixed | SMALL-MED |
| VA-10a | FA factor scores (`scores=`) | multivariate | SMALL-MED |
| VA-11 | omega² / Cohen's d + Games-Howell post-hoc | anova | SMALL-MED |
| VA-4b | `fixed=` parameter masking | timeseries | SMALL-MED |

**Pending-magnitude (resolve then implement-or-justify):**
| id | item | note |
|---|---|---|
| A7 | montecarlo BCa `empinf` on balanced/parametric/stratified bootstrap | measure the tail divergence; if bounded → B, else implement R's regression `empinf` |
| VA-6b | regression profile-likelihood CIs (`confint.glm`) | mainstream for GLM → lean IMPLEMENT; acceptable strong-B (Wald disclosed) if we choose |

### JUSTIFY → B (write into a new `CONVENTIONS.md` "Capability scope" section; NO code)
These are genuinely specialist, or interface-only (math identical), or covered by a better path:
- **gam exotic bases** `re`/`ds`/`gp`/`fs` — specialist; `cr`/`tp`(/`cc`/`ps`) cover the mainstream; unknown `bs=` fails loud.
- **regression non-treatment contrasts** (helmert/sum/poly) — **math identical** (fitted values unchanged); only the coefficient *coding* differs → a pure interface choice, treatment coding is R's default too.
- **regression fully-general `quasi(link, variance)`** — quasipoisson/quasibinomial (the mainstream overdispersion cases) ARE implemented (Tier B); the arbitrary-variance constructor is specialist.
- **survival `exact` (exact partial-likelihood) ties** — Efron (default) + Breslow cover standard practice; `exact` is rarely needed and costly; fail-loud.
- **montecarlo `h`/`hinv` variance-stabilizing transforms, antithetic sampling, stratified/blocked permutation** — advanced variance-reduction; all 5 CI types + ordinary/stratified bootstrap present; transform is applyable by the caller.
- **multivariate prcomp `tol=` rank truncation** — `n_components=` is the equivalent control.

### Proposed release shape (versions indicative, will shuffle)
1. **A0 outcome** (mvnmle/mice) — possibly a patch if their gate needs a fix; sets the floor.
2. **4.7.0** — Tier B implements + Tier C + the JUSTIFY-B `CONVENTIONS.md` "Capability scope" section + A9/A8/A5/A6. The bar's "no undocumented gaps" is met here for everything except the Tier-A big builds.
3. **4.8.0** — gam tensor smooths (VA-1).
4. **4.9.0** — survival feature cluster (A1+VA-8).
5. **timeseries xreg/drift** (VA-4) — own medium release, sequence to taste.
6. **Whole-library bless** once Tier A lands and every item is IMPLEMENT-done or B-documented.

**The gate (unchanged):** blessable when A0 is clean/fixed and every A/VA item is either implemented
or resting as a B with a written GOOD justification — nothing undocumented (R18 + the parity bar).
