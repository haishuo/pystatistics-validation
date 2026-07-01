# mixed (LMM) v4.4.1 — findings ledger

Status: **measurement complete on 4.4.1; bless target is the bundled 4.4.2.**
GPU chip (task_6f7467bd) complete → verdict **(B) no GPU for general LMM/GLMM**
(documented decision; `mixed` stays GPU-free, no `backend=`). Both sessions
independently found the `_compute_se` defect; the CPU fix is THIS session's to own
(per `handoffs/mixed-recommendation.md`).

**Decisions (user, 2026-06-30):** ONE consolidated **v4.5.0** update (minor — additive
API + perf rewrite), implemented by a SEPARATE implementation chip, in strict gated
order: **(1) CPU fixes** (F2a `_compute_se` form A; F1 singular warning + `is_singular`;
F2b structure-exploiting PLS incl. crossed; A.3 autodiff θ-gradient) → **(2) test CPU
finished correctly** → **(3) GPU** low-rank/GRM model (chip §C, separate honestly-named
model, CF-1 fp64-Gram gate) → **(4) test GPU** → **(5) hand back to THIS validation
session** for final frozen validation + red-team + bless against PyPI 4.5.0. The chip
does NOT publish and does NOT bless (this session owns that). This session does NOT edit
`../pystatistics`. GLMM full validation still deferred ([[mixed-glmm-deferred]]).

Scope of this pass: **LMM only** (`lmm`); GLMM (`glmm`) deferred. CPU-only module
(no GPU backend exists — GPU warrant under separate investigation).

## What passed (priorities 1 + 2)

- **Correctness vs R (lme4 + lmerTest), two-tier contract — clean.**
  - TIGHT tier (fixed effects β, logLik, REML criterion, AIC/BIC): ≤ ~3e-14 across
    sleepstudy, Penicillin, Dyestuff, Dyestuff2, Pastes, under both REML and ML.
  - OPTIMIZER tier (SE, t, Satterthwaite df, p, variance components, BLUPs): ≤
    ~1.2e-4 (bobyqa vs scipy L-BFGS-B). RE correlation is the loosest sub-tier.
  - Artifacts: `runs/correctness_cpu_powerhouse.json` + `_summary.csv`.
- **R10 hard cases — match R's behaviour, incl. its diagnostics.**
  - Dyestuff2 (scalar boundary) and HC1 (2x2 correlation→±1 boundary): estimates
    match; both ARE singular. HC2 unbalanced/singletons, HC3 extreme ICC≈0.9999
    (widest optimizer-tier varcomp gap, 1.4e-3, documented), HC4 nested-correct
    (machine precision) + naive-crossed (matched footgun).
  - Artifacts: `runs/hardcases_cpu_powerhouse.json` + `_summary.csv`.
- **R15 default invocation:** bare `lmm(y, X, groups=...)` ≡ explicit `(1|g)` spec
  exactly (0.0) and matches lmer (3e-16).

## Findings

### F1 — no singular/boundary-fit diagnostic (priority 1 / A6) — GATHER
- lme4 emits `boundary (singular) fit` (a *message*) on degenerate fits; isSingular()
  is TRUE. pystatistics returns the **correct** boundary MLE but `converged=True`,
  emits **no warning** (`py_warnings=[]`) and exposes **no `is_singular` accessor** —
  the user gets no signal a variance/correlation hit the boundary.
- Evidence: Dyestuff2 (`correctness_*`) and HC1 (`hardcases_*`): `py_singular`
  (driver-side detector) = True while `py_warnings` = [] and R's `r_diagnostics`
  carries the boundary message.
- Classification: numbers correct → NOT silent-wrong, NOT R16. Missing fail-loud
  diagnostic → R18 gather. Candidate library fix (propose, don't edit unilaterally):
  a boundary warning + `is_singular` accessor matching lme4.

### F2 — CPU complexity-class gap vs lme4 (priority 3 / R1) — SPLIT into F2a + F2b

**F2a — `_compute_se` dense n×n O(n³), ~83% of `lmm()` runtime (the bulk).** Line
`solvers.py:547` forms `V_star = ZΛΛ'Z' + I` (n×n) and `np.linalg.solve`s it. Fix:
the p×p Schur form already available — `Var(β̂) = σ²·(RX·RXᵀ)⁻¹` where `pls.RX` is
the lower-Cholesky of the Schur complement. Implement as
`sigma_sq * RX_inv.T @ RX_inv` (**form A**). VERIFIED here: machine-identical to the
dense path (2.2e-15 vs current SEs, which match R), and ~12,000× faster at n=3200
(145.7ms→12µs; class change n³→p³). NOTE: do NOT copy `_compute_se_glmm`'s literal
`RX_inv @ RX_inv.T` — that is the wrong product for the LMM RX convention
(0.17 rel error); the GLMM RX is the transpose convention.

**F2b — dense Z / Z'Z in the deviance loop (the residual architectural cost).**
`build_z_matrix` materializes dense Z (n × ΣJ_k·q_k); `solve_pls` forms dense Z'Z +
Cholesky each eval → OOM at G≈2000 (chip: 27 GB) / InstEval infeasible. Fix
(structure-exploiting PLS): single grouping factor → batched per-group dense
Cholesky (block-diagonal system; chip measured ~1400× over dense, no new dep);
crossed → genuine sparse Cholesky with fill-reducing ordering (CHOLMOD-class;
scipy `splu` alone hits the fill-in wall, chip: q=15 000→134 s) — needs a
dependency decision. **In scope for 4.4.2 per user.**

Original combined F2 measurement (dense path, 4.4.1): empirical exponent vs n
pystat ~1.41–1.46 vs lmer ~0.19–0.34; 25–63× slower at n=3200, widening.
Evidence: `runs/cpu_speed_powerhouse.json`.

#### (historical) F2 — surfaced before the split
- pystatistics builds a DENSE Z (`np.zeros((n, ΣJ_k·q_k))`) and a dense PLS/Cholesky;
  lme4 uses sparse Z + sparse Cholesky.
- Empirical exponent (time vs n, group-count sweep at fixed group size 8):
  pystatistics ~1.41 (intercept) / ~1.46 (intercept+slope); lme4 ~0.19 / ~0.34.
  Crossover G≈100 (intercept) / G≈10–25 (slope); at n=3200 pystatistics is 25×
  (intercept) / 63× (slope) slower, and widening. Effectively unusable at
  InstEval scale (n≈73k, thousands of groups; dense Z is memory-bound).
- User buys NOTHING for the slowdown (same REML estimates) → R1 defect, not R2.
- Severity: complexity-class gap but NOT a correctness showstopper (no wrong
  numbers; slow-not-silent). Sits on the R16/R18 line — surfaced to the user.
  Any fix (sparse Z/Cholesky) is a `../pystatistics` change → propose, don't edit
  unilaterally. **The GPU investigation chip (task_6f7467bd) independently flagged
  this structure; F1/F2 disposition to be decided jointly with its findings.**
- Evidence: `runs/cpu_speed_powerhouse.json` + `_summary.csv`.

## Not yet done (blocked on the F1/F2 decision)
- subsystems/mixed/meta.json, reports/mixed-v4.4.1.md (render-from-artifacts),
  R4 constitutional audit write-up, manifest, commit + push.
