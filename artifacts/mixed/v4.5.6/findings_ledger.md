# mixed (LMM + GLMM + GRM) v4.5.6 — findings ledger (BLESSED)

Validation of the **PyPI 4.5.6** build. Report: `reports/mixed-v4.5.6.md`. Scope:
first-time validation of `glmm()` (GLMM, Laplace/nAGQ=1) vs `lme4::glmer`, bundled
with a re-validation of `lmm()` (LMM) and `grm_lmm()` (low-rank / GRM) — both
byte-identical at 4.5.6 (the `4.5.1→4.5.6` diff touches only glmm files).

## Version arc

`4.5.1` (LMM + GRM blessed; glmm deferred) → first-time glmm validation surfaced a
cascade, each fixed + released before re-validating:

- **`4.5.2`** — G1 (nAGQ=0→true Laplace nAGQ=1) + G2 (fixed-effect SE transpose).
- **`4.5.3`** — G3 (fail-loud on free-dispersion families).
- **`4.5.4`** — G4 (Poisson PIRLS overflow + silent variance-collapse).
- **`4.5.5`** — G5 (structure-exploiting solver; removes the O(#groups³) gap).
- **`4.5.6`** (blessed) — G4 completed (variance-boundary fits verified by a
  derivative-free rescue on flat deviance surfaces).

## Correctness — clean

- **GLMM vs lme4::glmer (Laplace nAGQ=1), two-tier contract:** fixed-effect β +
  logLik/deviance/AIC/BIC ~1e-3 or better (tight); SEs, Wald z, variance
  components, BLUPs ~1–2% (optimizer/Laplace tier). Across binomial/logit (cbpp),
  poisson/log (grouseticks), a correlated random slope (glmm_slope_synth) and a
  probit link (cbpp_probit).
- **R10 red-team:** quasi-separation (huge-but-finite slope, ~4e-5), singular RE
  (variance→0 matching glmer isSingular), unbalanced/singletons (~2e-4) — all
  match glmer's behaviour. **R15 default invocation** equals the explicit
  `(1|group)` spec (0.0) and matches glmer's default (variance ~6e-5). **A6
  fail-loud:** gaussian/gamma families raise `ValidationError`.
- **Variance estimation is a true classifier:** on a seed×true-variance sweep,
  `glmm` variance matches `glmer` to ≤ ~1e-3 absolute — never collapsing when the
  variance is real, never inventing it when it is zero.
- **LMM vs lme4/lmerTest** and **GRM vs rrBLUP** — byte-identical to the blessed
  4.5.1 evidence (re-confirmed on CPU here).

## Performance

- **GLMM CPU vs glmer:** no complexity-class gap after 4.5.5 — empirical exponent
  ~O(n^0.48) (pystatistics) vs ~O(n^0.68) (glmer). A residual constant factor
  (speedup 0.29× at G=25 → 0.63× at G=400/n=6000) is a documented R2 cost
  (two-stage Laplace + in-fit SEs + robustness probe). CPU-only by design.
- **LMM CPU vs lme4:** random-intercept 3.0–4.7× faster; random-slope competitive
  → 0.4× at scale (finding F4, documented). GRM GPU (CUDA/MPS) carried forward.

## GPU

- **grm_lmm only** (general LMM/GLMM CPU-only by design). CF-1 no-silent-wrong gate
  + earns-its-keep (CUDA) + MPS-honest studies **carried forward unchanged from
  4.5.1** — the grm_lmm GPU/backend code is byte-identical at 4.5.6.

## Findings — final disposition

- **G1** nAGQ=0-not-Laplace — RESOLVED 4.5.2.
- **G2** silently-wrong fixed-effect SE (correlated predictors) — RESOLVED 4.5.2.
- **G3** free-dispersion families silently-wrong — RESOLVED 4.5.3 (fail loud).
- **G4** Poisson overflow + silent variance-collapse — RESOLVED 4.5.4, completed 4.5.6.
- **G5** dense O(#groups³) scaling gap — RESOLVED 4.5.5 (structure-exploiting solver).
- **G6** no `is_singular` diagnostic for glmm — DOCUMENTED (behaviour-correct; gap only).
- **G7** no aggregated-binomial / prior-weights / offset — DOCUMENTED (scope).
- **F1–F4** (LMM/GRM) — unchanged from 4.5.1.
