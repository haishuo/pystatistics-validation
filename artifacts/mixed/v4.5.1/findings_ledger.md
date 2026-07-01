# mixed (LMM + GRM) v4.5.1 — findings ledger (BLESSED)

Validation of the **PyPI 4.5.1** build. Report: `reports/mixed-v4.5.1.md`. Scope:
`lmm()` (LMM) + the new `grm_lmm()` (low-rank / GRM) model. `glmm()` deferred.

## Version arc

`4.4.1` (baseline first-time validation, CPU-only, dense) → surfaced F1/F2a/F2b →
bundled into `4.5.0` (singular reporting, p×p SEs, structure-exploiting solver, +
the new `grm_lmm` GPU model) → red-team of 4.5.0 found F3 (+ a silent-wrong
sibling) → fixed in **`4.5.1`** (blessed). F4 is a documented R2 cost.

## Correctness — clean

- **LMM vs lme4/lmerTest, two-tier contract:** fixed-effect β ~1e-16..3e-13 (tight);
  logLik/REML/AIC/BIC ~1e-11; SE/df/p/varcomps/BLUPs ≤ 1.3e-4 (optimizer tier).
  REML+ML. R10 hard cases (boundary ×2, unbalanced/singletons, extreme ICC, nested +
  matched footgun) match R's behaviour incl. diagnostics. R15 default exact.
- **GRM vs rrBLUP::mixed.solve:** β ≤ 3.4e-7, varcomps ≤ 2.0e-5, h² ≤ 9.3e-6, BLUP
  correlation = 1.000000. (logLik not compared — different additive constant.)

## GPU (grm_lmm only; general LMM/GLMM CPU-only by design, verdict B)

- **CF-1 no-silent-wrong gate:** silent_wrong_count = 0 on MPS **and** CUDA
  (accept⟹correct ≤2.3e-3, refuse⟹loud past cond(W)≈1e4). R9/R12/R13/R14 satisfied.
- **Earns its keep (CUDA):** fp32 6.6×/14.7× over CPU; gpu_fp64 exact (1.8e-8),
  1.2–1.6× (R11 hardware pivot). **MPS correctness-only (0.27–0.74×) — reported
  honestly, not a win.**

## Findings — final disposition

- **F1** (no singular diagnostic) — **RESOLVED 4.5.0** (`is_singular` + warning).
- **F2a** (O(n³) SEs) — **RESOLVED 4.5.0** (p×p Schur form A; machine-identical).
- **F2b** (dense-design complexity gap) — **RESOLVED 4.5.0** (structure-exploiting
  solver; random-intercept now beats lme4 3.2–5.3× to G=2000; no new dependency).
- **F3** (extreme variance-ratio non-convergence + silent-wrong sibling) —
  **RESOLVED 4.5.1** (derivative-free Nelder-Mead fallback; verified vs live lme4,
  0 non-convergences across the residual-sd × seed sweep, ≤ 8e-4).
- **F4** (multi-term RE ~2× slower than lmerTest at scale) — **DOCUMENTED (R2)**:
  ~26% is in-fit Satterthwaite (lmerTest defers it), rest the correlated-RE
  optimizer; autodiff θ-gradient is the roadmap fix. Not a complexity-class gap.

## R4 constitutional audit — conforms

`lmm`/`glmm` expose no `backend=` (CPU-only, matching the no-GPU decision);
`grm_lmm` uses `backend=<device>[_<precision>]` (no `use_fp64`), unknown backend →
`ValidationError`, fp32-Gram refusal → `NumericalError`; `LMMSolution.is_singular`
bool; `GRMSolution.z_values` (Wald z, A3). No deviations.

## Roadmap (not in this bless)

- Autodiff θ-gradient (F4 optimization; deferred pending library-wide torch-tier decision).
- GLMM (`glmm()`) full first-time validation vs lme4::glmer.
