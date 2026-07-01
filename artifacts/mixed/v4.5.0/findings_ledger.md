# mixed (LMM) v4.5.0 — findings ledger

Validation of the **PyPI 4.5.0** build (the consolidated bundle the implementation
chip shipped). LMM surface. GLMM still deferred ([[mixed-glmm-deferred]]). GRM
model validation tracked separately (Phase 2 below).

## Prior findings — DISPOSITION at 4.5.0

- **F1 (no singular diagnostic) — RESOLVED.** `LMMSolution.is_singular` + a boundary
  `RuntimeWarning` now ship. Verified: dyestuff2 (scalar boundary) AND HC1
  (correlation→±1) both give `py_singular==r_singular==True` with the warning; the
  4.4.1 gap where HC1 was undetected is closed. Clean fits stay `is_singular=False`,
  no warning.
- **F2a (`_compute_se` dense n×n O(n³)) — RESOLVED.** Form A p×p Schur factor.
  Correctness unchanged (SE still matches R to the optimizer tier, e.g. sleepstudy
  se_max_rel 1.4e-5).
- **F2b (dense-Z complexity-class gap) — RESOLVED (structure-exploiting solver).**
  intercept-only now **near-linear (exponent 0.45 vs lmer 0.39) and BEATS lmer at
  every size (3.2–5.3×)**, incl. G=2000/n=16000 (30ms vs 127ms) — vs 25× slower at
  4.4.1. No new dependency (scipy `splu` + MMD_AT_PLUS_A for crossed). See F4 for the
  residual multi-term-RE gap. Evidence: `runs/cpu_speed_powerhouse.json`.

## Correctness at 4.5.0 — clean (unchanged vs 4.4.1)

Two-tier contract vs lme4/lmerTest holds. Honest tiering (corrected from the 4.4.1
ledger, which overstated the tight tier):
- **Fixed-effect β:** ~1e-16 .. 3e-13 (near machine; conditional-on-θ linear solve).
- **logLik / REML criterion / AIC/BIC:** ~1e-11 — technically optimizer-bound
  (bobyqa vs L-BFGS-B) but far tighter than varcomps because the likelihood is flat
  near the optimum. (The 4.4.1 ledger's "≤3e-14 tight tier" captured only β; logLik
  is ~2e-11. Recorded via memory `mixed-gate-tight-tier-liveR`.)
- **SE, t, Satterthwaite df, p, variance components, BLUPs:** ≤ ~1.3e-4 (optimizer
  tier). penicillin Satterthwaite df 8.1e-5 (baseline ~1e-4) — within tier.
REML+ML both. R10 hard cases + R15 default all hold (except the F3 tail below).
Evidence: `runs/correctness_cpu_powerhouse.json`, `runs/hardcases_cpu_powerhouse.json`.

## NEW findings at 4.5.0

### F3 — structured optimizer fails to converge in the extreme variance-ratio tail — GATHER (fail-loud), disposition pending
- Regime: ICC → 1 with residual variance ~5 orders below the RE variance
  (HC3: between≈100, residual sd 0.03). The structured deviance triggers an
  L-BFGS-B **ABNORMAL line-search termination** at ~15 iters; returns
  `converged=False` + a warning and a variance component ~17% off (107.9 vs R
  130.3), worse logLik (105.309 vs R 105.471). The **4.4.1 dense path converged
  here** (n_iter=19, matched R) — so it is a **regression** vs the dense predecessor,
  in a regime lme4 handles.
- Bounded: residual sd sweep × 5 seeds — non-convergence only at sd=0.03
  (ICC≈0.99999) and only 1/5 seeds; 0/5 by sd≥0.1 (ICC≤0.9999).
- Severity: **fails LOUD** (converged=False + warning) → NOT silent-wrong, NOT R16.
  Narrow + intermittent → R18 gather. But it is a Prime-Directive "match R" regression
  in a legitimate (if extreme) regime. Candidate fix: robust fallback (dense/eigh) or
  better start/bounds when the ratio is extreme. Evidence: HC3 in
  `runs/hardcases_cpu_powerhouse.json` + bounding sweep (this session).

### F4 — multi-term (random-slope) RE ~2× slower than lmerTest at scale — R2/gather
- intercept+slope: competitive at small G (1.0–1.3×) but 0.54× (G=400) → 0.41×
  (G=2000, 528ms vs 214ms); exponent 0.66 vs lmer 0.45. Much milder than the old
  F2 complexity-class gap (was 63× slower) — now a ~2× constant-factor gap.
- Attribution: ~26% is the Satterthwaite df pystatistics computes **in-fit** by
  default, which `lmerTest::lmer` defers (R2 — a quantity the timed lmer fit skips);
  with `compute_satterthwaite=False`, G=2000 is 364ms vs 214ms (~1.7×). The rest is
  the correlated-2×2-RE deviance/optimizer. Prime candidate for the deferred autodiff
  gradient (A.3, ~7× fewer evals). Evidence: `runs/cpu_speed_powerhouse.json` +
  Satterthwaite-share probe (this session).

## Phase 2 — GRM / low-rank model (grm_lmm)

**CPU correctness vs R `rrBLUP::mixed.solve` (rrBLUP 4.6.3) — VALIDATED (dev run on
4.5.0; re-freeze at 4.5.1).** Grid h²∈{0.2,0.5,0.8} × sizes (n=400/M=80, n=800/M=150),
K = W Wᵀ/M ≡ rrBLUP `mixed.solve(y, X, K, Z=I)`:
- β ≤ 3.4e-7, β SEs matched, σ_g² ≤ 2.0e-5, σ_e² ≤ 4.7e-6, heritability ≤ 9.3e-6,
  genetic-value BLUP correlation = 1.000000 (max-scaled ≤ ~3e-5). All optimizer-tier
  or tighter. Evidence: `runs/grm_correctness_cpu_powerhouse.json`.
- **logLik NOT compared across engines:** rrBLUP's restricted LL uses a different
  additive constant (values differ by ~6 while every estimate matches to ~1e-6) —
  recorded, not toleranced. Documented, not a defect.

**GPU (CF-1 gate + R11 + earns-keep) — SCRIPTED, run PENDING 4.5.1 + devices.**
`generate_grm_gpu.py` ready: CF-1 cond(W) boundary sweep (assert silent_wrong=0;
accept⟹correct-vs-fp64, refuse⟹loud), R11 gpu_fp64-vs-cpu_fp64 (CUDA), and the
earns-keep cpu/gpu(fp32)/gpu_fp64 speed pivots. To run on MPS (Mac) + CUDA (Forge)
against PyPI 4.5.1. Chip already reported CF-1 silent_wrong=0 on MPS+CUDA and CUDA
fp32 13–16× (MPS correctness-only); this session re-verifies at the blessed version.

## Bless plan
Blocked on the F3 fix landing as **4.5.1** (chip task_6dcf1e17). On 4.5.1: re-freeze
ALL artifacts (LMM correctness/hardcases/cpu_speed + GRM cpu + GRM gpu MPS/CUDA)
against PyPI 4.5.1, R4 audit of the new surface (grm_lmm/GRMSolution/is_singular),
write meta.json + reports/mixed-v4.5.1.md, bless. F4 documented as an R2/autodiff
roadmap item; A.3 autodiff + torch-policy are roadmap.
