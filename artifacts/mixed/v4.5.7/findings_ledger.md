# mixed (LMM + GLMM + GRM) v4.5.7 — findings ledger (BLESSED · module complete)

Validation of the **PyPI 4.5.7** build. Report: `reports/mixed-v4.5.7.md`. This is
the R6 re-validation after the **A.3** optimization; it **completes the `mixed`
module** (`lmm` + `glmm` + `grm_lmm` all validated). `gam` is now unblocked.

## Version arc

`4.5.1` (LMM + GRM blessed) → glmm first-time validation cascade `4.5.2–4.5.6`
(G1–G5; G6/G7 documented) → **`4.5.7` A.3** (analytic θ-gradient; closes F4).

## A.3 — analytic θ-gradient (the F4 fix)

- `lmm()`'s batched single-factor optimizer now uses the **exact analytic gradient**
  of the profiled REML/ML deviance instead of finite differences — one deviance
  eval per L-BFGS-B step instead of `2·dim(θ)+1` (~2.3× fewer).
- **Correctness preserved:** LMM estimates unchanged vs 4.5.6 — fixed effects
  identical, variance components within the ~1e-4 optimizer tolerance; still match
  `lme4` on the two-tier contract. Analytic gradient matches finite-diff to 1.3e-7;
  its deviance is bit-identical to the plain objective.
- **F4 closed:** correlated random-slope LMM ~1.35× faster end-to-end; competitive
  with `lmerTest` at small/moderate G (speedup_vs_r 1.97× at G=10, 0.99× at G=100,
  0.52× at G=2000 — up from 1.06×/0.73×/0.42×). The residual at-scale gap is the
  in-fit Satterthwaite df (documented R2 cost lmerTest defers), not a defect.
- **torch rejected on evidence:** benchmarked torch-CPU autograd vs numpy-analytic —
  torch is slower until n≈12k and only ~1.3× at n≈24k. Gradient is pure numpy; no
  new dependency. (See memory `a3-torch-loses-on-cpu`.)
- Scope: single grouping-factor (batched) path (where F4 lives). Crossed/nested
  (sparse) designs keep the finite-difference path.

## Unchanged surfaces (R8)

- **glmm()** — BIT-IDENTICAL to 4.5.6 (A.3 does not touch its numerics; verified on
  the correctness grid).
- **grm_lmm()** — CPU unchanged; GPU studies (CUDA/MPS) carried forward from 4.5.1
  (backend code byte-identical).

## Findings — final disposition

- **G1–G5** (glmm) — RESOLVED (4.5.2–4.5.6). **G6/G7** (glmm) — DOCUMENTED.
- **F1–F3** (LMM/GRM) — RESOLVED (≤4.5.1).
- **F4** (multi-term RE slow) — **RESOLVED at 4.5.7 (A.3)**; residual is a
  documented R2 cost (in-fit Satterthwaite df).

**`mixed` is complete.** No open findings. Next: `gam` (unblocked).
