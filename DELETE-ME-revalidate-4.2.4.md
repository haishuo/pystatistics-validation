# Re-validate against pystatistics 4.2.4 (MPS GLM gated-CG inner solve)

> Throwaway coordination note (mirrors the 4.2.2 / 4.2.3 hand-offs). Delete once
> the survival + regression reports are re-run and committed for 4.2.4.

## What changed in 4.2.4 (and why benchmarks move)

The **MPS float32 GLM IRLS inner solve** changed from Cholesky-of-`XᵀWX` to a
**matrix-free gated conjugate gradient** on the operator `H v = Xᵀ(W(X v))`
(squaring-free — `XᵀWX` is never formed). The host float64 Newton-decrement
acceptance gate is unchanged and remains the sole arbiter of acceptance.

Consequences for the benchmarks (Apple Silicon / MPS only — **CUDA is unchanged**,
its Cholesky path is robust and `gpu_fp64` still matches CPU fp64 to ~1e-12):

- **MPS now CONVERGES where it previously failed loud.** The discrete-time flchain
  person-period sweep at **quarterly** (and the rest of the sweep) now fits on MPS
  via CG to float32 tier (covariate coef rel err ~1e-5–4e-5 vs CPU fp64). The old
  Cholesky path crashed not-positive-definite from a cold MPS context at quarterly.
  So the **MPS device-pivot table and the precision/accuracy table will change** —
  cells that were "fail-loud → route to CPU" at quarterly are now "converged on MPS".
- The refuse path still exists for genuinely float32-infeasible designs, but on MPS
  the **refuse message now recommends `backend='cpu'`** (not `gpu_fp64`, which is
  CUDA-only). No automatic CPU fallback — it raises and the user chooses.

## What to do

1. **Install the PyPI-released 4.2.4** (never a local checkout — `require_pypi`
   must pass). This note is written before publish; do not start until
   `pip install pystatistics==4.2.4` resolves from PyPI.
2. **Re-run the survival benchmark** (discrete-time GPU/CPU; the flchain
   person-period sweep) and regenerate `reports/survival-v4.2.4.md` from the frozen
   v4.2.3 report. Update the MPS narrative prose:
   - was: *"MPS works-or-fails-loud"* (quarterly routed to CPU on a loud failure).
   - now: *"MPS converges via gated CG; refuses loud → `backend='cpu'` at the
     genuine precision floor."* The on-device envelope widened (quarterly now fits
     on MPS); the loud-fail boundary is now the genuine fp32 floor, not the
     squaring knife-edge.
3. **Re-run the regression benchmark** (GPU GLM stability across binomial/Poisson/
   Gamma on MPS) and regenerate `reports/regression-v4.2.4.md`. The MPS device-pivot
   / precision tables move for the same reason; CUDA and CPU rows should be
   unchanged.
4. **Fold in the design study's conclusions.** The one-time research artifact
   (`pystatistics/docs/research/mps-glm-solver-study/`) is being deleted from the
   pystatistics repo as part of the 4.2.4 change (recoverable from git history);
   its conclusions belong in these validation reports going forward.

## Caveat to record in the reports

MPS kernel behaviour is **torch-version-sensitive** (the study saw cold/warm
state-dependence on torch 2.12.1). State the torch version the MPS rows were
produced on. The host float64 acceptance gate is the version-independent guarantee
that a wrong call fails loud rather than returning silently wrong — that property
holds regardless of torch version; the *solver convergence envelope* is what is
validated per-version.

## Affected reports

- `reports/survival-v4.2.4.md` (new; from `survival-v4.2.3.md`)
- `reports/regression-v4.2.4.md` (new; from `regression-v4.2.3.md`)
