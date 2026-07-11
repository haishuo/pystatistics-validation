# A7 — montecarlo BCa acceleration on non-ordinary bootstrap: measurement + decision

**Question (from the close-out brief):** on balanced / parametric / stratified
bootstrap, pystatistics' BCa falls back to Efron's delete-1 **jackknife**
acceleration instead of R `boot.ci`'s default **regression `empinf`** (`type="reg"`).
Measure the tail divergence; implement `empinf` if material, else document as B.

## Method
The acceleration `a = Σ L³ / (6 (Σ L²)^{3/2})` is a property of the statistic at the
data; the jackknife and regression estimates of the empirical influence `L` differ,
so BCa endpoints differ. Measured the endpoint difference between the two accelerations
on **identical** bootstrap replicates in R (`boot` + `boot.ci(..., L=)`), isolating the
acceleration-method effect from Monte-Carlo noise. Statistics of increasing non-linearity
(mean, variance, skewness, a skewed ratio `mean/(var+0.1)`); sample sizes n=40 and n=15.

## Result
| regime | statistic | endpoint shift (jackknife vs reg `empinf`), % of CI width |
|---|---|---|
| n=40 | mean | 0.00% (identical) |
| n=40 | variance | ~0.1% |
| n=40 | skewness | ~0.2% |
| n=15 | variance | <1% |
| **n=15** | **skewed ratio** | **up to 7.35%** (typical 2–7% on the upper tail) |

- **Smooth / near-linear statistics:** the two accelerations agree to well within
  Monte-Carlo noise — the jackknife fallback is fine.
- **Strongly non-linear statistics at small n:** the jackknife (a linear delete-1
  approximation of the influence) diverges from the regression estimate enough to move
  a BCa **tail** endpoint by several percent of the CI width — **material**. This is
  precisely why `boot.ci` defaults to `type="reg"`.

## Decision
- **Balanced & stratified bootstrap → IMPLEMENT regression `empinf`. DONE (4.7.0).**
  `regression_influence` now regenerates the balanced/stratified resample frequencies
  from the seed (mirroring the CPU backend's index generation), computes the regression
  influence centred within strata, and self-checks by reproducing replicate 0. A
  truncating `rcond=1e-8` in the pinv solve removes the degenerate-frequency null space
  cleanly (the ordinary path's acceleration is unchanged to ~1e-5). Validated: the
  acceleration matches R's `empinf(type="reg")` on the same data for
  ordinary/balanced/stratified to Monte-Carlo tolerance (Δa ≤ 7e-4).
- **Parametric bootstrap → documented B.** A parametric bootstrap regenerates data from a
  fitted model, so there are **no resample frequencies** and the regression estimate does
  not apply; the delete-1 jackknife on the original data is the principled acceleration.
  R's `boot.ci` faces the identical limitation. Justification to record in the montecarlo
  report / CONVENTIONS capability notes.

Scripts: `scratchpad/a7_R.R`, `scratchpad/a7_stress.R`.
