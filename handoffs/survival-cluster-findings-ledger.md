# A1+VA-8 survival cluster — findings ledger

Findings from the pre-release passes on the survival feature cluster (stratified
Cox/KM, counting-process Cox, left-truncation KM, cox.zph, robust/cluster SE).
Sources: R-validation dry-run (R1 timing), adversarial correctness review
(14 agents, 9 confirmed findings). Bless precondition (RIGOR R18): every fixable
finding fixed in the version the report blesses, or a true documented limitation.

## Fixed (in the 4.8.0 working tree, pre-release)

| ID | Sev | Finding | Fix |
|----|-----|---------|-----|
| P1 | R1 defect | Cox ~5-11x slower than R on tied data (Efron per-tie-group Python loop + per-row Python concordance Fenwick). | numba concordance kernel + vectorized Efron correction → R parity-or-faster (0.78-1.18x). Bit-identical. |
| F1 | MAJOR | Uncentered covariates → numerically indefinite information → `se=0`/`p=1` (silent-wrong) on large-magnitude covariates; R centers and is correct. | Mean-center covariates in every fit/robust/zph path (shift-invariant); now matches R exactly. |
| F4 | MAJOR | NaN/None stratum label silently dropped rows while `n_observations` still counted them. | Fail loud on missing strata labels at the `SurvivalDesign` boundary. |
| F5 | MAJOR | `median_survival` didn't apply R survfit's `minmin` averaging when S touches exactly 0.5. | Replicated `minmin`; matches R on touch/cross/never cases. |
| F6 | MAJOR | Non-finite `time` slipped past the `< 0` guard (NaN/inf both False). | Reject non-finite time at the boundary. |
| F7 | MAJOR | NaN `event` was masked out of the 0/1 check but retained in the array. | Reject non-finite / non-0/1 event at the boundary. |
| F0 | MINOR | Zero-event Cox fit returned `converged=True`, `concordance=0.5` (fabricated). | `converged=False`, `concordance=NaN` (matches the discrete-time twin and R's NA). |
| F2 | MINOR | Harrell's C returned a hardcoded 0.5 when there were no comparable pairs. | Return `NaN` (as R does). |

All fixes carry regression tests (`tests/survival/test_survival_hardening.py`,
plus the machine-precision R-validation across the cluster). Full library suite
green.

## Deferred (documented limitations — not blocking, revisit at bless)

| ID | Sev | Finding | Disposition |
|----|-----|---------|-------------|
| F3 | MINOR | `KMParams.n_censored` omits censoring tied to an event time and after the last event, so it doesn't equal R's `survfit` `n.censor` per interval. | Pre-existing; a display-only field, not used in S(t)/variance/CI and not in the R-validated quantity set (time/survival/n_risk/se/ci all match R). To fix at bless if we choose to expose an R-exact `n.censor`, else document as a deliberate n_censored convention. |
| F8 | MINOR | `coxph(robust=True)` on an all-censored (zero-event) fit silently reports `.robust=False`. | The whole fit is degenerate and already flagged `converged=False`; a sandwich on zero events is undefined. Acceptable given the degeneracy signal; revisit if we want an explicit robust-on-degenerate error. |

## Perf note for R1/R2 (report)

CPU-vs-R stratified/plain Cox across n=8k-50k, Efron + Breslow: py/R 0.78-0.99x
(faster than R) on the synthetic grid; flchain (n=7874) efron ~1.18x, breslow
~1.08x — the residual is the O(n·p²) risk-set covariance tensor rebuilt each
Newton step, which R computes in C; pystatistics overtakes R by n≈20k. Document
per R2 (name the extra work, it is small and vanishes at scale).
