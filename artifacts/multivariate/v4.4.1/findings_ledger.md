# Multivariate validation — findings ledger (v4.4.1 bundle)

The R18 bundle that closes the three factor-analysis findings gathered during the
4.4.0 first-time validation (see `../v4.4.0/findings_ledger.md`). F1 and F2 were
fixed in the library (proposed by the validation, implemented under explicit
authorization, published as 4.4.1) and are **CLEARED** here; F3 is a documented
minor R2 cost, unchanged.

---

## F1 — varimax absolute convergence test → CLEARED at 4.4.1

- **Was:** the default `factor_analysis(X, n_factors>=2)` (rotation=varimax)
  raised `ConvergenceError` on clean, well-fitting data — an ABSOLUTE stop test
  `abs(d_new-d_old)<1e-6` in `_rotation.varimax`, vs R `stats::varimax`'s RELATIVE
  test `d < dpast*(1+tol)`.
- **Fix (4.4.1):** `_rotation.varimax` adopts R's relative convergence criterion.
- **Verified:** the multi-factor cases that raised at 4.4.0 now CONVERGE and match
  factanal — synth2f (2-factor) and synth3f (3-factor): `py_converged=True`,
  loadings agree to <=1.0e-5 (up to column permutation + sign), uniquenesses to
  <=2.5e-5, ML objective to <=3.2e-8. Evidence:
  `runs/correctness_fa_powerhouse_summary.csv`.
- **Status:** CLEARED.

## F2 — Heywood uniqueness floor differed from R → CLEARED at 4.4.1

- **Was:** `factor_analysis` let uniquenesses fall to ~0 (Heywood case) while R
  `factanal` floors them at its default `lower=0.005`; on iris 1-factor
  pystatistics landed at a different, lower-objective constrained optimum
  (uniqueness gap ~5e-3, objective gap ~3%).
- **Fix (4.4.1):** `factor_analysis` gains a `lower` parameter (default 0.005,
  matching factanal) bounding the uniquenesses.
- **Verified:** iris 1-factor Petal.Length uniqueness now floors at exactly
  0.005; the full uniqueness vector agrees with factanal to ~1.5e-6 (was ~5e-3)
  and the objective to ~8e-13.
- **Status:** CLEARED.

## F3 — small-n FA iteration count → documented R2 cost (unchanged)

- The ML L-BFGS optimiser takes ~38 iterations at n=500 vs ~10 at larger n
  (noisier small-sample correlation → flatter likelihood), so FA CPU speed dips
  to ~0.65x R factanal at n=500 before leading 1.7-3.3x at n>=2000. A
  data-dependent iteration count, not a per-iteration or complexity-class deficit
  (R's sub-5ms times are timer-quantised). Left as a documented R2 cost; not a
  defect, correctness unaffected.
- **Status:** documented (R2), unchanged at 4.4.1.

---

## Disposition

Factor analysis is now VALIDATED at 4.4.1 (the deferred multi-factor claim is
substantiated). PCA is unchanged from 4.4.0 (the fix touched only `_rotation.py`
and `_factor.py`) and re-confirmed at 4.4.1. CARRY-FORWARD CF-1 remains CLEARED
(the CUDA gate boundary reproduces silent_wrong_count=0 at 4.4.1). The blessed
report is `reports/multivariate-v4.4.1.md`; `reports/multivariate-v4.4.0.md` is
the superseded first-time validation kept as the historical record.
