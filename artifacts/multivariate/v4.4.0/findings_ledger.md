# Multivariate validation — gathered findings ledger (v4.4.0 run)

R18 ledger for non-showstopper findings surfaced during the 4.4.0 multivariate
validation. None is a showstopper (none is silently-wrong): per R18 they are
**gathered** here and scheduled into a bundled **4.4.1** patch, after which the
affected surface (factor analysis) is re-validated. The PCA validation pass
continues on 4.4.0 unaffected.

Both fixes live in the sibling library `../pystatistics` and were **proposed, not
applied** (Coding Bible Rule 8). They are not yet in `.release/UNRELEASED.md`.

---

## F1 — varimax rotation: absolute convergence test → fails loud on clean data

- **Symptom:** the DEFAULT call `factor_analysis(X, n_factors>=2)` (rotation
  defaults to `'varimax'`) raises `ConvergenceError("Varimax rotation did not
  converge after 1000 iterations")` on a clean, well-fitting 2-factor model
  (n=400, p=8, simple structure, uniquenesses ~0.3–0.5) where R's
  `factanal(..., rotation="varimax")` converges immediately.
- **Root cause:** `pystatistics/multivariate/_rotation.py:67` tests an **absolute**
  change `abs(d_new - d_old) < tol` (tol=1e-6) on `d = sum(singular values)`,
  whose scale depends on the loadings. R's `stats::varimax` uses a **relative**
  test `if (d < dpast * (1 + tol)) break` (R default tol=1e-5). Observed
  increments plateau at ~8.5e-3 decaying ~1%/iter — never reaching 1e-6 abs in
  1000 iters. The FA ML core is sound (`rotation='none'` converges with correct
  uniquenesses); only the rotation stop-criterion is wrong. This is the R9
  false-negative convergence pattern, on the CPU rotation.
- **Severity (R18):** non-showstopper — FAILS LOUD (no wrong number returned).
  But it breaks the advertised default multi-factor FA path on textbook data
  (R15), so it blocks the FA-vs-factanal multi-factor correctness claim until
  fixed.
- **Proposed fix:** change `_rotation.varimax` to R's relative convergence test
  (`d < dpast*(1+tol)`, tol≈1e-5), matching `stats::varimax`. Re-validate the
  varimax loadings against R after the fix.
- **Status:** OPEN → 4.4.1 bundle.

## F2 — factor_analysis: Heywood uniqueness floor differs from R (no `lower=`)

- **Symptom:** on iris 1-factor, pystatistics lets Petal.Length's uniqueness go
  to **0.0** (a Heywood case) and lands at a lower objective (0.566 vs R 0.585,
  chi2 82.76 vs 85.51); R's `factanal` floors uniquenesses at its default
  `lower=0.005`. Uniqueness gap ~5e-3, objective gap ~3%, loadings gap ~6e-3.
- **Root cause:** `_factor.factor_analysis` clamps `psi` to ~0 lower bound
  (`np.clip(psi, 1e-10, 1-1e-10)`) and exposes no `lower=` parameter; R's
  `factanal` constrains uniquenesses to `>= lower` (default 0.005). Different
  constrained optima on Heywood-prone models. Not silently wrong — pystatistics
  reports a valid (less-constrained) optimum — but it diverges from R and admits
  degenerate Heywood solutions R guards against.
- **Severity (R18):** non-showstopper — different-but-defensible constrained
  optimum, not a wrong number. A convention difference (R10): match R or
  document deliberately.
- **Proposed fix:** add a `lower=` floor (default 0.005, matching R) to
  `factor_analysis`; clamp uniquenesses to `>= lower` during/after optimisation.
  Or document the Heywood difference as deliberate with a defensible reason.
- **Status:** OPEN → 4.4.1 bundle.

## F3 — factor_analysis small-n iteration count (minor R2 perf)

- **Symptom:** in the CPU-vs-R speed sweep, FA at n=500 (p=12) is ~0.53x R
  (py ~3.4ms vs R ~2ms), the only size where pystatistics FA lags. pystatistics
  wins 1.7–3.3x at n>=2000.
- **Root cause:** the ML L-BFGS optimiser takes ~38 iterations at n=500 vs ~10 at
  n=2000 — the small-sample correlation matrix gives a flatter/noisier likelihood,
  so more iterations to reach tol. Per-iteration cost and complexity are
  competitive; this is a data-dependent iteration count, not a complexity gap.
  R's sub-5ms times are also timer-quantized (2.00/2.00/3.00/8.00 ms), so the
  small-n comparison is imprecise on R's side.
- **Severity (R18):** non-showstopper, minor — an explained R2 constant-factor at
  one tiny size, correct result. Candidate for the 4.4.1 FA work (a better
  starting value to cut small-n iterations) but not required.
- **Status:** OPEN (minor) → 4.4.1 bundle if cheap; else documented (R2).

---

## Disposition

- **PCA (priority 1):** unaffected — machine-precision agreement with `prcomp`
  (sdev 1.2e-15, rotation 6e-16, scores 1.2e-13 on iris). The 4.4.0 PCA pass
  proceeds: R10 hard cases, R15 defaults, CPU-vs-R speed, Mac MPS value, Forge
  CUDA + CF-1.
- **FA (priority 1):** the 1-factor iris case is recorded (it exercises F2, not
  varimax); the multi-factor FA correctness claim is DEFERRED to the 4.4.1
  re-validation once F1+F2 land. The 4.4.0 report will state FA's status honestly
  and point here.
