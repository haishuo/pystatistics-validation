# VA-4 (+VA-4b) — timeseries `xreg` + drift + `fixed=` — implementation + validation status

**Status: implemented, unit-tested, and R-validated against the local build. Canonical
bless is a post-publish step (see below).** This is one of the three converging Tier-A chips;
an external system collates the version once all three land. Target library version **4.8.0**.

## What shipped in the library (`Dev/pystatistics`, working tree — NOT yet published)

Regression with ARIMA errors, matching `stats::arima` / `forecast::Arima` / `forecast::auto.arima`:

- `arima(y, order, xreg=X)` — fits `y = Xβ + η`, η an ARIMA process. Coefficients on the
  solution as `xreg_coef` / `xreg_names` (`intercept` / `drift` / `xreg1..xregk`), joint SEs in
  `vcov`. Method: diff both `y` and the design identically (differencing commutes with the linear
  regression), then β sits in the optimizer vector beside the (factored) ARMA params — exactly R's
  `armafn`. New module `pystatistics/timeseries/_arima_xreg.py`.
- `include_drift=True` — a `1..n` time-trend regressor (R's "with drift"); reproduces R's
  drift/`d` interaction (ARIMA(p,1,q)+drift = ARMA-with-mean on the differenced series). Fails
  loud when `d + D >= 2` (trend unidentifiable).
- `fixed=` — hold coefficients constant. Pythonic `{name: value}` dict (primary) or R's positional
  nan-vector. Fixed coefs carry zero variance and are excluded from the IC free-parameter count.
- `forecast_arima(..., newxreg=)` — forecasts `u = y − Xβ` then adds the future regression mean;
  drift/intercept future columns synthesized; `newxreg` required (and shape-checked) for user xreg.
- `auto_arima(..., allowdrift=True)` (default) — selects drift when `d + D == 1` and drift lowers
  the IC, like `forecast::auto.arima`. Seasonal/`d+D!=1` selections are byte-identical to before
  (each visited order is fit with/without drift only when `d+D==1`).
- The regression path is **CPU-only** (the Whittle/`arima_batch` GPU kernels carry no regression
  term); `xreg`/`include_drift`/`fixed` fail loud with `method='Whittle'` or a GPU backend.
- Refactor: the numerical Hessian moved to `_arima_likelihood.compute_numerical_hessian` (keeps
  `_arima_fit.py` under the 500-LoC cap). The plain (no-regressor) ARIMA path is byte-unchanged.

Tests: `tests/timeseries/test_arima_xreg.py` (40, Rule 7) against
`tests/fixtures/arima_xreg_r_reference.json` (R 4.5.2; regen with
`generate_arima_xreg_r_reference.R`). Release log in `.release/UNRELEASED.md`. Full library suite
green after the change.

## R-agreement measured (local build vs `stats::arima` + `predict.Arima`)

coefficients ~5e-6, log-likelihood ~1e-9, AIC/sigma2 machine-precision, SE ~4e-5; forecast point
~1e-6, forecast SE ~3e-7. R10 hard cases all pass: xreg under d=1, near-collinear xreg, all-but-one
fixed, drift under d=1, seasonal+xreg. Speed: py matches/beats R for n>=1000 (0.029s vs 0.043s at
n=5000); slightly slower only at n=200 (fixed Hessian/two-stage overhead — the module's documented
small-n constant factor, no complexity-class gap).

**Convention (documented):** prediction intervals use the MLE `sigma2 = SSR/n` (matches
`stats::arima`/`predict.Arima`, and the rest of the module). `forecast::Arima` reports a
df-adjusted `sigma2 = SSR/(n-ncoef)` → slightly wider intervals. Point forecasts and coefficients
are identical to both R paths.

## Validation repo — what's ready now

- `drivers/timeseries/run_xreg.py` — harness-conforming driver (require_pypi + `r_reference` +
  `build_run`/`write_run`). Emits `artifacts/timeseries/v<ver>/runs/xreg.json`. Its `run_fits` /
  `run_forecasts` / `run_auto` / `run_fidelity` were verified against the **local build** (all
  green: fits 8/8, forecasts 3/3, auto 2/2, fidelity 8/8); `main()` is `require_pypi`-gated so it
  only writes canonical evidence from the released package.
- `drivers/timeseries/r_reference.R` — added `arima_xreg` / `forecast_arima_xreg` handlers (xreg as
  a JSON list-of-rows; `fixed` as a nan→null vector; drift as an appended `1..n` column).
- `drivers/timeseries/build_manifest.py` — added the `xreg` flattener (`g1_xreg_summary.csv` +
  `g1_xreg_aux_summary.csv`) and two study entries; provenance note updated with A9 + VA-4.
- `subsystems/timeseries/meta.json` — ARIMA algorithm prose extended; new `g1_xreg_two_tier`
  tolerance; two new limitations (the MLE-vs-df sigma2/PI convention; the CPU-only regression path).

## Post-publish canonical bless (do after the user authorizes the 4.8.0 release)

R5 forbids canonical evidence from a local build (`require_pypi`), and the GPU rows are on Forge, so
the frozen v4.8.0 report is a post-publish step — same pattern as A0:

1. `pip install pystatistics==4.8.0` in the throwaway PyPI venv.
2. Re-run the CPU drivers at 4.8.0: `run_deterministic.py`, `run_mle.py`, **`run_xreg.py` (new)**,
   `run_fidelity.py`, `run_performance.py`, `run_batch_contract.py`.
3. Re-run the GPU rows on Forge (CUDA): `generate_gpu_cuda.py` (arima_batch is UNCHANGED by VA-4, so
   this reproduces 4.6.6's numbers — carry-forward is legitimate if Forge is unavailable, but a
   fresh run is cleaner).
4. `python drivers/timeseries/build_manifest.py 4.8.0` → manifest + summary CSVs (now including the
   two `g1_xreg*` studies; provenance already reflects A9 + VA-4).
5. `python render_report.py timeseries 4.8.0` → `reports/timeseries-v4.8.0.md`.
6. **A9 is closed by this render**: the 4.8.0 report reflects the 4.6.12 arima GPU fail-loud fix
   (the reason the 4.6.6 report was code-stale) plus VA-4. Bless timeseries at 4.8.0.

## Coordination

Ship VA-4 as part of the single additive 4.8.0 minor alongside the other two Tier-A chips (survival
cluster, gam tensor smooths). Do NOT publish without the user's explicit go (OPERATIONS.md; flow in
`.release/CHECKLIST.md`).
