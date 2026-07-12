# VA-2 (by=) / VA-3 (nb) gam validation — verdict

**VA-2 CLOSED-BLESSABLE · VA-3 CLOSED-BLESSABLE — CLEAN at `pystatistics==4.8.1` (both 4.8.0 findings RESOLVED and re-validated).** Continuous by=, factor by=, and nb dispersion θ all match mgcv 1.9.3 / R 4.5.2; the fail-loud guards fire. No open findings. The whole-library bless gate is cleared.

> **Update 2026-07-12 — findings resolved, 4.8.1 released.** The two findings below were fixed in the library and shipped as **`pystatistics==4.8.1`** (on PyPI), then re-validated at that PyPI artifact:
> - **VA3-F1 RESOLVED:** `GAMSolution.theta` / `GAMParams.theta` now expose the estimated nb dispersion; θ read from the accessor matches mgcv `getTheta` to ≤~1.3e-5 (study `va3_nb`, `theta_accessor=yes`).
> - **VA2-F1 RESOLVED:** `s(x, by=g, by_type='factor')` now fits per-level smooths matching mgcv `s(x, by=factor(g))+factor(g)` — identical coefficient count and per-level EDF, total EDF/deviance/fitted to the optimizer tier (~1e-5..1e-7) on tp and cr (study `va2_factor_by`); a categorical-looking by-column with `by_type` unset now **fails loud** (study `va2_va3_fidelity`).
> The full gam validation set (correctness, tensor, by/nb, performance) was re-run at 4.8.1 and shows **no regression** from the by/nb fix. Evidence: `artifacts/gam/v4.8.1/`, report `reports/gam-v4.8.1.md`. The 4.8.0 evidence below is retained as the honest snapshot of the version that shipped with the gaps.

---

## Original verdict (at 4.8.0)

**VA-2 CLOSED-BLESSABLE (continuous by=, mgcv-exact) · VA-3 CLOSED-BLESSABLE (nb estimator, mgcv-exact) — with two OPEN fail-loud/completeness findings (VA2-F1, VA3-F1) that did NOT block the cut but should be fixed: neither is a silently-wrong estimator on the shipped surface.**

Both surfaces that shipped in 4.7.0 are numerically correct against mgcv 1.9.3 / R 4.5.2 at the PyPI artifact `pystatistics==4.8.0`. There is **no showstopper** (no silently-wrong estimator). The two API/fidelity gaps below were cheaply fixable — VA2-F1 is the factor-by case the brief flagged as a potential R16 showstopper; it was a *missing guard on an out-of-scope input*, not a defect in the continuous surface. (Both are now fixed at 4.8.1 — see the update above.)

---

## Evidence (artifacts/gam/v4.8.0/runs/by_nb.json; report reports/gam-v4.8.0.md §3 studies va2_by_tier1 / va2_by_tier2 / va3_nb / va2_va3_fidelity)

Throwaway venv, `install_source=pypi`, pystatistics 4.8.0, numpy 2.4.6, R 4.5.2 / mgcv 1.9.3. Data generated in R, both engines read identical float64 bytes. Driver: `drivers/gam/run_by_nb.py`.

### VA-2 `s(x, by=z)` continuous varying-coefficient — EXACT (blessable)

- **TIER 1 (mgcv's selected sp fed to both engines):** the by-multiplied cr basis + penalty is a single stable-QR solve → fitted / coef / total EDF / scale / posterior SE agree to **fp64 arithmetic (≤~1e-12)**. Holds for a by-variable with **negative values + a zero-variance stretch** (`by_neg`) and a **large-magnitude by-column z~N(1000,200)** (`by_big`, fitted ~1.1e-12).
- **TIER 2 (free REML):** the selected sp is **directly comparable** to mgcv's — `sp_rel_max ≤ ~1e-6` on every case, *including `by_big`* — so there is **no sp-scale reparameterisation** for by= (unlike tp/ti). REML score bit-identical; total/per-smooth EDF, deviance, fitted to the optimizer tier (≤~1e-6). `by_two` exercises `s(x,by=z)+s(w)` (two penalties) and matches.
- **The brief's "py lambda 4.75 vs mgcv sp 0.0123" did NOT reproduce** under any controlled continuous-by design (positive, negative, or ×1000 magnitude). It was a measurement artifact of the earlier ad-hoc probe, not an estimator/parametrization gap. The sp is on mgcv's scale.

### VA-3 `family='nb'` negative-binomial dispersion theta — EXACT (blessable)

- Estimated **theta agrees with mgcv `getTheta(TRUE)` to ≤~1.3e-5 relative** across three R10 regimes: moderate θ≈3 (py 3.05317 vs 3.05317), **large θ near the Poisson limit** (py 25.837 vs 25.836), and **small counts / low mean** (py 3.187 vs 3.187). Total & per-smooth EDF ≤~1e-6, deviance/REML/fitted to the optimizer tier.
- θ is estimated by **profiling the Laplace-REML score over log θ** (mgcv's `nb()` convention), not MASS's plain-ML θ. `family='nb', method='GCV'` **correctly fails loud** (profiled UBRE degenerate in θ).

---

## Open findings (reported to the library repo — NOT fixed here; this session is read-only on `pystatistics`)

### VA2-F1 — factor-by is SILENTLY misinterpreted as a continuous by-variable (fail-loud gap)

`s(x, by=z)` multiplies the basis rows by the numeric column `z`. There is **no factor/categorical by-variable path and no guard**. Handed a 3-level factor coded `{0,1,2}`, pystatistics **silently fits one varying-coefficient smooth** (`total_edf 5.06`, 11 coef, `converged=True`, **no warning**) where mgcv's `s(x, by=factor(g))` builds **3 per-level smooths** (30 coef, edf 18.16). A user porting the common mgcv idiom `s(x, by=group)` gets a meaningless model with no signal.

- **Why not a showstopper:** the *continuous* by= surface that shipped is mgcv-exact; this is a missing fail-loud guard (or an unimplemented feature), not a wrong estimator. But it is a silent misinterpretation of a common input, so per the bless bar it **should be fixed, not documented away**.
- **Recommended library fix:** fail loud when the by-column is low-cardinality/categorical (e.g. reject an integer column with ≤ small number of distinct values, or require an explicit `by_type`), OR implement per-level factor-by. A few lines in `pystatistics/gam/_smooth.py` / `_basis.py`.

### VA3-F1 — estimated nb theta has no public accessor (completeness gap)

`family='nb'` estimates θ internally and it drives every reported quantity, but the fitted `GAMSolution`/`GAMParams` expose it **nowhere** — `family_name` is the bare string `'negative.binomial'` and no attribute holds θ. This validation had to **recover θ by inverting the NB deviance** at the fitted mean (strictly monotone → unique) to compare against mgcv. The defining output of an NB model is unreadable through the API.

- **Why not a showstopper:** not silently-wrong — the fit and θ are correct. It is an incompleteness vs the reference (mgcv exposes `getTheta`).
- **Recommended library fix:** add a `theta` property to `GAMParams` populated from the fitted family (the `_fit_nb_outer` path already has it — it is discarded when `family_name=fam.name` is set in `_gam.py`).

---

## What was covered (R10 hard cases)

by= with negative values + zero-variance stretch (`by_neg`); by= alongside a second smooth (`by_two`); large-magnitude by-column (`by_big`, sp-scale probe); factor-by fidelity probe (`factor_by` → VA2-F1); nb moderate/large-θ-near-Poisson/small-counts (`nb_default`/`nb_bigtheta`/`nb_small`); nb+GCV fail-loud guard (passes). Two-tier fixed-sp cross-feed established (not assumed) that by= sp is on mgcv's scale.

## Deliverables produced

- Driver `drivers/gam/run_by_nb.py` (self-contained R worker; PyPI-only, CPU-only).
- Frozen artifact `artifacts/gam/v4.8.0/runs/by_nb.json` + 4 summary CSVs; `build_manifest.py` extended (4 new studies); `manifest.json` re-emitted (11 studies).
- Report `reports/gam-v4.8.0.md` re-rendered (VA-2/VA-3 tables in §3, findings VA2-F1/VA3-F1 in §6 + audit block). Render test suite: 8/8 pass.
- `subsystems/gam/meta.json` updated: procedure, algorithms (by=/nb bullet), tolerances (`by_continuous_va2`, `nb_negative_binomial_va3`), limitations + audit findings VA2-F1 / VA3-F1.

_Validated 2026-07-12 · pystatistics 4.8.0 (PyPI) · mgcv 1.9.3 / R 4.5.2 · CPU._
