# `survival` validation drivers

Generators that produce the frozen evidence behind `reports/survival-v<X.Y.Z>.md`.
They fit a **PyPI-installed** `pystatistics` (never a local checkout — `require_pypi`
enforces it) and the R `survival` reference on the shared `pystatsval` harness.

## Layout

| file | job |
|---|---|
| `datasets.py` | curate the canonical designs from `survival::lung`: `lung_km` (time, event, sex; KM + log-rank) and `lung_coxph` (time, event, age, sex, ph.ecog; Cox + discrete-time). Reads the committed CSVs; `discrete_interval_bounds()` makes coarse, well-posed bins for discrete-time. |
| `run_pystatistics.py` | fit + time `kaplan_meier` / `survdiff` / `coxph` / `discrete_time` → `validation-run/v1` records with the full estimate vectors. |
| `run_r_survival.py` + `_r/survival_run.R` | fit the R reference (`survfit` / `survdiff` / `coxph`, and `glm(binomial)` for discrete-time), timing inside R, → comparable records. |
| `_r/prep_lung.R` | emit the lung KM + Cox designs from `survival::lung`; run by the store generator into the central HDF5 store (`lung_km.h5`, `lung_coxph.h5`). The driver loads them from the store via `MVNMLE_DATA_DIR` — no committed CSV (R17). |
| `_person_period.py` | reconstruct the discrete-time person-period design, mirroring `pystatistics/survival/_discrete.py`, so the R `glm` reference fits the identical matrix (guarded by a `person_period_n` check). |
| `agreement.py` | reduce a pystatistics-vs-R pair to per-quantity agreement rows (max abs / max rel). |
| `generate_correctness.py` | KM + log-rank + Cox + discrete-time, pystatistics vs R, on lung → `correctness_cpu_<host>.{json,_agreement.csv,_timing.csv}`. |
| `generate_scaling.py` | synthetic proportional-hazards sweep, Cox PH wall vs R across n → `scaling_cpu_<host>.{json,_summary.csv}` (exposes the O(n²) concordance + risk-set hotspots). |

## Reproduce

Install the library from PyPI (plus the harness + driver deps) into a clean env:

```bash
pip install pystatistics==4.0.0           # the version under validation
pip install -e ../pystatistics/validation # pystatsval harness (not in the wheel)
pip install pandas numpy scipy
```

Then, from the validation repo root (`PYTHONPATH=.`):

```bash
# Correctness vs R + per-procedure timing (run on each host):
python -m drivers.survival.generate_correctness --host powerhouse
# Cox PH CPU scaling vs R (the performance / optimization pass):
python -m drivers.survival.generate_scaling --host powerhouse
```

On macOS set `KMP_DUPLICATE_LIB_OK=TRUE` (libomp double-init). Requires R with the
`survival` package on `PATH` (`Rscript`).

## Notes

- **CPU only.** Per the PyStatistics Constitution, Cox PH is a deliberate no-GPU
  module (the partial likelihood is inherently sequential); KM and log-rank are
  cheap one-pass reductions. Only `discrete_time` has a GPU path, inherited from
  the regression binomial GLM it delegates to. There is no GPU device-pivot for
  survival, and that is the correct result — see the report's reconciliation
  section.
- **Discrete-time** is validated against R `glm(binomial)` on the *identical*
  person-period design (reconstructed in `_person_period.py`). The generator
  asserts the reconstruction's row count equals the library's `person_period_n`
  and fails loud otherwise, so a future change to the library's expansion can't
  silently invalidate the head-to-head.
