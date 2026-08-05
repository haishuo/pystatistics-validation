# `regression` validation drivers

Generators that produce the frozen evidence behind `reports/regression-v<X.Y.Z>.md`.
They fit a **PyPI-installed** `pystatistics` (never a local checkout — `require_pypi`
enforces it) and the R reference on the shared `pystatsval` harness.

## Layout

| file | job |
|---|---|
| `datasets.py` | curate the canonical designs: California Housing (OLS/binomial/Poisson), `datasets::airquality` (Gamma/log), `MASS::quine` (negative binomial). Deterministic; intercept column included. |
| `run_pystatistics.py` | fit + time one `fit(design, family=…, backend=…)` → a `validation-run/v1` record with the full estimate vectors. |
| `run_r_regression.py` + `_r/regression_run.R` | fit the R reference (`lm`/`glm`/`glm.nb`), timing inside R, → a comparable record. |
| `_r/prep_datasets.R` | emit the airquality + quine designs from R; run by the store generator into the central HDF5 store (`airquality.h5`, `quine.h5`). The driver loads them from the store via `DATASETS_ROOT` — no committed CSV (R17). |
| `generate_correctness.py` | OLS + 5 GLM families, pystatistics vs R, on real data → `correctness_<device>_<host>.{json,csv}`. |
| `generate_scaling.py` | synthetic (n × p) sweep per device → `scaling_<device>_<host>.{json,csv}` for the device pivots. |
| `generate_hardcases.py` + `_r/hardcase_run.R` | RIGOR R10 adversarial grid: collinear/GPU-refusal boundary, logistic separation, factor/contrast coding, weights/offset, rank-deficient — match R's *behaviour* (failures + warnings), not just coefficients → `hardcases_<host>.json` + `hardcases_behavior_*`/`hardcases_collinear_*` CSVs. |
| `generate_precision_isolation.py` | RIGOR R11: same-precision `gpu_fp64` vs `cpu_fp64` (CUDA) alongside the bundled fp32 numbers → `precision_isolation_cuda_<host>.{json,csv}`. |
| `_r/blas_info.R` | RIGOR R11: record the BLAS/LAPACK R linked against per host → `blas_<host>.json` (combined into `blas_provenance_summary.csv`). |
| `generate_fp32_boundary.py` | RIGOR R12: adversarial no-silent-wrong proof for the relaxed fp32 GPU GLM — straddle the float32 floor on MPS + CUDA, classify each accept/refuse against the fp64 optimum → `fp32_boundary_<device>_<host>.{json,csv}`. |

## Reproduce

Install the library from PyPI (plus the harness + driver deps) into a clean env:

```bash
pip install pystatistics==3.18.0          # the version under validation
pip install -e ../pystatistics/validation # pystatsval harness (not in the wheel)
pip install pandas torch                   # driver CSV IO; torch for GPU rows
```

Then, from the validation repo root (`PYTHONPATH=.`):

```bash
# Correctness vs R (run on every host — Mac and Linux):
python -m drivers.regression.generate_correctness --host powerhouse --device cpu
# Device scaling (one invocation per device):
python -m drivers.regression.generate_scaling --host powerhouse --device cpu
python -m drivers.regression.generate_scaling --host powerhouse --device mps
python -m drivers.regression.generate_scaling --host forge      --device cuda
```

On macOS set `KMP_DUPLICATE_LIB_OK=TRUE` (libomp double-init). CUDA rows run on
Forge under the standing CUDA-testing allowance (clone the GPU env, PyPI-install,
run, tear down — never disturb the editable worktree).
