# pystatistics-validation

Cross-platform validation suite for [pystatistics](https://pypi.org/project/pystatistics/) and [pystatsbio](https://pypi.org/project/pystatsbio/) on Linux/NVIDIA.

Installs both libraries **from PyPI** into a fresh conda environment and validates correctness against R, GPU acceleration performance, and numerical stability on real-world datasets.

## Why this exists

All development and initial testing of pystatistics and pystatsbio was done on macOS (Apple Silicon). This suite verifies that:

1. **Correctness** -- results match R on Linux, not just Mac
2. **Speed** -- CUDA GPU acceleration delivers meaningful speedup on NVIDIA hardware
3. **Stability** -- no silent numerical blow-ups; fail fast, fail loud

## Test suites

### Suite 1: PyStatistics vs R

Every row of the table below has a WHAT (which pystatistics call is being
exercised), a HOW (what the R reference fits, and what we compare), a DATASET
(the specific file the test reads), and a WHY (why that dataset — not some
other — is the right one for this test). Tests also record Python-vs-R
runtime and fail if Python is more than 20× slower than R (for fits where R
takes ≥ 50 ms — below that, wall-clock noise dominates).

**Core suite — California Housing** (20,640 rows from StatLib):

| WHAT (pystatistics) | HOW (R ref) | DATASET | WHY that dataset |
|---|---|---|---|
| OLS `fit(X, y)` | `lm()` — coefs, SEs, R², residuals (rtol=1e-10) | `california_prepared.csv` | Large, real, non-pathological dataset with both continuous and categorical predictors |
| GLM binomial | `glm(family=binomial)` — coefs, deviance | `california_prepared.csv` (binarized target) | Gives a real binary outcome (`high_value`) with real predictors |
| GLM Poisson | `glm(family=poisson)` — coefs, deviance | `california_prepared.csv` (population/100) | Turning population into counts gives a realistic Poisson response |
| Descriptive stats | `colMeans`, `sd`, `cor`, `quantile`, `moments::skewness/kurtosis` | `california_prepared.csv` | Non-normal columns (right-skewed MedInc, bounded Latitude) stress skewness/quantiles |
| Hypothesis tests | `t.test`, `chisq.test`, `wilcox.test`, `prop.test`, `ks.test` | `california_prepared.csv` | Real contingency tables (region × high_value); real skewed distributions for Wilcoxon/KS |
| ANOVA | `aov`, `car::Anova(type=2)`, `TukeyHSD`, `leveneTest` | `california_prepared.csv` (region, old_house factors) | Gives real factorial design with unequal cell sizes — exercises Type II SS |
| Kaplan-Meier | `survfit(Surv())` | `california_prepared.csv` (HouseAge as proxy time, high_value as event) | KM works even when the event isn't a true "death"; the curve math is dataset-agnostic |
| Log-rank test | `survdiff()` | Same proxy survival | Proxy is fine for comparing two *curves*, which is what log-rank does |
| **Cox PH** | `coxph(Surv(time, event) ~ age + sex + ph.ecog)` — coefs, hazard ratios, concordance | **`survival::lung`** (NCCTG advanced lung cancer; `fixtures/lung_coxph.csv`, complete cases) | **Not California Housing, not simulated.** `survival::lung` (Loprinzi et al. 1994) is the canonical real-world Cox PH dataset used in every survival tutorial. R writes the prepared complete-case CSV; Python reads the same bytes. California's HouseAge/high_value proxy is not a survival process and caused Cox PH non-convergence in the original version |
| Mixed models (LMM) | `lmer()` / lme4 | `california_prepared.csv` + `block_id` grouping | The block_id variable gives a real clustered structure |
| Bootstrap | `boot::boot()` | `california_prepared.csv` MedInc[1:1000] | Skewed real data gives a non-trivial bootstrap distribution |

**New-module suite (pystatistics 1.6.x)** — each module is tested on the
canonical real R dataset. `run_r_newmodules.R` loads the R built-in,
writes the exact numeric frame Python will read to
`suite1_pystatistics/fixtures/newmodules/`, and saves the R reference
values to `r_results.json`:

| WHAT (pystatistics) | HOW (R ref) | DATASET (real) | WHY that dataset |
|---|---|---|---|
| `fit(..., family=GammaFamily(link='log'))` | `glm(family=Gamma(link="log"))` on Ozone ~ Solar.R + Temp + Wind | **`datasets::airquality`** (fixtures/newmodules/airquality.csv, 111 complete cases) | Ozone is positive, continuous, right-skewed — a textbook Gamma regression target on real NYC air-quality data |
| `fit(..., family='negative.binomial')` | `MASS::glm.nb(Days ~ Eth + Sex + Age + Lrn)` | **`MASS::quine`** (fixtures/newmodules/quine.csv) | THE canonical negative binomial dataset (McCullagh & Nelder) — Australian schoolchildren absence counts |
| `polr(y, X)` | `MASS::polr(Sat ~ Infl + Type + Cont)` expanded by Freq | **`MASS::housing`** (fixtures/newmodules/housing.csv, 1,681 rows) | The dataset the `MASS::polr` help page itself uses — Copenhagen housing satisfaction survey |
| `multinom(y, X)` | `nnet::multinom(type ~ RI + Na + Al)` on the 3 largest glass classes | **`MASS::fgl`** (fixtures/newmodules/fgl.csv, restricted to WinF/WinNF/Head) | Canonical forensic glass dataset. 6-class fit doesn't converge in pystatistics' IRLS; 3 largest classes give a real, stable multiclass problem |
| `pca(X, scale=True)` | `prcomp(center=TRUE, scale.=TRUE)` | **`datasets::USArrests`** (fixtures/newmodules/USArrests.csv) | The dataset the `prcomp` help page uses — 50 US states × 4 crime rates |
| `factor_analysis(X, n_factors=2)` | `factanal(factors=2, rotation='varimax')` | **`datasets::mtcars`** (fixtures/newmodules/mtcars.csv) | USArrests has too few variables for a 2-factor fit; mtcars is the standard |
| `arima(y, order=(0,1,1), seasonal=(0,1,1,12))` | `stats::arima()` airline model on log(AirPassengers) | **`datasets::AirPassengers`** (fixtures/newmodules/airpassengers.csv) | THE canonical Box-Jenkins seasonal time series (1949-1960). The SARIMA airline model on it is the Box-Jenkins worked example |
| `acf`, `pacf` | `stats::acf`, `stats::pacf` — max_lag=20 | `AirPassengers` | Same series |
| `decompose(y, period=12)`, `stl(y, period=12)` | `stats::decompose`, `stats::stl(s.window='periodic')` | `AirPassengers` | Same series |
| `ets(y, model='MAM', period=12)` | `forecast::ets()` on raw AirPassengers | `AirPassengers` | Multiplicative-seasonal ETS is the natural model for this series |
| `gam(y, smooths=[s('times', k=20)])` | `mgcv::gam(accel ~ s(times, k=20), method='REML')` | **`MASS::mcycle`** (fixtures/newmodules/mcycle.csv) | THE canonical mgcv::gam example — simulated motorcycle-crash head acceleration vs. time |

### Suite 2: PyStatsBio vs R

**Core suite** — 5,891-subject CDC NHANES 2017-2018 dataset (demographics +
biomarkers) plus synthetic dose-response and PK data:

| WHAT (pystatsbio) | HOW (R ref) | DATASET | WHY that dataset |
|---|---|---|---|
| Power / sample size | `pwr::pwr.t.test()`, `pwr.2p.test()`, `pwr.anova.test()` — solve-for-n, solve-for-power | analytic (no dataset) | Power calculations are closed-form; no data needed |
| Dose-response (LL.4) | `drc::drm(fct=LL.4())` — EC50, hill, top/bottom | `dose_response_data.npz` (simulated 50 compounds × 8 doses from LL.4 with known parameters) | Real dose-response datasets are messy; a simulated one lets us verify parameter recovery |
| ROC / AUC | `pROC::roc()`, `pROC::ci.auc()` — AUC, DeLong SE/CI, DeLong test | NHANES biomarkers (e.g. GHb for diabetes classification) | Real diagnostic problem with a real continuous biomarker and binary outcome |
| Batch AUC | `pROC::roc()` loop over 16 markers | NHANES biomarker panel | Needs many markers for a meaningful batch-vs-sequential comparison |
| Optimal cutoff | Youden J from `pROC` | NHANES biomarkers | Real biomarker + clinical cutoff question |
| NCA pharmacokinetics | `PKNCA::pk.nca()` — AUC_last, Cmax, Tmax, half-life | `pk_data.npz` (5 oral + 1 IV simulated subjects, 1-compartment model) | Real PK datasets are proprietary; simulated one-compartment is the standard NCA teaching case |

**New-module suite (pystatsbio 1.5.0)** — uses canonical real-world R
datasets (or published real data) where one exists. `run_r_newmodules.R`
writes the inputs to `suite2_pystatsbio/fixtures/newmodules/` and the R
references to `r_results.json`:

| WHAT (pystatsbio) | HOW (R ref) | DATASET (real where possible) | WHY that dataset |
|---|---|---|---|
| `epi.epi_2by2(table)` | Manual R formulas (Woolf log-OR CI, log-RR Wald, Wald RD) | **Physicians' Health Study aspirin MI data** (NEJM 1989; 321:129-135). 11,037 aspirin / 104 MI vs. 11,034 placebo / 189 MI. `epi_2by2.json` | Published, famous real-world 2×2 where aspirin roughly halves MI risk. Result is textbook |
| `epi.mantel_haenszel(tables)` | `stats::mantelhaen.test(correct=FALSE)` | **`datasets::UCBAdmissions`** — 6 Berkeley graduate departments × {Admit, Reject} × {M, F}, 1973. `mh_tables.json` | THE canonical stratified-analysis / Simpson's paradox dataset (Bickel 1975). Shows apparent sex bias in aggregate that disappears after stratifying by department |
| `epi.rate_standardize(...)` | `epitools::ageadjust.direct()` | **Synthetic** (no compact canonical real age-standardization dataset). `rate_std.json` | Documented as synthetic; 5 age strata + standard pop exercises the direct code path. Keep an eye out for real US vital-stats age-standardization examples for future swaps |
| `meta.rma(yi, vi, method='REML')` | `metafor::rma()` on log ORs from `escalc()` | **`metafor::dat.bcg`** — 13 BCG tuberculosis vaccine trials. `meta_yi.csv` | THE canonical metafor example. Historical meta-analysis with real heterogeneity (I² ≈ 92%) so REML vs FE results differ meaningfully |
| `gee.gee(y, X, cluster_id, corr='exchangeable')` | `geepack::geeglm(Weight ~ Time + Cu, id=Pig, corstr='exchangeable')` — coefs + robust SEs | **`geepack::dietox`** — 72 pigs × ~12 weekly weights, dietary copper study. `gee_long.csv` | THE canonical geepack example, used in the vignette and in Fitzmaurice/Laird/Ware. Real repeated-measures longitudinal data |

### Suite 3: GPU stress test (TCGA BRCA RNA-seq + large synthetic)

Real-world TCGA breast cancer gene expression data (1,155 patients, 20,000 genes) from UCSC Xena, plus large synthetic regression data:

| Category | Tests | Key results |
|----------|-------|-------------|
| **Correctness** | GPU matches CPU to rtol=1e-3 | OLS, GLM, batch AUC, coefficient correlation > 0.9999 |
| **Speed** | OLS 500K x 200: **42x**, OLS 1M x 100: **44x**, GLM 50K x 100: **5x** | RTX 5070 Ti vs CPU |
| **Stability** | Deterministic, no NaN, memory cleanup | 5 repeated runs, full pipeline |

## Hardware

Tested on **Forge**: Ubuntu 24.04, AMD Ryzen 5 7600X, 64GB DDR5, NVIDIA RTX 5070 Ti (16GB VRAM, CUDA 12.0).

## Setup

Expects `pystatistics` and `pystatsbio` source trees at `../pystatistics` and `../pystatsbio` (sibling directories). The setup script installs them as editable (`pip install -e`) so you are always testing the latest local code, not a PyPI snapshot.

```bash
# Create conda environment and install from local source
bash setup_env.sh
conda activate test

# Download datasets (California Housing, NHANES, TCGA BRCA)
python data/download_datasets.py

# Generate derived data for core suites
python suite1_pystatistics/generate_data.py
python suite2_pystatsbio/generate_data.py
python suite3_gpu_stress/generate_stress_data.py

# Generate R reference fixtures (core + new-module).
# The new-module R scripts also WRITE the input CSVs by loading canonical
# R built-in datasets — no Python-side generator needed for them.
Rscript suite1_pystatistics/run_r_validation.R
Rscript suite1_pystatistics/run_r_newmodules.R
Rscript suite2_pystatsbio/run_r_validation.R
Rscript suite2_pystatsbio/run_r_newmodules.R
Rscript suite3_gpu_stress/run_r_stress.R

# Generate CPU fixtures for GPU comparison
python suite3_gpu_stress/generate_cpu_fixtures.py
```

## Running tests

```bash
conda activate test

# All suites (excludes slow GPU speed tests by default)
pytest

# All suites including GPU speed benchmarks
pytest -m "slow or not slow"

# Individual suites
pytest suite1_pystatistics/ -v
pytest suite2_pystatsbio/ -v
pytest suite3_gpu_stress/ -v -m "slow or not slow"
```

## Datasets

| Dataset | Source | Size | Suite |
|---------|--------|------|-------|
| California Housing | sklearn / StatLib | 20,640 rows x 8 features | Suite 1 |
| NHANES 2017-2018 | CDC | 5,891 subjects x 20 biomarkers | Suite 2 |
| Synthetic dose-response | Generated (LL.4 curves) | 50 compounds x 8 doses | Suite 2 |
| Synthetic PK profiles | Generated (1-compartment) | 6 subjects | Suite 2 |
| TCGA BRCA RNA-seq | UCSC Xena / GDC | 1,155 patients x 20,000 genes | Suite 3 |
| Simulated survival | Generated by R (`rexp`) | 500 subjects, 2 covariates | Suite 1 (Cox PH) |

## R packages required

Core: `jsonlite`, `moments`, `car`, `survival`, `lme4`, `lmerTest`, `boot`,
`pwr`, `drc`, `pROC`, `PKNCA`.

New-module tests additionally require: `MASS` (polr, glm.nb), `nnet`
(multinom), `mgcv` (gam), `forecast` (ets, auto.arima), `metafor` (rma),
`geepack` (geeglm), `epitools` (ageadjust).

## Notes

- **Dataset policy for new-module tests.** Wherever a canonical real-world
  R dataset exists for the module being tested, we use it (survival::lung,
  MASS::quine, MASS::housing, MASS::fgl, datasets::USArrests,
  datasets::mtcars, datasets::AirPassengers, MASS::mcycle,
  datasets::UCBAdmissions, metafor::dat.bcg, geepack::dietox). The R
  validation script loads the built-in and writes the exact numeric frame
  Python will read to `fixtures/newmodules/…` — so both languages fit
  byte-identical data. The single remaining synthetic fixture
  (`rate_std.json`) is flagged as such in the README and in its test
  docstring. Going forward, any test added for a new module must state
  WHAT, HOW, DATASET (name the real dataset, or state explicitly that it
  is synthetic and why), and WHY in its module docstring.
- **Runtime parity.** Every new-module test records Python wall-clock time
  alongside the R wall-clock time (from `system.time(...)$elapsed`) and
  asserts Python is no more than 20× slower than R, for fits R completes in
  ≥ 50 ms. Shorter fits are noise-dominated and skip the ratio check. The
  goal is to catch pathological regressions (we once shipped a solution that
  was 1000× slower than R); we do not require Python to beat R.
- **Cox PH** (core suite) uses **`survival::lung`** (NCCTG advanced lung
  cancer, Loprinzi et al. 1994) — the canonical real-world Cox PH dataset
  — **not** the California Housing dataset used by every other core-suite
  test. The proxy (HouseAge as "time", high_value as "event") is not a
  valid survival process and caused Cox PH non-convergence in the original
  version. R writes the prepared complete-case CSV to
  `fixtures/lung_coxph.csv` so Python fits the same bytes.
- Bootstrap and permutation GPU backends detect if the user statistic is mean/mean-difference and vectorize on GPU; arbitrary user functions fall back to CPU transparently.
- Batch AUC GPU is beneficial at scale (1000+ markers); for small marker counts, CPU is faster due to GPU launch overhead.
- The top discriminative gene for ER status in TCGA BRCA is ESR1 (estrogen receptor 1) -- biologically correct, validating the end-to-end pipeline.
