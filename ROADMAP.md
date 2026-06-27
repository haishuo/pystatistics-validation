# Validation program roadmap

The module-by-module plan for the pystatistics validation corpus. Tackled **one at
a time**, each to the rigor of `RIGOR.md` (treat as if publishing). The
**coordinator chip** owns this file: it advances the program one module at a time
and keeps the status table current.

**Current library version:** pystatistics **4.1.0** (on PyPI). The 4.0 "consistency
release" standardized the API to `CONVENTIONS.md`; 4.1.0 caught a few more lurking
inconsistencies (and the coxph O(n²)→O(n) complexity fix — see R1).

## Order + status

Ordered by: foundation-first (the shared `core/compute` kernel) → highest-leverage
rideable methods → heaviest optimization headroom → correctness-dominant tail.

| # | Module | Status | Notes |
|---|---|---|---|
| 1 | regression (OLS + GLM families) | ✅ done | baseline v3.18.0 → optimized v3.20.0 |
| 2 | survival (KM / log-rank / coxph) | ✅ done @ v4.0.0; **re-validate @ 4.1.0** | the coxph O(n²)→O(n) fix landed in 4.1.0 and must be documented per R1/R6; the v4.0.0 report did NOT catch the O(n²) — the gap R1 exists to close |
| 3 | multivariate (PCA + factor analysis) | ⬜ next | SVD/eigendecomposition; the most GPU-amenable op; TCGA large-matrix scaling; self-contained |
| 4 | mixed (LMM) | ⬜ pending | biggest optimization headroom (least GPU-touched); REML over variance components |
| 5 | gam | ⬜ pending | penalized IRLS + REML smoothing selection; rides the kernel |
| 6 | timeseries (ARIMA/ETS/STL) | ⬜ pending | largest module; Kalman/state-space + optimizer loops |
| 7 | montecarlo (bootstrap/permutation) | ⬜ pending | embarrassingly parallel → clean GPU story |
| 8 | ordinal (polr) + multinomial | ⬜ pending | IRLS-family; de-risked by the optimized kernel |
| 9 | anova, descriptive, hypothesis | ⬜ pending | correctness-dominant, optimization-light; corpus completeness; batch last |

Done already (pre-order): mvnmle (v3.18.0), mice (v3.16.3 / v3.18.0).

## Standing coordination constraints

- **Release-hold (active):** `pystatsbio` and `sgcbio` are mid-consistency-release,
  pinning pystatistics 4.1.0. **Do NOT cut a new pystatistics release** (which moves
  the dependency under them) until those land. Validate 4.1.0 as-is; if a module
  finds a fix worth shipping (R6), implement on a branch and **hold the release**,
  surfacing it to the user to schedule. Re-check this constraint each module.
- **One at a time.** Do not spawn the next module chip until the current one is done
  AND the user says go. The coordinator confirms before spawning.

## Per-module chip template (the coordinator fills `<<…>>` and embeds the rules)

Each module chip is self-contained (the spawned session has no prior context) and:
1. Points at `ARCHITECTURE.md`, `RIGOR.md`, `CONVENTIONS.md`, and the completed
   examples (`reports/regression-v3.20.0.md`, `survival-v4.0.0.md`, mvnmle, mice) +
   the harness `pystatsval` API + the salvageable R refs in `_archive/`.
2. States the module, its R reference (the R package/function), the canonical
   dataset(s), and the target version (**4.1.0**).
3. Mandates the full `RIGOR.md` deliverables — correctness vs R, the **R1/R3
   complexity+scaling study across n/p**, the **R4 constitutional audit**, and an
   honest perf story (R2: document any justified slowdown; never sweep).
4. Repeats the **release-hold** constraint and the Forge standing-CUDA-testing
   allowance (only if a GPU path is warranted per the constitution).
5. Deliverables: `drivers/<m>/`, `artifacts/<m>/v4.1.0/`, `subsystems/<m>/meta.json`,
   `reports/<m>-v4.1.0.md`; commit to validation `main` and push.
6. Opens with discuss-before-acting: understanding + plan first.

(The `survival` and `regression` chips are the worked examples of this template.)
