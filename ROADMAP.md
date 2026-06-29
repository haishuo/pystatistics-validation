# Validation program roadmap

The module-by-module plan for the pystatistics validation corpus. Tackled **one at
a time**, each to the rigor of `RIGOR.md` (treat as if publishing). The
**coordinator chip** owns this file: it advances the program one module at a time
and keeps the status table current.

**Current library version:** pystatistics **4.2.4** (on PyPI). Lineage: 4.0
standardized the API to `CONVENTIONS.md`; **4.1.0** was a consistency-only sweep of a
multi-module inconsistency survival surfaced (R8); **4.2.0** landed survival's
optimizations (incl. the coxph O(n²)→O(n) fix, R1); **4.2.3** shipped the plain-fp32
GPU GLM convergence fix (R9) re-validated across survival + regression; **4.2.4**
made the MPS fp32 GLM path a gated matrix-free CG solver (squaring-free, host-fp64
acceptance gate kept).

**Survival is DONE** (validated at 4.2.0 → GPU/real-flchain 4.2.1 → 4.2.3).
**Regression re-validated at 4.2.3.** The **one open item** is a re-render of survival +
regression against **4.2.4** — see "Open work" below.

## Order + status

Ordered by: foundation-first (the shared `core/compute` kernel) → highest-leverage
rideable methods → heaviest optimization headroom → correctness-dominant tail.

| # | Module | Status | Notes |
|---|---|---|---|
| 1 | regression (OLS + GLM families) | ✅ done (re-validated v4.2.3) | baseline v3.18.0 → optimized v3.20.0 → re-validated v4.2.3 (CPU-beats-R + fp32 GLM fix). Pending 4.2.4 re-render (Open work). Red-team gaps to close on next re-validation: R10 hard-case grid, R11 precision/hardware isolation, R12 fp32 no-silent-wrong stress test. |
| 2 | survival (KM / log-rank / coxph) | ✅ done | v4.0.0 baseline → v4.2.0 (coxph O(n²)→O(n), R1) → v4.2.1 (GPU on real flchain) → v4.2.3 (fp32 GLM fix). Pending 4.2.4 re-render (Open work). |
| 3 | multivariate (PCA + factor analysis) | ⬜ next NEW module | SVD/eigendecomposition; the most GPU-amenable op; TCGA large-matrix scaling; self-contained. Starts after the 4.2.4 re-render AND user says go. |
| 4 | mixed (LMM) | ⬜ pending | biggest optimization headroom (least GPU-touched); REML over variance components |
| 5 | gam | ⬜ pending | penalized IRLS + REML smoothing selection; rides the kernel |
| 6 | timeseries (ARIMA/ETS/STL) | ⬜ pending | largest module; Kalman/state-space + optimizer loops |
| 7 | montecarlo (bootstrap/permutation) | ⬜ pending | embarrassingly parallel → clean GPU story |
| 8 | ordinal (polr) + multinomial | ⬜ pending | IRLS-family; de-risked by the optimized kernel |
| 9 | anova, descriptive, hypothesis | ⬜ pending | correctness-dominant, optimization-light; corpus completeness; batch last |

Done already (pre-order): mvnmle (v3.18.0), mice (v3.16.3 / v3.18.0).

## Open work

- **4.2.4 re-render (survival + regression).** Spec'd in `DELETE-ME-revalidate-4.2.4.md`
  (repo root). Re-validate both modules against PyPI **4.2.4** (`require_pypi`),
  regenerate artifacts (Mac CPU+MPS; Forge CUDA), render reports, commit+push, delete
  the hand-off note. Expected change: MPS now CONVERGES via gated CG where it failed
  loud at quarterly-cold; survival MPS prose shifts to "converges via gated CG; refuses
  loud → CPU at the genuine precision floor." Prior reports (v4.2.3/4.2.1/4.2.0) stay
  frozen. **This is the next chip**, ahead of any new module.
- **Regression red-team hardening — CHIP RUNNING (combined, v4.2.4).** Closes the
  red-team gaps at v4.2.4 by ADDING evidence (existing v4.2.4 numbers stay frozen),
  ordered by the new RIGOR priority hierarchy: **(1) correctness** vs promised
  tolerance; **(2) R10 hard-case grid** matching R's failures/warnings; **(3) the
  mandatory CPU-vs-R speed study ACROSS SIZES** (small-n overhead → large-n) — the
  biggest addition, since the report only had 5 CPU-vs-R points; **(4) GPU at the weak
  bar** — R12 no-silent-wrong boundary stress test (correctness, stays high), R11
  precision-isolation + reference BLAS framed as "GPU meets its weak bar," not GPU
  superiority. Forge/CUDA allowance granted. NOTE: the chip launched just before the
  priority section landed (9772b49) — it was told to `git pull` + re-read RIGOR and
  reorder. If R12 finds a silent-wrong band → R6 library fix on a branch, held,
  surfaced to the user.

## Standing coordination constraints

- **Release-hold: CLEARED.** The `pystatsbio`/`sgcbio` consistency releases have
  landed and pystatistics has since shipped through **4.2.4**. No active hold. Re-check
  before any new library release; reinstate this line if a downstream consistency
  release is mid-flight again.
- **One at a time.** Do not spawn the next module chip until the current one is done
  AND the user says go. The coordinator confirms before spawning.
- **Hard cases + match R's failures/warnings (RIGOR R10).** Correctness grids must
  reach the adversarial regime (collinearity to the refusal boundary, separation,
  factor coding, weights/offsets, rank-deficiency) and match R's failures and warnings
  — not just easy-data numbers.
- **Isolate precision from hardware in GPU benchmarks (RIGOR R11).** Report `gpu_fp64`
  vs `cpu_fp64` (hardware alone) alongside any bundled fp32-GPU-vs-fp64-CPU-R number,
  and name the BLAS R linked against.
- **Relaxing fail-loud → convergence needs a no-silent-wrong proof (RIGOR R12, the
  complement to R9).** Any fail-loud→"converges" relaxation must carry an adversarial
  stress test that the accept/refuse boundary is principled (no accepted-but-wrong
  band). R9 + R12: the gate must be a true classifier both directions.
- **fp32 GPU non-convergence is a false-negative until proven otherwise (RIGOR R9).**
  CPU is fp64; MPS/CUDA fp32 cannot meet R's absolute `|Δβ| < 1e-8`. Before recording
  any MPS/CUDA path as failing/unstable, compare its coefficients to the fp64 fit —
  agreement to the fp32 tier (`max_rel ~1e-6`) means the fit is correct and the
  convergence *test* is wrong. Do not blame the GPU backend without cause.
- **Cross-module consistency discoveries → pause + library-wide minor release** (see
  RIGOR.md R8). If, while validating a module, you find a consistency issue that
  affects **more than one module**, pause that module's work and ship a consistency
  minor release for the library as a whole (this is exactly what **4.1.0** was — a
  multi-module fix surfaced by survival, spun out so survival could continue). **If
  the fix is breaking, STOP and discuss with the user** (a breaking change is not a
  quiet minor release).

## Per-module chip template (the coordinator fills `<<…>>` and embeds the rules)

Each module chip is self-contained (the spawned session has no prior context) and:
1. Points at `ARCHITECTURE.md`, `RIGOR.md`, `CONVENTIONS.md`, and the completed
   examples (`reports/regression-v4.2.3.md`, `survival-v4.2.3.md`, mvnmle, mice) +
   the harness `pystatsval` API + the salvageable R refs in `_archive/`.
2. States the module, its R reference (the R package/function), the canonical
   dataset(s), and the target version (current PyPI release — **4.2.4**).
3. Mandates the full `RIGOR.md` deliverables — correctness vs R **incl. the R10
   hard-case grid (match R's failures/warnings)**, the **R1/R3 complexity+scaling
   study across n/p with R11 precision-vs-hardware isolation**, the **R4
   constitutional audit**, **R12 no-silent-wrong proof for any relaxed fail-loud
   path**, and an honest perf story (R2: document any justified slowdown; never sweep).
4. Repeats the current **release-hold** status and the Forge standing-CUDA-testing
   allowance (only if a GPU path is warranted per the constitution).
5. Deliverables: `drivers/<m>/`, `artifacts/<m>/v4.2.4/`, `subsystems/<m>/meta.json`,
   `reports/<m>-v4.2.4.md`; commit to validation `main` and push.
6. Opens with discuss-before-acting: understanding + plan first.

(The `survival` and `regression` chips are the worked examples of this template.)
