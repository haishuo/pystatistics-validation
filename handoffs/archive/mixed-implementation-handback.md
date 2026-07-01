# `mixed` v4.5.0 — Implementation Hand-back (to the validation session)

**From:** the implementation session (the IMPLEMENTER chip).
**Date:** 2026-06-30.
**Status:** All CPU fixes + the GPU low-rank/GRM model are implemented, unit-
tested, and through their own gates. **Nothing is published; the version is not
bumped.** This session did not bless. The intended release version is **4.5.0**
(minor — additive API + perf rewrite, no breaking changes).

The work lives **uncommitted in the working tree** of
`/Volumes/Archive/Documents/Dropbox/Dev/pystatistics` (branch `main`, on top of
`24c0b09`). 19 files, +2096/−145. Full library suite green: **3008 passed, 110
skipped, 0 failures**.

---

## 1. What changed, per file

**Phase 1 — CPU fixes (LMM):**

| File | Change |
|---|---|
| `mixed/solvers.py` | F2a: `_compute_se` → form A `σ²·(RX⁻¹)ᵀ(RX⁻¹)` (was dense n×n `V*` solve). F1: `is_singular` detection + `RuntimeWarning` + field. Routed LMM deviance / final solve / Satterthwaite to the structured solver; `_compute_fit_stats` uses `logdet_M`. **GLMM path untouched.** |
| `mixed/_common.py` | `LMMParams.is_singular: bool` (new field). |
| `mixed/solution.py` | `LMMSolution.is_singular` accessor. |
| `mixed/_random_effects.py` | `is_singular_fit()` (lme4 `isSingular` rule); `parse_random_effects(build_dense=False)` + `value_cols` (compact n×q; LMM no longer materializes the dense per-factor block). |
| `mixed/_struct_batched.py` (new) | Single-factor batched per-group dense Cholesky (block-diagonal M), ragged-safe via segment sums. |
| `mixed/_struct_sparse.py` (new) | Crossed/nested sparse factor: sparse Z + `splu(M, permc_spec='MMD_AT_PLUS_A')`. |
| `mixed/_pls_structured.py` (new) | Dispatcher + shared Schur assembly + structured deviance; `StructuredPLSResult`. |
| `mixed/_satterthwaite.py` | Rewired onto the structured solver; `C(θ)=(RX·RXᵀ)⁻¹` (no dense `V*`). Algorithm unchanged. |

**Phase 2 — GPU low-rank / GRM model (new, separate, honestly-named):**

| File | Change |
|---|---|
| `mixed/grm.py` (new) | Public `grm_lmm(y, X, W, *, backend, reml, names, tol, max_iter, conf_level, force)`; validation, `resolve_backend`, dispatch, result assembly (β SE form A, z/p, varcomps, h², logLik/AIC/BIC). |
| `mixed/_grm_cpu.py` (new) | CPU float64 reference: Woodbury M-space profiled deviance + 1-D θ search (`GRMFit`). |
| `mixed/backends/grm_gpu.py` (new) | torch GPU backend (CUDA/MPS, fp32/fp64) + the **CF-1 conditioning gate**. |
| `mixed/grm_solution.py` (new) | `GRMParams` + `GRMSolution` (uniform accessors + `heritability`, `var_genetic`, `var_residual`, `variance_ratio`, `genetic_values`, `summary`). |
| `mixed/__init__.py` | Export `grm_lmm`, `GRMSolution`. |
| `mixed/backends/__init__.py` (new) | package marker. |

**Tests (new):** `test_compute_se.py` (4), `test_singular.py` (7),
`test_structured.py` (9), `test_grm.py` (18). All pass; 1 GRM test is
CUDA-gated (skips locally, see §4).

---

## 2. Decisions actually made (the ones the brief flagged)

- **F2b crossed-design dependency: scipy-only, NO new dependency.** The crossed /
  nested path uses scipy `splu` with the `MMD_AT_PLUS_A` fill-reducing ordering.
  It met correctness + speed at InstEval-scale (a 2000×1000-level, 60k-obs crossed
  fit that previously OOM'd now completes in ~10 s), so I did **not** need to
  bring you the `scikit-sparse`/CHOLMOD question. No hard dependency was added.
- **A.3 (autodiff θ-gradient): DEFERRED, not included.** It is torch-only, and
  wiring it into the LMM deviance would pull `torch` from optional → required on
  the core CPU path. Left as a roadmap item pending the library-wide torch-policy
  decision (raised separately to the coordinator session). The CPU fixes are pure
  numpy/scipy and ship the same regardless of how that lands.
- **GRM model naming/scope:** implemented as a separate `grm_lmm()` with a
  `backend=`, per §C. `lmm`/`glmm` remain CPU-only with no `backend=`. Never
  described as "LMM on GPU".

---

## 3. Gate results (numbers)

### Phase 1 gate — CPU correctness (PASS)

- **Full pystatistics suite:** 3008 passed, 0 failures. New unit tests for
  F2a/F1/F2b all green.
- **Correctness vs live lme4/lmerTest — UNCHANGED vs the pristine 4.4.1
  baseline.** I ran the validation repo's runner/agreement functions directly
  (bypassing the `require_pypi` `generate()` entry point), then `git stash`'d my
  changes and re-ran the identical harness on pristine 4.4.1:

  | | worst TIGHT | worst OPT |
  |---|---|---|
  | baseline 4.4.1 (dense) | 2.6e-11 | 1.0e-4 |
  | v4.5.0 (structured) | 2.2e-11 | 1.3e-4 |

  Tight tier is identical (mine marginally better). **Note the live-R tight
  agreement is ~2e-11, not the ledger's 3e-14** — that figure was a *stored-R*
  artifact; under live R both versions are optimizer-bound (bobyqa vs L-BFGS-B).
  (Saved as memory `mixed-gate-tight-tier-liveR`.)
- **SE specifically still matches R after form A** (sleepstudy se_max_rel 1.4e-5,
  identical to baseline).
- **F1 vs R:** on dyestuff2, `py_singular=True` ↔ `r_singular=True` with R's
  boundary message — exact match. HC1 correlation→±1 covered by unit test.
- **Scaling (the F2a+F2b payoff):** single-factor G=2000 **197 s → 0.05 s**;
  G=5000/n=100k in ~0.1 s; crossed 2000×1000/n=60k (previously OOM) in ~10 s. No
  OOM anywhere it previously failed.
- **GLMM smoke:** 15 GLMM tests pass (`solve_pls`/PIRLS/dense `build_z_matrix`
  untouched).

### Phase 2 gate — GPU GRM (PASS), verified on Forge CUDA (RTX 5070 Ti)

- **CPU reference exact:** M-space deviance == independent n×n V-space deviance to
  ~2e-13; heritability recovers truth (0.2/0.5/0.8 → 0.188/0.483/0.791).
- **`gpu_fp64` vs cpu:** β to **1e-10** (fp64-exact tight tier); heritability /
  varcomps to **~3e-8** — the 1-D θ-optimizer landing a hair differently off
  ~1e-12 cuSOLVER-vs-LAPACK deviance rounding, **not** an fp64 defect (identical
  to the documented polr gpu_fp64 precedent: the stop bounds deviance, not the
  derived quantity).
- **`gpu` (fp32) vs cpu:** |Δh²| ~1–3e-4, |Δβ| ~1e-6 — GPU_FP32 statistical-
  equivalence tier.
- **CF-1 boundary sweep: `silent_wrong_count == 0`** on both MPS and CUDA.
  Structurally so: the `+I` floor makes the fp32 Gram Cholesky either accurate
  (|Δh²| ≤ 9e-4 across cond(W)∈[1e3,2e4]×8 seeds) or a **loud** failure — never
  silently wrong. The conditioning gate (`_MIN_EIG_RATIO_FP32 = 1e-7`,
  cond(W)≈3e3) is calibrated to the measured fp32-correct boundary (accepts
  correct fits — R9; refuses biased/failing ones — R12; re-proven on this regime
  — R13; guarantee rests on the host-fp64 gate + loud Cholesky — R14).
- **Genuine speedup (full-fit, n=20000):** `gpu` fp32 **13–16×** over CPU; fp64
  1.1–2× (consumer card throttles fp64; more on a datacenter card). **MPS is
  correctness-only — no speedup (0.6–1.5×);** reported honestly as a portability
  path, not a performance one. The speedup story is CUDA.
- **Forge hygiene:** yield-checked (`nvidia-smi`: 0% util, one idle 260 MiB
  resident process), ran in the `gpumice` env against a patch-overlay of the
  `pystatistics-3.15.3` worktree, **restored the worktree** to its prior ref
  (`d363102`) and removed temp files afterward.

---

## 4. What the validation session must re-check

1. **Bump + publish 4.5.0.** UNRELEASED.md is staged (full content in §5). Run
   `.release/release.py` per your flow; publish to PyPI with the user's separate
   authorization. Then run the full frozen validation + red-team against **PyPI
   4.5.0** and bless `mixed-v4.5.0`.
2. **The one honest optimizer-tier nuance:** penicillin's Satterthwaite df is a
   hair looser than baseline (1.3e-4 vs 1.0e-4) — the structured deviance's
   rounding shifts where L-BFGS-B stops; the structured Satterthwaite is
   machine-identical to the dense one at *fixed* θ (unit-tested), so this is
   optimizer-stop, not a defect. Confirm it against PyPI 4.5.0 and record it as
   within the optimizer tier (it is 0.013% on a df).
3. **The CUDA-gated GRM test** (`test_gpu_fp64_matches_cpu`) skips off-CUDA. I
   verified its exact assertions on Forge via a gate script (β rtol 1e-6, h² <
   1e-5 both hold: measured β 1e-10, Δh² ≤ 3.3e-8), but the *test itself* has not
   run in CI against the published wheel — your CUDA CI / a Forge run should
   execute it against PyPI 4.5.0.
4. **GLMM remains out of scope for bless** (deferred). F2a/F2b touched shared-
   adjacent code but I left `solve_pls`/`_pirls`/dense `build_z_matrix`
   untouched; the GLMM suite is green. Still worth your smoke check post-publish.
5. **Torch policy is unresolved** (raised to the coordinator). It does not affect
   this release: the CPU fixes are numpy/scipy; the GRM GPU path is torch-gated
   as usual. A.3 is the only thing waiting on that ruling.

## 5. Staged `.release/UNRELEASED.md` content

The four bullets under `## Changes` (verbatim in the repo):

1. `lmm()` fixed-effect SEs now O(p³) not O(n³) (form A from the p×p Schur
   factor; machine-identical, several orders faster at large n).
2. `lmm()` reports boundary (singular) fits (`LMMSolution.is_singular` +
   `RuntimeWarning`, mirroring lme4 `isSingular`).
3. `lmm()` structure-exploiting solver (batched single-factor / sparse crossed;
   197 s→0.05 s at G=2000; crossed-scale OOM fixed; pure numpy/scipy; GLMM
   unchanged).
4. New model `grm_lmm()` — low-rank / GRM mixed model with a `backend=`
   (cpu/gpu/gpu_fp64/auto), CF-1 fp32 conditioning gate, `GRMSolution` with
   heritability + genetic-value BLUPs.

**Intended version: 4.5.0.** Do not treat this file as final wording — it is the
public changelog draft; adjust per your README/CHANGELOG discipline (Rule 9).
