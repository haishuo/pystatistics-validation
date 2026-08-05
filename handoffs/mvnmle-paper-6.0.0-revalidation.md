# mvnmle — 6.0.0 re-validation & paper red-team (handoff)

**Question that started this:** did the `mvnmle` module (subject of the forward-Cholesky
paper) and `mice` get the same heavy red-team treatment as the other 9 pystatistics
modules — and does the paper, pinned to **3.18.0**, still hold against the current
**6.0.0**?

**Answer:** mvnmle *was* red-teamed (at 4.6.13, findings M1/M2 fixed), but the paper is
pinned to **3.18.0 — which predates those fixes**. Fresh re-validation at 6.0.0 across
CPU / Apple-Metal(MPS) / CUDA reproduces **every table and figure** in the manuscript.
Re-pinning the paper to 6.0.0 costs nothing on the reported numbers and *fixes* a
prose-vs-code contradiction in the pinned version.

All evidence: `artifacts/mvnmle/v6.0.0/runs/` (8 JSON artifacts, CPU+MPS+CUDA).
Regenerated via `drivers/mvnmle/*` against a PyPI `pystatistics==6.0.0` install (never a
local checkout). Paper repo was NOT modified.

---

## 1. Reproduction: paper (3.18.0) vs 6.0.0

**Table `rcompare` — WVS log-likelihood**

| p | Paper CPU/FP64 | 6.0.0 CPU/FP64 | Paper GPU/FP32 | 6.0.0 GPU/FP32 (MPS) |
|---|---|---|---|---|
| 15 | −479040.327 | −479040.327 (exact) | −479040.375 | −479040.125 |
| 20 | −665376.165 | −665376.165 (exact) | −665376.062 | −665376.0 |
| 25 | −829652.971 | −829652.971 (exact) | −829652.938 | −829653.125 |
| 50 | −1495012.87 | (see CUDA) | −1495013.75 | −1495013.375 |

- CPU/FP64 column **bit-identical** to the paper; vs live R `mvnmle` at p=15, rel 5.9e-10.
- CPU path bit-stable 3.18.0→6.0.0 (mvnmle carries no Cython `.pyx`; migration was TS/survival only).

**Cross-architecture (CUDA, RTX 5070 Ti / sm_120, torch cu128):**
- WVS p=100: **iter=86** — exact match to the paper's "same iteration count (86 at p=100)" claim; ll −1776136.25 (paper −1776135.75, fp32 tol).
- WVS p=50: ll −1495014.125 — matches paper's "−1.49501e6 on both CUDA and Metal".

**Factorial (Fig. 2) — device-dispatch contract, both backends reproduce:**
- Metal p=50 obj solve→blocked: 6308→177 ms (paper 5785→165); grad blocked autodiff→analytic 5080→205 ms (paper 5085→192).
- CUDA p=50 obj solve→blocked: 18.8→34.4 ms (paper 18.80→34.37) — blocked *counterproductive* on CUDA, confirming solve-on-CUDA dispatch; grad 399.6→373.0 (paper 399.48→372.93).

**fp32 no-silent-wrong classifier (R12/R13), FRESH at 6.0.0 on both MPS and CUDA:**
- κ sweep: safe-subset refuse at κ=1e6 (fp64 accepts, fp32 refuses) on both devices — never the dangerous direction.
- CUDA numbers match the 5.0.0 report's *carried-forward* CUDA leg ~exactly → **carried-forward gap now closed** with a fresh measurement; fp32/fp64 kernels byte-stable across the Cython migration.

---

## 2. Red-team findings

- **M1 (paper-exposing).** On the pinned **3.18.0**, a constant / zero-variance column
  returns `converged=True` with a meaningless loglik — exactly the "spurious 'converged'
  fit" the manuscript's Discussion says *cannot* happen. **Fixed at 4.6.13**; on **6.0.0**
  the same input raises `SingularMatrixError`. → The paper's fail-loud prose is **false for
  3.18.0, true for 6.0.0.** Strongest argument for re-pinning.
- **M2 (framing).** fp32 *covariance* Frobenius error reaches ~2% at κ≈3e5 (reproduces on
  MPS+CUDA at 6.0.0). The paper reports fp32 *log-likelihood* agreement (1e-6…3e-4) and
  hedges the conditioning boundary, but does not report that the fp32 *covariance estimate*
  — the actual estimand — carries a few-percent error at GSS-like conditioning. Judgment
  call, not a defect.
- **GSS CPU-vs-R gap (not a defect).** GSS p=15 CPU loglik is 16.7 *higher* than R with a
  valid PD solution → R's finite-difference optimizer under-converges on ill-conditioned
  GSS. Pre-existing (5.0.0 showed ~20). GSS is not a headline table.
- **"All fits converged" holds.** A single MPS p=20 `converged=False` was fp32
  non-determinism (8/8 on repeat).

---

## 3. Recommendation: re-pin the paper to 6.0.0

Because the headline numbers reproduce and M1 makes the Discussion honest.

**Specific manuscript changes (for the author to apply — paper repo not touched here):**
1. `final_submission/software.txt` + any in-text version mention: **3.18.0 → 6.0.0**
   (and update the "numbers require version X" note — they now require 6.0.0, a wheel-shipping
   release, which also removes the numpy-2.x/numba install fragility of the 3.18.0 pin).
2. Table `rcompare`, `endtoend`, factorial: values are within fp32/timing tolerance of the
   reprints above; regenerate cleanly if desired (all drivers + this env ready).
3. Optional (M2): add one sentence to §Discussion that the fp32 *covariance* error (not just
   loglik) grows to a few percent at high conditioning — pre-empts a referee.

**Reusable test env (persistent, on Forge):** conda env `pystatistics-test` (cloned from
`gpumice`, has cu128/sm_120 torch + pystatistics 6.0.0). Future version tests: just
`pip install pystatistics==X` into it. Drivers at `~/mvnval`.

Datasets need no configuration on Forge as of 2026-08-05: the mirror at
`/mnt/data/pystatistics-datasets` is namespaced and is in `store_io`'s fallback
list, so `DATASETS_ROOT` can be left unset. (This previously read "Datasets at
`~/mvndata`" — that directory held a two-file copy of `gss`/`wvs` that duplicated
the mirror byte-for-byte and has been removed.)

---

## 4. Not done / next

- **mice** fresh 6.0.0 re-validation (parallel track, lower urgency — not paper-coupled).
- Optional residuals: explicit PD-by-construction stress; GSS gradient≈0 confirmation
  (deprioritized — clean evidence across all three backends).
