# GAM GPU-feasibility investigation — verdict (Forge CUDA evidence)

**Date:** 2026-07-02 · **Author session:** gam GPU-feasibility chip (per the
per-submodule CUDA-first GPU investigation directive) · **Status:** investigation
complete, no library edits (Coding Bible Rule 8 — recommendations handed to the user
as separate first-class tasks).

> This is a **design memo**, not a version-pinned validation report. It informs the
> `gam` validation chip (task_31d5fd2b, target PyPI 4.5.7) and the GPU-backend
> decision. Companion scripts + raw outputs: `handoffs/gam-gpu-investigation-scripts/`.
> Reference shape: `mixed-gpu-investigation.md` (verdict B — general LMM/GLMM CPU-only).

> **Revision note (process + substance).** The first draft of this memo was built
> from local MPS measurements plus evidence carried from the GRM study, despite the
> chip's explicit CUDA-first instruction and the Forge standing allowance. That was
> wrong — and the Forge run **materially changed the numbers** (see §0), proving
> R13 the hard way: MPS/carried evidence does not transfer to CUDA. Every claim
> below is now measured on the target hardware (Forge: RTX 5070 Ti, sm_120, torch
> 2.11.0.dev+cu128; CPU baseline AMD Ryzen 5 7600X, numpy 2.3.5 + **MKL**), except
> where explicitly labeled otherwise.

---

## TL;DR verdict

**Can `pystatistics.gam` be genuinely GPU-accelerated on CUDA?**

- **In the regime GAMs are typically used (n ≲ 10k, p ≈ 30–80): NO meaningful win.**
  The inner evaluation is sub-millisecond on the CPU; even where CUDA edges ahead
  per-op (parity at n≈2k, a clear win by n=5k at p=50), the sequential ~50-eval
  L-BFGS-B λ-search × a few P-IRLS steps saves at most tens of milliseconds per fit. Nothing a user
  would notice; not worth a backend.
- **At large n (≥ ~100k): a genuine but MODEST safe win exists — ~3× (fp32
  augmented-QR) to ~5× (batched fp64 multi-λ grid) — and it requires a formulation
  the library does not currently have.** The spectacular number (25–78× for the
  fp32 normal-equations gram — the current `gpu_pirls.py` architecture) is **not
  shippable**: it carries a now-CUDA-proven silent-wrong EDF band (§4) with no gate.
  The guaranteed-correct fp64 variant of the same architecture **loses to the CPU
  at typical p** (0.67× at p=50) and reaches only ~3× at p=200.
- **The current `gpu_pirls.py` is the wrong architecture on both axes:** its fp32
  default is unsafe (ungated CF-1 band, worse on CUDA than anywhere measured), and
  its correct-by-construction fp64 variant doesn't win. Recommendation: **do not
  present GAM as GPU-accelerated as-is**; the near-term wins are CPU (§7). If
  large-n GAM becomes a real target, the safe formulations are now mapped with
  measured numbers (§5, §6).

No prior art contradicts this: mgcv's own big-`n` path (`bam`) is CPU/parallel-CPU;
there is no established GPU penalized-spline GAM.

---

## 0. What the Forge run overturned (corrections to the local draft)

Recorded explicitly, because each is an R13 lesson (a guarantee/measurement is
regime- and device-conditional):

| Claim (local MPS / carried draft) | Forge CUDA measurement |
|---|---|
| fp32-hybrid crossover ≈ n=20k; large-n win 2.8–7.7× | **Parity at n=2k (0.96×), clear win by n=5k (1.7×) at p=50; parity from n=500 at p=200; large-n win 43–78×.** MPS massively understated CUDA. |
| fp64 GPU ceiling ~1.2–1.6× (carried from GRM, same card) | **Wrong kernel shape to carry.** GAM-shaped fp64: **0.67× at p=50 (a loss)**, ~2.97× at p=200. |
| fp32-gram EDF error ≈ 9 DOF at λ=1e-8 (Accelerate BLAS) | **Worse on CUDA (cuBLAS): 33.6 DOF error (n=5k); EDF = −0.94 (negative!) at n=100k.** |
| `gpu_pirls._solve_for_edf` docstring: cuSOLVER vs LAPACK "diverge by factors of two" on near-singular A | **Not reproduced** up to cond 2.1e16 in fp64: agreement ≤2.7e-5 rel at cond 1e14; worst 0.098 EDF (3.7e-3 rel) at 2.1e16. The fp64 host round-trip is about **bit-parity with the CPU reference**, not numerical necessity. |
| TF32 hazard (not measured locally) | **Quantified:** TF32-formed gram → EDF error 2.2 DOF at λ=1e-6, 19.7 at λ=1e-8 — corrupts even moderately-smoothed fits. |
| Batched multi-λ (chip regime 3, not measured locally) | **Measured: 5.1× in fp64** (B=50, n=100k, p=50) at machine precision (1.4e-14). |

---

## 1. What the GAM fit actually computes (the compute profile)

From the source (`../pystatistics/pystatistics/gam/`):

- **Basis build (once per fit):**
  - `cr` (cubic regression spline, the mgcv default): a **Python loop over k** calling
    `scipy.interpolate.BSpline` column-by-column (`_basis._bspline_basis`), plus a
    grid-integrated penalty. O(n·k). The penalty `S` is **banded** (bandwidth = spline
    degree = 3).
  - `tp` (thin-plate regression spline): forms the **full n×n kernel**
    `E_ij=|xᵢ−xⱼ|³`, projects out the null space with n×n matmuls, and does a **full
    n×n `np.linalg.eigh`** (`_basis.thin_plate_spline_basis`). **O(n³) / O(n²) memory**
    — the heaviest op in the module and a scaling defect vs mgcv (which uses a
    low-rank/knot-based TPS). One-time setup, not in the loop; the right fix is
    algorithmic (CPU), not GPU (§7).
- **Model matrix:** `X_aug = [X_param | B₁ | … | Bₘ]`, shape (n, p); padded penalties
  `Sⱼ` (p, p). Typical `p = parametric + Σkⱼ ≈ 30–80`.
- **Smoothing-parameter selection:** `scipy.optimize.minimize(L-BFGS-B, maxiter=50)`
  over `log λ` — **~50 outer evaluations, inherently sequential** (finite-difference
  gradients, so 2·n_smooths+1 evals per step).
- **Each outer eval:** P-IRLS to convergence (1 step Gaussian; a few Newton steps
  otherwise). **Each P-IRLS step:** the `X'WX` GEMM — **O(n·p²), the only n-scaling
  linear algebra** — then `A = X'WX + Σλⱼ Sⱼ`, a p×p Cholesky solve, and the EDF
  hat-trace `F = A⁻¹ X'WX` (both **O(p³), tiny**).

The n-scaling work is a GEMM (GPU-friendly); the numerically hard work is a small,
potentially very ill-conditioned p×p solve. §2–§5 quantify how that tension plays
out on the target card.

---

## 2. Measured on Forge CUDA: where the time goes, and the crossover

`gam_cuda_study.py`, study A — the core inner op (elementwise n-vector work +
`X'WX` GEMM + p×p solve + EDF trace), per-eval µs. Four execution models:
`cpu_fp64` (numpy/MKL), `cuda_fp32_hybrid` (GEMM fp32 on device + host-fp64 p×p
solve — **what `gpu_pirls.py` does**), `cuda_fp64_hybrid`, `cuda_fp64_device`
(all-on-device fp64).

**p = 50 (typical GAM):**

| n | cpu_fp64 | cu_fp32_hyb | × | cu_fp64_hyb | × | cu_fp64_dev | × |
|---|---|---|---|---|---|---|---|
| 500 | 133 | 1 900 | 0.07 | 440 | 0.30 | 2 271 | 0.06 |
| 2 000 | 225 | 236 | **0.96** | 639 | 0.35 | 858 | 0.26 |
| 5 000 | 412 | 242 | 1.70 | 1 246 | 0.33 | 1 470 | 0.28 |
| 20 000 | 1 971 | 296 | 6.66 | 4 303 | 0.46 | 4 502 | 0.44 |
| 100 000 | 13 991 | 326 | 42.9 | 20 604 | **0.68** | 20 841 | 0.67 |
| 1 000 000 | 136 879 | 1 748 | **78.3** | 204 764 | 0.67 | 204 851 | 0.67 |

**p = 200 (several smooths / tensor-product):**

| n | cpu_fp64 | cu_fp32_hyb | × | cu_fp64_hyb | × | cu_fp64_dev | × |
|---|---|---|---|---|---|---|---|
| 500 | 737 | 712 | 1.03 | 848 | 0.87 | 1 525 | 0.48 |
| 5 000 | 2 433 | 753 | 3.23 | 1 781 | 1.37 | 2 417 | 1.01 |
| 100 000 | 62 561 | 1 516 | 41.3 | 21 712 | 2.88 | 22 344 | 2.80 |
| 1 000 000 | 627 549 | 8 885 | 70.6 | 210 794 | **2.98** | 211 101 | 2.97 |

**Readings:**

- **The fp32-hybrid path reaches parity at n=2k (0.96×) and a clear win by n=5k
  (1.7×) at p=50 — from n=500 at p=200** — far lower than the local MPS sketch
  suggested — and reaches 43–78× at large n. **But this is the unsafe path** (§3);
  its speed is not shippable without solving the correctness problem it carries.
- **fp64 (the correct-by-construction remedy) LOSES at p=50 at every size: 0.67× at
  best.** GeForce Blackwell fp64 throughput is that low. Only at p=200 does fp64
  reach ~3×. So "just run it in fp64" is not a win at typical GAM widths on this
  class of hardware.
- **The hybrid architecture itself is fine:** `cuda_fp64_hybrid` ≈ `cuda_fp64_device`
  everywhere — the p×p host round-trip costs nothing measurable. The bottleneck is
  fp64 GEMM throughput, not the round-trip.
- **Typical regime in absolute terms:** at n=5k/p=50 the CPU inner op is 0.41 ms.
  A whole fit (~150 inner evals) is ~60 ms of linear algebra on the CPU; the GPU's
  1.7× on that is imperceptible. There is nothing here to accelerate.
- Baseline honesty (R11): CPU numbers are numpy 2.3.5 + **MKL** on a Ryzen 5 7600X
  (12 threads) — a strong desktop BLAS baseline, same host as the GPU.

*(The earlier local MPS run — crossover ≈ n=20k, 2.8–7.7× — is retained in
`gam_gpu_microbench.py` for the record but is superseded by the CUDA numbers.)*

---

## 3. The conditioning wall — the CF-1 band proven on CUDA (study B)

Near-rank-deficient basis (geometric singular-value decay — the spectrum a high-k
spline basis on clustered x has) + small λ: the regime the outer optimizer
*deliberately probes* during the λ-search. `X'WX` formed by cuBLAS in fp32,
promoted to host fp64, EDF compared against the fp64 gram (TF32 off):

| n | λ | cond(A) | EDF (fp64) | EDF (fp32 cuBLAS gram) | abs error |
|---|---|---|---|---|---|
| 5 000 | 1e-2 | 1.2e4 | 6.208 | 6.208 | 0.000 |
| 5 000 | 1e-4 | 2.9e5 | 9.194 | 9.195 | 0.001 |
| 5 000 | 1e-6 | 6.9e6 | 12.221 | 12.233 | 0.012 |
| 5 000 | 1e-8 | 4.0e8 | 15.696 | **49.253** | **33.56** |
| 100 000 | 1e-8 | 4.5e8 | 15.614 | **−0.942 (negative!)** | 16.56 |

- **The silent-wrong band is worse on CUDA than on any host measured**: 33.6
  effective-DOF error, and at n=100k a *negative* EDF — cuBLAS fp32 accumulation
  order differs from Accelerate/OpenBLAS, and the error lands directly on the
  smallest eigenvalues that the EDF trace depends on. Promotion to fp64 after the
  fact cannot heal it (the `gpu_pirls` docstring itself concedes this).
- EDF drives GCV/REML → wrong λ → wrong fit. This is **CF-1 for gam**, on the
  default `gpu`/`auto` path, in the regime the λ-optimizer visits by design. The
  current fp32 GAM path has **no gate** analogous to GRM's host-fp64 conditioning
  refusal — an open R12 exposure.
- **TF32 hazard quantified (R14):** with `allow_tf32=True` the gram error grows to
  7.5e-6 relative and the EDF is off by **2.2 DOF already at λ=1e-6** and 19.7 at
  λ=1e-8. torch currently defaults matmul TF32 off, but `gpu_pirls` neither pins nor
  asserts it — a global flag flip (user code, another library, a future torch
  default change) would silently degrade even moderately-smoothed fits. Any GPU path
  must explicitly set/assert `allow_tf32=False` (guarantee on the stable layer, R14).

**Flag for the gam validation chip (task_31d5fd2b):** treat the fp32 `gpu`/`auto`
GAM path as a suspected silent-wrong (R12/CF-1) until a boundary sweep proves a
principled accept/refuse gate on **both** MPS and CUDA (R13). A default-reachable
uncaught wrong answer is a showstopper (R16). Note the gate design problem in §5.

---

## 4. The fp64 host round-trip: parity nicety, not numerical necessity (study D/D2)

The `gpu_pirls._solve_for_edf` docstring claims LAPACK and cuSOLVER "pivot
differently on near-singular matrices and diverge by factors of two (including
sign flips)". Measured, fp64 on-device solve vs numpy LAPACK on the same A:

| cond(A) | EDF (LAPACK) | EDF (cuSOLVER) | rel err |
|---|---|---|---|
| 4.0e8 | 15.6961 | 15.6961 | ~3e-11 |
| 1.8e10 | 19.3392 | 19.3392 | 9.0e-9 |
| 1.4e12 | 22.5879 | 22.5879 | 3.2e-7 |
| 1.2e14 | 26.4339 | 26.4346 | 2.7e-5 |
| 2.1e16 | 26.6821 | 26.5842 | 3.7e-3 |

**The divergence claim does not reproduce generically** — agreement holds to
cond ~1e14, and even at 2.1e16 (where ε_fp64·cond ≈ O(1) and *both* answers carry
that uncertainty) the difference is 0.098 EDF, not "factors of two". The docstring
likely generalized from one pathological pivot case. Consequence: the host
round-trip in the fp64 path is justified as **bit-parity with the CPU reference**
(a validation-contract convenience — and measured free, §2), not as a numerical
requirement. *(Flagged to the gam chip: the docstring overstates; correct it
whenever `gpu_pirls` is next touched.)*

---

## 5. The safe fp32 formulation — Wood's augmented QR, measured on CUDA (study C/C2)

The library solves the **normal equations** `A = X'WX + Σλⱼ Sⱼ`, which *squares*
the condition number and creates both the CF-1 band and the fp32 wall. Wood
(2011) / mgcv solve the **augmented least-squares** `[√W·X ; B]` (with `B'B =
Σλⱼ Sⱼ`) by QR — cond(augmented) ≈ √cond(normal equations). On CUDA
(cuSOLVER geqrf, fp32), near-rank-deficient basis:

| n | λ | cond(NE) | cond(aug) | β rel-err fp32 **NE** | β rel-err fp32 **QR** |
|---|---|---|---|---|---|
| 5 000 | 1e-8 | 4.0e8 | 2.0e4 | **10.7 (garbage)** | 2.0e-4 |
| 100 000 | 1e-8 | 4.5e8 | 2.1e4 | 1.57 | 3.4e-3 |
| 1 000 000 | 1e-8 | ~4.5e8 | ~2.1e4 | — | 5.8e-3 |

**Timing at the adversarial λ, p=40 (C2):**

| n | CPU inner op | fp32 QR (CUDA) | safe speedup | fp32 gram (CUDA) | unsafe speedup |
|---|---|---|---|---|---|
| 100 000 | 6.7 ms | 2.9 ms | **2.3×** | 0.26 ms | 25× |
| 500 000 | 44.4 ms | 13.5 ms | **3.3×** | 0.84 ms | 53× |
| 1 000 000 | 90.3 ms | 26.5 ms | **3.4×** | 1.50 ms | 60× |

- **The safe path is real but modest: ~3×.** cuSOLVER QR costs ~10–18× the GEMM, so
  the QR formulation gives up most of the gram path's speed to buy its stability.
- **Accuracy at the adversarial corner is at the edge of the fp32 tier** (5.8e-3 at
  n=1M) — cond(aug)·ε_fp32 ≈ 2.5e-3 is exactly the observed order. A shipped path
  would still need a host-fp64 conditioning gate for extreme λ (or one fp64
  iterative-refinement step, O(np) — cheap, unmeasured here).
- **Why a gate is awkward for GAM specifically:** unlike GRM (one gate per fit), the
  conditioning of A depends on λ, and the optimizer probes small λ **transiently, by
  design**. A gate that refuses mid-λ-search kills fits whose *final* λ is perfectly
  safe. Workable designs (gate only the final fit; clamp the λ search box on the GPU
  path *with disclosure*; QR+refinement everywhere) all add complexity a ~3× win may
  not justify. This is the honest engineering reason the "spectacular" fp32 number
  stays unshippable.
- The augmented-QR reparameterization is worth adopting **on the CPU regardless** —
  it removes the cond~10¹⁶ normal-equations fragility the code currently patches
  with ridge fallbacks (§7).

---

## 6. Batched multi-λ (chip regime 3) — the other honest win, fp64-safe (study E)

Batch B=50 weight-vectors' `X'W_bX` + batched p×p solves + traces as one CUDA op
(n=100k, p=50, **fp64**): **5.1× over sequential CPU, max EDF deviation 1.4e-14**
(machine precision — no CF-1 exposure at all, it's fp64 end to end).

Why this matters: batching fixes exactly the utilization problem that makes
*sequential* fp64 lose (0.67×, §2). But it is an **algorithmic redesign**, not a
backend swap: the λ-search must become a grid/direct-search (natural for 1–2
smooths — which covers most single- and double-smooth GAMs — cursed for many
smooths), and non-Gaussian families need a batched P-IRLS with per-slice
convergence masking. Memory: the (B,n,p) weighted copy materializes 2 GB at
B=50/n=100k/p=50 fp64 — needs chunking at larger sizes. For Gaussian/identity fits
the gram is λ-independent (W≡1), so there is nothing to batch — this regime is
specifically the non-Gaussian large-n λ-grid.

**If a large-n GAM path is ever commissioned, this is the formulation to prototype
first**: fp64-safe (no gate needed), ~5× measured, and it composes with the
augmented-QR for the final fit.

---

## 7. Where the real GAM wins are (CPU — mirroring the `mixed` outcome)

Recommendations to the user for separate first-class tasks — none applied here
(Rule 8):

1. **`tp` basis: replace the full n×n eigendecomposition with a low-rank/knot-based
   TPS** (mgcv-style). O(n³)/O(n²)-memory today; caps practical n at a few thousand
   — this defect makes the "large-n GAM on GPU" question moot for `tp` smooths until
   fixed, and it is the module's heaviest op. *(Likely also a correctness/parity
   item for the gam validation chip.)*
2. **Adopt the stable augmented-QR / reparameterization on the CPU** — removes the
   normal-equations fragility and is the prerequisite for any future fp32 GPU path.
3. **Analytic GCV/REML gradient** (the `mixed` A.3 pattern): kills the
   finite-difference multiplier (2·n_smooths+1 evals/step) in the λ-search — the
   biggest practical end-to-end CPU speedup available.
4. **Vectorize the `cr` B-spline construction** (Python loop over k → one vectorized
   `BSpline.design_matrix` call); exploit the banded `cr` penalty where useful.
5. **Keep `gam` CPU-only for now.** The current `gpu_pirls.py` should not be
   presented as a GPU win: fp32 is unsafe (§3), fp64 loses at typical p (§2). If
   large-n non-Gaussian GAM becomes a target, prototype §6 (batched fp64 grid) +
   §5 (QR) — with an explicit `allow_tf32=False` pin and a CF-1-style gate design
   resolved up front.
6. **Correct the `gpu_pirls._solve_for_edf` docstring** (§4) whenever the file is
   next touched — the cuSOLVER-divergence rationale is overstated; the real
   justification is CPU-reference bit-parity.

---

## 8. RIGOR / carry-forward touch points

- **CF-1:** GAM's fp32 `X'WX` is a proven CF-1 instance **on CUDA** (§3: 33.6-DOF
  error, negative EDF), reachable via `backend='gpu'`/`'auto'` in the λ-probing
  regime. The gam validation chip must run the R12 boundary sweep on **both** MPS
  and CUDA (R13) or the fp32 path must be gated/removed. Gate design is nontrivial
  here (§5 — λ-transient conditioning); surface that to the user before the chip
  commits to a gate architecture.
- **R11:** precision and hardware effects are isolated (§2: fp32-hybrid vs
  fp64-hybrid vs fp64-device, same card, same host), and the CPU baseline BLAS is
  named (MKL). The GRM-carried fp64 ceiling was replaced by a direct measurement —
  and differed (0.67× vs 1.2–1.6×): kernel shape matters; do not carry ceilings
  across kernel mixes again.
- **R13 (self-inflicted lesson):** every load-bearing number in the first draft
  moved — some by an order of magnitude, one across the win/lose boundary — when
  re-measured on the target device. MPS evidence is a sketch, never a verdict, for
  a CUDA question.
- **R14:** the TF32 measurement (§3) is the version-sensitive hazard; the
  version-independent guarantee must be an explicit `allow_tf32` pin + a
  host-fp64-keyed gate, exactly as GRM's.
- **Guarantee 3 / GPU corollary:** a GPU that ties/loses in its intended regime has
  no reason to exist. The shippable-safe formulations win ~3–5× only at n ≥ 100k
  in formulations the library doesn't have; the typical regime has nothing to win.
  An honest "no GPU path (for now), and why" is the sanctioned outcome.

---

## Appendix — reproduction

Scripts + raw outputs in `handoffs/gam-gpu-investigation-scripts/`:

**Forge CUDA (primary evidence):**
- `gam_cuda_study.py` → `forge_cuda_results.json` — studies A (crossover, 4
  execution models), B (CF-1 band + TF32), C (augmented QR), D (cuSOLVER vs
  LAPACK), E (batched multi-λ).
- `gam_cuda_followup.py` → `forge_cuda_results_followup.json` — D2 (cond pushed to
  2.1e16), C2 (safe-path timing at n up to 1M).
- Host: **forge** — AMD Ryzen 5 7600X (12 threads), numpy 2.3.5 + MKL (conda;
  BLAS/LAPACK = `mkl-sdl`, confirmed by a direct `numpy.show_config(mode="dicts")`
  query on Forge — the `numpy_config_head` captured in the JSON truncates before
  the BLAS section, so it is not the evidence for this claim), NVIDIA RTX 5070 Ti
  (Blackwell sm_120, 16 GB), torch 2.11.0.dev20251226+cu128, TF32 off (asserted).
  Run under the OPERATIONS.md Forge CUDA standing allowance: nvidia-smi yield-check
  (GPU idle), `gpumice` env, throwaway `~/gam-gpu-scratch`, deleted after; GPU
  verified idle post-run. Memo numbers independently cross-checked against the raw
  JSONs (no discrepancies; two phrasings tightened in revision).

**Local (superseded sketch, kept for the record):**
- `gam_gpu_microbench.py`, `fp32_gram_check.py`, `fp32_illcond.py`,
  `qr_reparam_check.py` — Apple Silicon (CPU fp64 Accelerate + MPS fp32),
  numpy 2.4.3. Superseded by the CUDA runs per §0; retained to document the
  MPS-vs-CUDA deltas.
