# Mixed Models (`mixed`) — GPU Backend Investigation

**Date:** 2026-06-30
**Scope:** Does the `mixed` module (LMM `lmm()` / GLMM `glmm()`) warrant a GPU
backend? `mixed` is the only public pystatistics module with *no* GPU path at
all (not even CUDA). This is the standalone R&D investigation every other module
received.
**Library version under test:** `pystatistics==4.4.1` (pure fp64 NumPy/scipy;
optimizer = scipy `L-BFGS-B` over the profiled REML/ML deviance, Bates et al.
2015).
**Hardware:** Forge — NVIDIA RTX 5070 Ti (Blackwell sm_120, 16 GB), CUDA 13,
torch 2.11 cu128, fp64 + fp32. CPU baseline established on the same box.
**Rule 8 compliance:** No `../pystatistics` source was modified. All prototypes
ran in a throwaway Forge scratch dir (torn down after; GPU left clean). Every
proposed library change below is a **recommendation to surface**, not an applied
change.

---

## Recommendation: **(B) for general LMM/GLMM — no GPU backend**, with **one
narrow (A)-type exception: a *separate* low-rank / GRM mixed model** that is a
different model, not "lme4 on GPU."

For `mixed` as the general-purpose lme4-equivalent (`y ~ x + (1|group)`), a GPU
backend is genuinely unwarranted on measured evidence across all the non-obvious
formulations — **the real, high-value wins are on the CPU path.** The single
place a GPU earns its keep is a *different model class* — a low-rank /
genomic-relatedness (GRM) mixed model (§2e) — which, like discrete-time survival
vs. Cox, must be offered as its own thing and never billed as "LMM on GPU."

This is **not** "the obvious formulation doesn't parallelize, therefore no." It
is the conclusion *after* prototyping and measuring the four non-obvious routes
the brief named — batched per-group Cholesky, autodiff deviance, CG/Woodbury
low-rank, and GLMM PIRLS — each of which either loses to or ties a *correct* CPU
fp64 baseline in its intended regime, or requires switching to a different
(non-deterministic) estimator that is a research project in its own right.

The surprising fact the brief asked us to explain — *why does `mixed` have no GPU
path?* — turns out to have a sharp answer: **the current implementation is fully
dense, and that density masks two CPU algorithmic defects. Once the structure is
exploited on the CPU, the deviance evaluation drops from hundreds of
milliseconds (or out-of-memory) to single-digit milliseconds — at which point a
GPU has nothing left to win.**

---

## 1. Where the time actually goes (rock #1)

The implementation in `mixed/_random_effects.py` builds `Z` as a **dense**
`(n, q)` array with `q = Σ Jₖ·qₖ` (groups × terms). Every profiled-deviance
evaluation (`_pls.py`) then forms a dense `q×q` Gram `Λ'Z'ZΛ` and a dense `q×q`
Cholesky. For a single grouping factor `q = J` (number of groups); for a crossed
design `q = J₁ + J₂`. Both grow without bound.

**Profiling `lmm()` (single factor, random intercept, `satter=False`):**

| G (groups) | n | q | total time | peak RAM |
|---:|---:|---:|---:|---:|
| 500 | 10 000 | 500 | 3.7 s | 1.7 GB |
| 1 000 | 20 000 | 1 000 | 26.5 s | 6.7 GB |
| 2 000 | 40 000 | 2 000 | **196.6 s** | **26.9 GB** |

lme4's `InstEval` (~73 k rows, q ≈ 4 100 crossed) would need ~100+ GB and hours.
**The current `mixed` cannot run a large mixed model at all.**

**cProfile (G = 1 000, 26.3 s total) — the cost is not where a GPU would help:**

| component | time | share |
|---|---:|---:|
| `_compute_se` (dense **n×n** `solve`) | 21.9 s | **83 %** |
| θ-optimization (18 deviance evals) | 4.1 s | 16 % |

The dominant cost is a **CPU algorithmic defect** in `_compute_se`: it forms the
dense `n×n` matrix `V* = ZΛΛ'Z' + I` and calls `np.linalg.solve` on it (O(n³)),
purely to compute fixed-effect standard errors.

### 1a. The `_compute_se` defect (CPU finding, urgent)

`Var(β̂) = σ²(X'V⁻¹X)⁻¹`. By the Woodbury identity, `X'V⁻¹X = X'X − CX'CX = RtR =
RX·RXᵀ`, and **`RX` (a `p×p` factor) is already returned by `solve_pls`**. The
SE needs only the `p×p` factor, never the `n×n` solve. Measured, against the
current dense path:

| G | q | dense `n×n` SE (current) | `p×p` SE (from `RX`) | speed-up | rel. error |
|---:|---:|---:|---:|---:|---:|
| 500 | 500 | 3.04 s | 0.00004 s | 77 615× | 4.5e-14 |
| 1 000 | 1 000 | 21.99 s | 0.00004 s | 500 464× | 7.2e-14 |
| 2 000 | 2 000 | 166.50 s | 0.00005 s | 3 363 578× | 2.8e-14 |

Same answer to ~1e-14, thousands to millions of times faster. **The GLMM path
(`_compute_se_glmm`) already computes the SE the cheap `p×p` way from `RX`; only
the LMM path regressed to the `n×n` solve.** This is a one-function CPU fix,
independent of everything GPU.

---

## 2. The non-obvious formulations, measured

All torch deviance reimplementations were validated against the library's
`profiled_deviance_lmm` first: **batched and dense torch match the library
deviance to 0.0 (fp64) / 3.6e-12 (larger q), at identical θ.** The numbers below
are therefore comparing *correct* implementations.

### 2a. Batched per-group Cholesky — the most promising route (rock #3)

For **one grouping factor**, observations in different groups share no
random-effect columns, so `Λ'Z'ZΛ + I` is **block-diagonal with J blocks of size
qₜ×qₜ**. The library's dense `q×q` Cholesky is, mathematically, a *batched*
`[J, qₜ, qₜ]` Cholesky — exactly the regime GPUs are supposed to win.

**Per profiled-deviance evaluation (ms), single factor, qₜ = 2 terms:**

| G | q | library dense | dense GPU fp32 | **batched CPU fp64** | batched GPU fp64 | batched GPU fp32 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2 000 | 793 | 26 | **0.56** | 4.9 | 0.57 |
| 2 000 | 4 000 | OOM | 107 | **0.91** | 9.2 | 0.63 |
| 8 000 | 16 000 | — | — | **2.71** | 35.7 | 1.25 |

Held up at qₜ = 6 (6×6 per-group blocks, G = 8 000): batched CPU fp64 = 6.8 ms,
batched **GPU** fp64 = 36.5 ms.

**Findings:**

- **The structure-exploiting reformulation on the *CPU* is the win**, not the
  GPU: ~1 400× faster than the dense library, and it runs sizes the library
  OOMs on (G = 8 000 → q = 16 000 in 2.7 ms/eval; a full fit ≈ 20–70 evals →
  well under a second).
- **GPU loses to batched CPU fp64.** GPU-fp64 is consistently ~5× *slower* than
  CPU-fp64 — the per-group ops are too small, so kernel-launch overhead
  dominates (the textbook "many-small-fits" anti-pattern in
  `CONVENTIONS.md → When to add a GPU backend`). GPU-fp32 only edges ahead at
  very large G, and only by giving up precision for a sub-10ms → sub-2ms saving
  that is irrelevant at the scale of a whole fit.
- **CF-1 is sidestepped, not merely passed.** The CF-1 hazard is forming a large
  dense Gram in fp32 and losing trailing eigenvalues. The batched formulation
  forms *no* large Gram — only well-conditioned qₜ×qₜ blocks with an `+I` floor.
  Measured min-eigenvalue of the per-group Gram was ~1–2.7 in *both* fp32 and
  fp64 across all sizes; fp32-vs-fp64 deviance differed by only 7e-4 → 2e-2.
  There is no silent-wrong band here because the dangerous object never exists.

### 2b. Autodiff deviance (rock #4) — a CPU win, independent of GPU

The batched torch deviance is differentiable, so an **exact** gradient comes from
autograd, replacing scipy's finite differences (which cost `n_theta+1` deviance
evals per gradient). Single factor, qₜ = 3 (`n_theta = 6`), CPU fp64:

| G | finite-diff nfev | autograd nfev | FD time | AD time | θ agreement |
|---:|---:|---:|---:|---:|---:|
| 500 | 105 | **15** | 102 ms | 75 ms | 1.0e-7 |
| 2 000 | 105 | **15** | 171 ms | 74 ms | 2.5e-7 |

7× fewer evaluations, faster wall-clock, same optimum, and the advantage *widens*
with problem size. This is precisely what `glmmTMB`/`TMB` exploit (Laplace +
reverse-mode AD). It is a **CPU** recommendation, not a GPU one.

### 2c. Crossed designs & CG / Woodbury (rocks #2, #7) — the genuinely hard case

For crossed factors (e.g. students × instructors) the random-effect columns are
shared, so `Λ'Z'ZΛ + I` is **not** block-diagonal and the batched trick does not
apply. The correct CPU baseline is a sparse Cholesky (the lme4 / MixedModels.jl
approach). Two measured failure modes:

**Direct sparse factorization (scipy `splu`) hits the classic fill-in wall:**

| crossed | q | factor + solve | CG solve (iters) | matvec CPU | matvec GPU |
|---|---:|---:|---:|---:|---:|
| 1 000×500 | 1 500 | 35 ms | 3.0 ms (29) | 0.10 ms | 0.06 ms |
| 5 000×3 000 | 8 000 | 23.3 s | 18.3 ms (32) | 0.56 ms | 0.08 ms |
| 10 000×5 000 | 15 000 | **134 s** (196 M fill) | 53.5 ms (35) | 1.49 ms | 0.10 ms |

- **CG sidesteps the fill-in** entirely (the matvec `M·v = v + ZΛ'(ZΛ·v)` is
  O(nnz), never forms/factors M), and **the GPU matvec is genuinely favorable**
  at scale (0.06–0.10 ms flat vs CPU growing to 1.5 ms) — this reproduces the
  SAIGE-GPU finding (GPU wins the GRM-vector product).
- **But the profiled deviance needs `log|det M|`, and CG does not give it.** The
  only ways to obtain it are (a) the sparse factorization — i.e. pay back exactly
  the fill-in cost CG was avoiding — or (b) a **stochastic-trace estimator on the
  REML gradient** (AI-REML / BOLT-style), which abandons exact deviance
  minimization for a *different, non-deterministic* algorithm.

So a GPU *could* help the crossed regime — but only by adopting the
genomics-scale algorithm (CG + stochastic AI-REML), which is a separate research
project, gated behind a sparse-CPU path the library does not yet have, and
governed by the determinism rules (`seed`, statistical-equivalence-not-bit-equal).

### 2e. Low-rank / GRM mixed model (the one GPU-warranted formulation)

A *different model*: one variance component with a **low-rank** covariance,
`y = Xβ + g + ε`, `g ~ N(0, σ²_g·K)`, `K = WW'/M` (genomic relatedness, or any
reduced-rank random effect). Reparametrised `g = (W/√M)u`, this is a mixed model
whose random-effect design `Z = W/√M` is **dense `(n, M)`** and whose Gram is a
genuinely **large dense `M×M`** matrix — so the deviance evaluation is a big
dense Cholesky + `n×M` GEMMs, the cuBLAS/cuSOLVER regime (unlike the tiny
per-group blocks of an lme4 design). Validated M-space (Woodbury) vs direct
`n×n` V-space deviance to 1.5e-5.

**Per deviance-eval (ms), n = 20 000, varying rank M:**

| M | CPU fp64 | GPU fp64 | GPU fp32 | fp64 speed-up | fp32 speed-up | fp32 dev err | Gram min-eig 64 / 32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 46.9 | 26.5 | 1.08 | 1.8× | 43× | 1.1e-3 | 42 / 42 |
| 1 000 | 134.1 | 77.0 | 2.66 | 1.7× | 50× | 9.2e-4 | 18 / 18 |
| 2 000 | 478.9 | 251.3 | 7.39 | 1.9× | 65× | 3.9e-3 | 7.8 / 7.8 |
| 4 000 | 1 820.6 | 966.3 | 25.5 | 1.9× | 71× | 4.6e-3 | 3.2 / 3.2 |
| 6 000 | 3 987.6 | 2 172.9 | 55.7 | 1.8× | 72× | 1.5e-3 | 2.0 / 2.0 |

**Findings:**

- **GPU-fp64 earns its keep** (~1.8–1.9× for M ≥ 500), and this *understates* a
  datacenter card: the RTX 5070 Ti is a GeForce part with throttled fp64
  (~1/64 ratio); on an A100/H100 the fp64 win would be substantially larger.
- **GPU-fp32 is 15–72× faster** — the genuine "speed default with a loud escape
  hatch" GPU affordance.
- **CF-1 held here** (the Gram `c²W'W + I` is floored at ≥ 1, min-eigenvalue
  identical fp32/fp64, fp32 deviance within ~1e-7 relative). But this path *does*
  form `W'W`, so a real GRM offering still needs the CF-1 gate (form the Gram in
  fp64, or gate fp32 on the Gram condition number) — an ill-conditioned,
  collinear-marker GRM is not exercised by this synthetic.

This is the discrete-time-survival analogue: a legitimately different model that
classic sparse-design REML cannot reach at scale, GPU-warranted, and **must be
named honestly** (low-rank / GRM mixed model), never as "LMM on GPU." It is a
*new model*, not a backend for `lmm()`.

### 2d. GLMM / PIRLS (rock #5) — not a better GPU candidate

GLMM's inner loop is PIRLS = iterated penalized WLS, i.e. `solve_pls` called ~4–5
times per deviance eval. Profiling `glmm()` (G = 500, binomial): **91 % of time
in `solve_pls`** — the *same* dense kernel as LMM, just iterated — and the GLMM
SE path already uses `RX` (no `n×n` defect). GLMM is therefore strictly *more of
the same kernel*; the batched-CPU conclusion carries over unchanged. It is not a
distinct GPU opportunity.

---

## 3. Why no CUDA variant exists (rock #6)

Git history and a `grep` for `torch|sparse|gpu|backend` across `mixed/` show
**nothing** — no GPU code, no design note, no deliberate "no-GPU" decision.
`CONVENTIONS.md` explicitly lists `anova`, `coxph`, `factor_analysis`, etc. as
deliberate no-GPU modules but **does not list `mixed`**. So the absence is "never
got to it," combined with a dense implementation that never reached the scale
where the question becomes pressing. This investigation supplies the missing
rationale: a deliberate **no-GPU** decision, for the reasons above.

---

## 4. Prior art (rock #7, R13 trawl)

The literature is consistent and directly on point:

- **GPU wins for mixed models exist only where the bottleneck collapses to a
  large dense matrix-vector / matrix-matrix product inside an *iterative*
  solver.** SAIGE-GPU reports 72× per-PCG-iteration by running the implicit
  GRM-vector product as batched `gemv` on GPU; MegaLMM wins via a low-rank
  factor model. Both are the genomics large-N / low-rank regime.
- **The classic crossed-random-effects sparse-Cholesky regime has no GPU
  implementation.** lme4, Julia `MixedModels.jl`, and `glmmTMB`/`TMB` all stay
  on CPU sparse Cholesky / sparse-Hessian AD; none ships or argues for a GPU
  path. Sparse Cholesky with irregular fill-in is the canonical GPU-unfriendly
  workload.
- **`glmmTMB`/`TMB` get their leverage from autodiff** (exact gradients of the
  Laplace objective), on CPU — corroborating §2b.
- **Batched per-group Cholesky on GPU is an available *primitive***
  (cuSOLVER `potrfBatched`, MAGMA batched, `torch.linalg.cholesky`) but there is
  **no validated mixed-model package built on it** — consistent with our finding
  that, for the block sizes that arise in LMMs, the batched ops are too small to
  beat CPU.

The one regime where GPU is documented to win (genomics CG + low-rank) is exactly
the one §2c identifies as a separate, non-deterministic algorithm — not a backend
for the existing estimator.

---

## 5. RIGOR / CONVENTIONS framing

- **RIGOR priority 4 (GPU must earn its keep):** a GPU path gives up accuracy
  (fp32) and so must be *faster in its intended large-n regime* or it buys
  nothing. Measured: in every regime the GPU either loses to a correct CPU fp64
  baseline (single factor, GLMM) or requires a different estimator to be viable
  at all (crossed). It does not earn its keep.
- **RIGOR R11 (isolate precision from hardware):** the comparison was done on the
  gpu_fp64 vs cpu_fp64 pivot, on the *same* box, so the verdict is about hardware
  fit, not an fp32 accuracy artifact. GPU-fp64 lost to CPU-fp64 outright.
- **CF-1 / R12 (no silent-wrong fp32 Gram):** the only formulation that wins (the
  batched one) wins on CPU and forms no large Gram, so the silent-eigenvalue-loss
  hazard does not arise; an fp32 GPU variant would need the CF-1 gate but offers
  no benefit to gate.
- **CONVENTIONS "When to add a GPU backend":** the trigger is >~100 ms with >80 %
  in a cuBLAS/cuSOLVER-mappable kernel. After the CPU fixes, the deviance eval is
  single-digit milliseconds dominated by *many tiny* solves — squarely in the
  "Not reasonable targets: many-small-fits where launch overhead dominates"
  list. `mixed` joins `anova`/`coxph`/`factor_analysis` as a correctly no-GPU
  module — and its public API correctly continues to expose no `backend=`.

---

## 6. Recommendations to the user (CPU-first; none applied — Rule 8)

These are surfaced for decision, not implemented. They are ordered by value.

1. **Fix `_compute_se` (LMM):** compute the fixed-effect SE from the `p×p` `RX`
   factor already returned by `solve_pls` (`Var(β̂) = σ²(RX·RXᵀ)⁻¹`), matching
   what `_compute_se_glmm` already does. Removes the 83 % `n×n` hot spot;
   ~1e-14 identical results. **Smallest change, largest immediate win; also a
   Prime-Directive (R-parity) issue since the current path is far slower than
   lme4.** A high-value standalone task.
2. **Structure-exploiting PLS (the real unlock):** replace the dense `Z`/`Z'Z`
   with a structured solve — batched per-group dense Cholesky for single grouping
   factors, sparse Cholesky for crossed. This is what turns OOM/minutes into
   milliseconds and lets `mixed` handle realistic designs (InstEval-scale) at
   all. It is a **CPU** rewrite, not a GPU backend.
3. **Autodiff gradient for the θ-optimization:** exact gradient (≈7× fewer
   deviance evals), CPU, deterministic — the TMB approach.
4. **Do *not* add a GPU `backend=` to `lmm()` / `glmm()`.** Keep the general
   LMM/GLMM API GPU-free, like the other deliberately-no-GPU modules.
5. **The one honest GPU opportunity — a *separate* low-rank / GRM mixed model
   (§2e), a committed roadmap item to implement *after* the CPU fixes.** A new
   model (low-rank random effects / genomic relatedness) whose dense `M×M` Gram is
   genuinely GPU-favorable: measured GPU-fp64 ~1.9× (more on a datacenter card),
   GPU-fp32 15–72×, with a CF-1 fp64-Gram gate. Serves the genomics /
   quantitative-genetics audience, **not** the lme4 user, and must be named so it
   is never mistaken for "LMM on GPU." Sequenced after items 1–3 (the CPU fixes
   are more critical and unblock the core mission first). Its sparse/iterative
   cousin (CG + stochastic-trace AI-REML, SAIGE/BOLT-style) reaches the same
   audience via a harder, non-deterministic route; lower priority.
6. **Rejected alternatives:** Bayesian hierarchical MCMC (scope; PyMC/NumPyro own
   GPU MCMC), penalized fixed effects (CV'd ridge discards variance-component
   inference; compute already in `regression`), PQL / general approximate
   likelihoods (a correctness regression lme4 itself abandoned — Prime-Directive
   violation).

---

## Appendix — reproduction

Prototypes (throwaway, run on Forge `gpumice` env, scratch torn down):

- `01_profile_baseline.py` — profile `lmm()`/`glmm()`, find the dense scaling wall.
- `02_dissect.py` — `_compute_se` defect: `p×p` RX vs `n×n` solve, timing + correctness.
- `03_batched_deviance.py` — batched per-group deviance; validation vs library;
  CPU/CUDA × fp64/fp32 sweep; CF-1 eigenvalue check.
- `04_crossed_and_autodiff.py` — crossed sparse CPU baseline; autograd vs finite-diff.
- `05_cg_crossed.py` — CG/Woodbury crossed: fill-in wall, CG solve, GPU matvec, log-det blocker.
- `06_glmm_profile.py` — GLMM/PIRLS profile (same kernel as LMM).

All deviance reimplementations validated against `pystatistics.mixed._deviance.
profiled_deviance_lmm` at identical θ before any timing was trusted.
