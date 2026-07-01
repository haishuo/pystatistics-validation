# `mixed` — Consolidated Recommendation (for the calling/blessing session)

**From:** the GPU R&D investigation session (chip: "can `mixed` support a GPU
backend that earns its keep?").
**Date:** 2026-06-30.
**Full evidence:** [`handoffs/mixed-gpu-investigation.md`](mixed-gpu-investigation.md)
(+ prototypes in `handoffs/mixed-gpu-investigation-scripts/`). All measured on
Forge (RTX 5070 Ti, CUDA fp64+fp32; CPU baseline same box). **No `../pystatistics`
source was modified** (Rule 8); everything below is a recommendation.

---

## TL;DR — one sentence

For the general LMM/GLMM that replaces lme4, **the answer is no GPU — fix the
CPU**; the only GPU-warranted thing is a *separate, honestly-named* low-rank/GRM
mixed model for the genomics audience, which is a **committed roadmap item to be
implemented after the CPU fixes** — not part of blessing `mixed`.

## Coordination note (read first)

Both sessions independently found the `_compute_se` inefficiency. **This session
did not modify the library.** The CPU fix is yours to own; §A below gives the
exact math so we agree on it. Nothing here competes with your work — the GPU
verdict is "there is no general-LMM GPU backend to add," which *simplifies* the
bless: `mixed` stays GPU-free like `anova`/`coxph`/`factor_analysis`.

---

## A. CPU fixes (the actual wins — your territory)

Ordered by value. All CPU, all keep R-parity / improve it.

1. **`_compute_se` (LMM) — urgent, tiny.** It currently forms the dense **n×n**
   `V* = ZΛΛ'Z' + I` and `np.linalg.solve`s it (O(n³)) — **83 % of `lmm()`
   runtime** and the reason large fits crawl. The SE needs only the **p×p** factor
   `RX` already returned by `solve_pls`:

   > By Woodbury, `X'V⁻¹X = X'X − CX'CX = RtR = RX·RXᵀ`, so
   > `Var(β̂) = σ²·(RX·RXᵀ)⁻¹`.

   The **GLMM path `_compute_se_glmm` already does exactly this**; only the LMM
   path regressed. Measured: identical to ~1e-14, **77 000×–3 400 000× faster**
   (166 s → 0.00005 s at G=2000). This alone is a Prime-Directive (R-parity) fix.

2. **Structure-exploiting PLS (the real unlock).** The deviance loop forms `Z`
   and `Z'Z` densely; this is why it OOMs (G=2000 → 27 GB; InstEval infeasible).
   Replace with a structured solve:
   - **single grouping factor:** `Λ'Z'ZΛ+I` is block-diagonal → batched per-group
     dense Cholesky. Measured ~1400× over the dense path, sub-10 ms/eval at sizes
     the library can't currently run.
   - **crossed:** needs a real sparse Cholesky (CHOLMOD-class, fill-reducing
     ordering). scipy `splu` alone hits the fill-in wall (q=15 000 → 134 s).
   This is the difference between "toy-sized only" and "usable instead of R."

3. **Autodiff gradient for the θ-optimization.** Exact gradient (torch autograd)
   → **7× fewer deviance evals** (15 vs 105), faster, same optimum. The TMB
   approach; CPU, deterministic.

## B. GPU verdict for general LMM/GLMM: **no backend**

Measured against a *correct* CPU fp64 baseline, every non-obvious formulation the
chip named loses or ties:

- **Batched per-group Cholesky** → the win is **CPU**; GPU-fp64 is ~5× *slower*
  (tiny blocks are launch-overhead-bound), GPU-fp32 only ties at huge G.
- **CG/Woodbury (crossed)** → GPU matvec is favorable, but `log|det M|` is the
  blocker; getting it needs stochastic AI-REML (a different, non-deterministic
  algorithm) — see §C.
- **GLMM/PIRLS** → just the same `solve_pls` kernel iterated; not a distinct
  opportunity.

This matches prior art exactly: nobody GPUs the lme4-style sparse-Cholesky regime
(lme4, MixedModels.jl, glmmTMB all stay CPU). **`mixed` should remain GPU-free
and expose no `backend=`** — and that is now a *documented* decision, not an
oversight.

## C. The one honest GPU opportunity (the Cox→discrete-time analogue)

There **is** a GPU-favorable formulation, but it is a **different model**, exactly
as discrete-time survival is not "Cox on GPU":

**Low-rank / GRM mixed model** — one variance component with a low-rank covariance
`K = WW'/M` (genomic relatedness / reduced-rank random effect). Its dense `M×M`
Gram makes the deviance a big dense Cholesky + `n×M` GEMMs — the cuBLAS/cuSOLVER
regime. Measured (n=20 000): **GPU-fp64 ~1.9×** over CPU-fp64 (understated on a
GeForce; more on a datacenter card), **GPU-fp32 15–72×**. CF-1 held in the
well-conditioned synthetic but a real offering needs the fp64-Gram (or
condition-gated fp32) CF-1 guard.

**Honest framing:** a `grm_lmm()` / low-rank random-effects model, scoped to the
genomics / quantitative-genetics audience, **never** described as "LMM on GPU."
Demand is real but *adjacent* to the lme4-replacement mission.
→ **Committed roadmap item: implement it *after* the CPU fixes (§A); keep it a
separate, honestly-named model — do not bundle it with the `mixed` bless.**

## D. Rejected alternatives (with reasons)

| Candidate | Verdict | Why |
|---|---|---|
| Bayesian hierarchical (MCMC) | reject | GPU-favorable but it's a PPL/MCMC engine; PyMC/NumPyro already own GPU MCMC; massive scope. |
| Penalized fixed effects | reject | CV'd ridge ≡ throws away variance-component inference (the point of a mixed model); compute already in `regression`. |
| PQL / general approximate likelihood | reject | Correctness regression lme4 itself abandoned for binary GLMMs → Prime-Directive violation. |
| Low-rank / latent-factor | **accept (separate model)** | The §C win. |

---

## Single recommended path

1. **You:** land the `_compute_se` p×p fix (§A.1) → re-bless `mixed` on a correct,
   fast CPU baseline.
2. **Roadmap, CPU (do these next, in order):** structure-exploiting PLS (§A.2,
   the big one) and autodiff (§A.3) — CPU, where the value is.
3. **Roadmap, committed — AFTER the CPU fixes:** implement the genomics-facing
   low-rank/GRM model (§C) as a *new, separate, honestly-named* model
   (GPU-warranted, with the CF-1 fp64-Gram gate). Sequenced last because the CPU
   fixes are more critical and unblock the core lme4-replacement mission first.
4. **Close the GPU chip** with verdict **(B) for general LMM/GLMM**, recording the
   deliberate no-GPU rationale + the committed low-rank roadmap item.
