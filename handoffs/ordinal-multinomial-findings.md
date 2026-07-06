# ordinal + multinomial validation — findings ledger (COMPLETE @ 4.6.9)

**DONE.** All findings resolved in 4.6.9 (released to PyPI); report blessed at
`reports/ordinal-multinomial-v4.6.9.md`. CF-1 fixed (fp64 vcov + PD gate, GPU kept).


Running ledger for the combined ordinal (`polr`) + multinomial (`multinom`)
validation. Triage per RIGOR R18 (gather vs R16 showstopper). CF-1 GPU
determination (task #4, Forge) is the potential showstopper and is still pending.

## CPU correctness — CLEAN (both surfaces)
- polr vs MASS::polr: housing (all 3 links), synth continuous+factor, collapsed
  category — coef/zeta max|Δ| 2e-7..1.4e-4, SE rel 2e-7..5e-5, loglik |Δ| ~1e-9,
  implied fitted probs 1e-7..4e-5. R15 default link=logistic exact.
- multinom vs nnet(decay=0): synth 3-class + synth 5-class + unbalanced — coef
  max|Δ| 4e-5..3.5e-4, SE rel 2e-5..6e-5, loglik ~1e-6, fitted probs ~1e-4.
  R15 default (decay=0) matches.

## Findings

### F-A — multinom `converged=True` on complete separation (R18 gather / fidelity nuance)
On complete separation (constructed: class 2 iff x>0.8), `multinom` returns
`converged=True` with runaway coefficients (|coef|~6250) and **no warning**. The
fit self-diagnoses via exploding SEs (max SE ~3.7e5, max|z| ~0.004, no p<0.05) —
the classic separation signature. nnet returns the same degenerate regime but with
a non-convergence flag (conv=1, |coef|~567). **Not silently-wrong** (the numbers
are the correct separation limit; huge SEs are unmissable). BUT: (a) inconsistent
with `polr`, which fails LOUD on separation (rejects the non-PD observed
information); (b) `converged=True` is a weaker signal than nnet's conv flag.
→ Document as limitation + raise with user at consolidation. Candidate: a
separation warning to match polr's loud posture. NOT a brakes-slam.

### F-B — polr `link=` argument surfaces as "method" in errors/info (R4 gather)
`polr(link='cauchit')` fails loud (good — no silent substitution) but the message
reads `Unknown method: 'cauchit'`. `sol.info` also carries a legacy `'method'` key
holding the link string alongside `'link'`. Amendment A3 renamed method→link; this
is a vestige. Cosmetic/consistency; gather.

### F-C — no `predict`/fitted-probability accessor on polr (R4 / limitation)
`OrdinalSolution` exposes no `predict` or fitted-probability accessor (validated by
computing implied probs from its own coef+thresholds → reproduce R's predict to
~1e-7). `MultinomialSolution` exposes `fitted_probs`/`predicted_class` but no
`predict(newdata)`. Document as a known API gap.

### F-D — multinom has no `.warnings` accessor (R4 minor)
`OrdinalSolution` has `.warnings`; `MultinomialSolution` does not. Minor
cross-surface inconsistency; gather.

## Fail-loud (G2) — CLEAN
polr + multinom both raise (no silent substitution) on: unsupported links
(cauchit/loglog → ValidationError, NOT a silent logistic fallback), l2>0 on GPU,
unknown backend, gpu_fp64 on non-CUDA (RuntimeError: MPS has no fp64), y gaps /
non-0-start / non-integer, non-finite X.

## SHOWSTOPPER (R16) — CF-1 GPU fp32 Hessian, CUDA-PROVEN silent-wrong

### F-0 — fp32 GPU Hessian inversion → silently-wrong / negative variances (R16)
**Proven on CUDA (RTX 5070 Ti, torch 2.11+cu128) AND MPS.** Both GPU paths invert
the model Hessian in the working dtype; on `backend='gpu'` (fp32) the vcov/SE come
from an fp32 inverse of the normal-equations Gram (multinom `X'WX` block Hessian;
polr autograd Hessian). On ill-conditioned designs this SILENTLY produces wrong /
NEGATIVE variances while reporting `converged=True`.

Evidence (`artifacts/.../runs/cf1_cuda.json`, cond(X) sweep, n=4000, p=6):
- **fp64 pivot is clean** (R11): gpu_fp64 vs cpu coef ~2.9e-7, SE ~2e-6..2e-4 —
  GPU fp64 == CPU. The defect is fp32 precision, not hardware/algorithm.
- **polr @ cond(X)=1e4 (X'WX~1e8): backend='gpu' returns converged=True with
  SE_relerr=1.00 and a NEGATIVE variance (negvar=True)** — while backend='gpu_fp64'
  CORRECTLY fails loud (non-convergence) at the same design. A user gets a
  silently-wrong, negative-variance SE where the safe path refuses.
- **multinom @ cond(X)=1e4: SE_relerr=0.39 black-box; white-box fp32 var 160% wrong
  + NEGATIVE variance.** Degradation is monotonic (WB var_relerr 6.8e-6 → 2.8e-4 →
  3.3e-2 → 1.60 across cond 1e1→1e4). No gate, no warning.
- Reachable via `backend='gpu'` and `backend='auto'` (auto picks GPU-fp32 on CUDA).

**Classification: R16 showstopper** (silent-wrong variance a user would trust;
fail-loud A6 bypassed; violates R12/R13). Same class as gam's CF-1.

### The remedy is DIFFERENT from gam — FIX, don't remove
Unlike gam (GPU never won → removed), **this GPU path WINS big**
(`cf1_gpu_perf_cuda.json`): polr fp32 **46× CPU at n=100k** (6.3× at 20k),
multinom **44× at n=100k** (9.8× at 20k) — GPU time ~flat while CPU grows linearly.
Removal would forfeit real value. Proposed remedy = the **regression-4.3.2 CF-1
fix**: compute the vcov Hessian/inverse in **fp64** on the GPU path (coefficients
may stay fp32 — the fp64 pivot shows they're recovered). The fp64 Hessian is a
single post-fit computation, so the large-n speedup is preserved. Plus a
conditioning gate for the truly-singular regime (refuse, matching gpu_fp64's loud
failure). **MPS wrinkle:** no fp64 on MPS → gate/refuse ill-conditioned there, or
apply the CUDA-first "MPS second-class" ruling.

**→ Needs: user authorization for a library patch (4.6.8 → 4.6.9) + PyPI release,
then RESTART validation on the new version (R16). Proposing to the user now.**

### R9 note (secondary): GPU convergence gates on L-BFGS-B `result.success`, not the
CPU score guard. Fold into the same patch review.

## FIX DRAFTED + CUDA-VERIFIED (awaiting release authorization)
Branch `fix/cf1-gpu-fp64-vcov-ordinal-multinomial` in ../pystatistics (7 files,
+187/-28, all 117 ordinal+multinom tests pass):
- **F-0**: GPU vcov now computed in float64. polr forms the observed-information
  Hessian on-device in fp64 on CUDA (host finite-diff on MPS); multinom forms its
  closed-form softmax Hessian on the host in fp64 (measured faster than re-uploading
  X for an on-device fp64 build). Inversion + PD-gate run once in fp64 on the host.
  CUDA-verified: no negative variances, SE err <2% across cond(X) 1e1..1e4.
  **GPU wins preserved: polr 45×, multinom 16× over CPU at n=100k.**
- **F-A**: multinom now raises NotPositiveDefiniteError on non-PD observed
  information (complete/quasi separation) — matching polr — instead of a silent
  pseudo-inverse with negative variances. New test added.
- **F-B**: polr "Unknown link" (was "Unknown method"). Test updated.
- **F-D**: MultinomialSolution.warnings accessor added.
- UNRELEASED.md updated (Rule 10). Awaiting user OK to release **4.6.9** (push =
  OPERATIONS item 6; PyPI publish = outward/irreversible), then revalidate on 4.6.9.
- **Runner updates needed for 4.6.9 revalidation**: run_g1_multinom `_fit`/`_separation`
  must catch NotPositiveDefiniteError (iris/fgl still return; sep_complete now raises);
  add GPU study runners (cf1 + gpu two-tier + gpu perf) as artifact generators.
