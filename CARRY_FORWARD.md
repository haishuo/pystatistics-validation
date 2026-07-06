# Cross-module carry-forward ledger

**Purpose.** When validating one module surfaces an issue whose root cause lives in
**shared code** (a `core/compute` kernel, a GPU precision pattern, a convention) and
therefore affects **other modules too**, it must not be silently fixed in just the
current module and forgotten. The whole library is in scope during any module's work —
the "I shouldn't touch module X, this is the Y session" instinct confuses the
**sibling-repo** boundary (Coding Bible Rule 8, about *other projects*) with
**intra-library modules**, which are all fair game. Cross-module fixes are **R8**.

**The rule (RIGOR R8).** When you find such an issue:
1. **Preferred — fix it library-wide now** and ship a consistency release (R8). Don't
   leave a known shared defect latent in sibling modules.
2. **If it genuinely cannot be fixed now** (e.g. the affected module isn't validated yet
   and the fix needs that module's test bed), **record it here immediately** — before you
   forget. Each new module's chip MUST read this ledger and clear any item targeting it.

Format per item: what the issue is · where it was already fixed · which modules it likely
affects · what to verify/do · status.

---

## Open

_(none)_

---

## Cleared

### CF-1 — polr/multinom fp32 GPU Hessian band → CLOSED at 4.6.9 (GPU vcov moved to fp64; path KEPT)
- **The ordinal/multinomial instance of CF-1 materialised, and was closed by FIXING
  the path (not removing it — unlike gam).** Both `polr` and `multinom` fit the
  coefficient variance-covariance by inverting the model information matrix
  (`X'WX`-structured Hessian) in the working dtype; on `backend='gpu'` (fp32) that
  single-precision inverse silently lost precision on ill-conditioned designs. Proven
  on **CUDA** (RTX 5070 Ti): at cond(X)~1e4 (a near-collinear design the CPU/fp64 path
  fits correctly, and where `gpu_fp64` correctly fails to converge) `backend='gpu'`
  returned `converged=True` with a **100%-wrong, NEGATIVE-variance** SE for polr, and
  ~40% wrong (white-box 160% + negative variance) for multinom. **No gate.** The fp64
  GPU pivot was machine-precision vs CPU (R11) → pure fp32 defect. Treated as an R16
  showstopper mid-run.
- **Remedy differed from gam because the GPU path WINS** (polr ~50×, multinom ~120× over
  CPU at n=100k): the vcov is now computed in **float64** — polr forms the
  observed-information Hessian on-device in fp64 on CUDA (host finite-diff on MPS, no
  fp64), multinom forms its closed-form softmax Hessian on the host in fp64 — inverted
  once in fp64 with a positive-definiteness gate. The fast fp32 fit is unchanged, so the
  speedup stands. CUDA-verified: no negative variances, SE error <2% across cond(X)
  1e1..1e4. Evidence: `artifacts/ordinal_multinomial/v4.6.10/runs/cf1_summary.csv`
  (+ `v4.6.8/runs/cf1_cuda.json` defect proof). Reported in
  `reports/ordinal-multinomial-v4.6.10.md` (finding F-0).
- **Lesson refinement:** the CF-1 fp32-Gram exposure is real for ANY small-p MLE that
  inverts an `X'WX` Hessian for its SEs (not just penalized-IRLS like gam). The remedy
  is remove-vs-fix by whether the GPU path *wins*: gam didn't (removed), polr/multinom
  do (fixed with fp64 vcov). Compute the vcov Hessian in fp64 (on-device on CUDA, host
  on MPS) and gate on positive-definiteness.

### CF-1 — gam fp32 GPU Gram band → CLOSED at gam 4.6.0 (GPU path removed)
- **The gam instance of CF-1 materialised, and was closed by removing the path.**
  gam's penalized IRLS formed the normal-equations Gram `X'WX + λS` in fp32 on
  the `backend='gpu'` path. Unlike PCA (SVD-of-X, immune) this DID lose the
  smallest eigenvalues in exactly the small-λ regime the smoothing-parameter
  optimizer probes by design → silently-wrong EDF → wrong λ → wrong fit, proven
  on **both** CUDA (33.6-DOF EDF error; negative EDF at n=100k) and MPS (a
  wiggly n=1000 fit silently over-smoothed EDF 24.1→3.0). There was **no gate**.
  The GPU-feasibility investigation (`handoffs/gam-gpu-investigation.md`) found
  no shippable GPU win for GAM in any regime, so the fp32 (and the
  also-defective, slower fp64) GPU path was **removed** at 4.6.0 rather than
  gated — gam is CPU-only. Evidence: `artifacts/gam/v4.5.7/runs/cf1_mps.json` +
  the investigation handoff. Reported in `reports/gam-v4.6.0.md` (finding H1).
- **Lesson for future modules:** a module forming a normal-equations Gram
  `X'WX(+λS)` on an fp32 GPU path is a live CF-1 exposure whenever an optimizer
  visits the ill-conditioned (small-λ / near-singular) regime — the SVD-of-X
  immunity that saved PCA does not transfer. Check it, don't assume.

### CF-1 — fp64 Gram/covariance on the fp32 GPU path → CLEARED at multivariate 4.4.0
- **Cleared 2026-06-30** by the multivariate validation. The original fear — that
  PCA's GPU path forms an fp32 Gram and silently loses trailing eigenvalues — does
  **not** materialise:
  - **Default paths form no big fp32 Gram:** CPU PCA uses SVD-of-X (no Gram, immune
    by construction — the near-collinear cond~1e8 hard case recovers the 7e-9
    smallest singular value to machine precision); default CUDA = SVD-of-X; MPS
    routes to randomized SVD (only a tiny gated l×l sketch Gram).
  - **The opt-in `solver='gram'` fp32 path is gated LOUD, never silently wrong.**
    On CUDA the CF-1 boundary sweep (cond straddling the fp32 threshold) gave
    **silent_wrong_count = 0**: every accepted fp32 fit was correct vs fp64
    (subspace ~0°, sdev ~3e-6); every design past the fp32-safe boundary REFUSED
    (NumericalError) with force=True the documented override. The randomized gate
    behaves identically (R13 re-proven on CUDA + MPS). The gate is a true
    classifier — accept⟹correct, refuse⟹loud.
  - Evidence: `artifacts/multivariate/v4.4.0/runs/cuda_forge.json` (CF-1 gate
    study) + `gpu_mac_powerhouse.json` (MPS gate). Reported in
    `reports/multivariate-v4.4.0.md`.

---

## Superseded entry (kept for history)

### CF-1 — fp64 Gram/covariance on the fp32 GPU path (regression → PCA / multivariate)
- **Issue:** On `backend='gpu'` (fp32), forming the Gram/covariance matrix in single
  precision loses the **smallest eigenvalue(s)**. In regression this understated GPU OLS
  standard errors several times over — a coefficient that looked far more precise than it
  was (a silent-wrong inference).
- **Fixed in:** regression, **pystatistics 4.3.2** — the coefficient covariance is now
  computed in **double precision** (`X'X` / `X'WX`) even on the fp32 GPU path. (See the
  4.3.2 CHANGELOG entry and `reports/regression-v4.3.2.md` → `inference_se_r12`.)
- **Likely also affects:** **multivariate / PCA** — PCA *is* eigendecomposition of the
  covariance/Gram matrix, so an fp32 Gram loses trailing eigenvalues → wrong
  explained-variance / loadings / trailing components on ill-conditioned data. Possibly
  any future module that forms a Gram/normal-equations matrix on the GPU (gam, mixed).
- **To verify (multivariate chip):** does PCA's GPU path form the covariance/Gram (and
  compute eigenvalues/SVD) in **fp64**, or in fp32? If fp32, trailing eigenvalues will be
  silently wrong on ill-conditioned data — treat as a priority-1 / R13 finding; fix
  library-wide (R8) or, if a showstopper silent-wrong, R16 (stop-fix-release-restart).
  Build an ill-conditioned correctness case (R10) that would expose a lost smallest
  eigenvalue, on both MPS and CUDA.
- **Source-read finding (4.4.0, to be confirmed on PyPI + CUDA):** the **default** paths
  do **not** form a big fp32 Gram, so the original silent-wrong fear does not reach a
  default user. CPU PCA uses SVD of X directly (no Gram). Default CUDA path is SVD of X.
  MPS now routes to randomized SVD (CholeskyQR2; only a tiny gated l×l Gram of the
  sketch). The **only** fp32-Gram exposure is the **opt-in `solver='gram'`** path, which
  is **gated loud** (`NumericalError` via `_MIN_EIG_RATIO`) unless `force=True`. The
  sharp residual (R12/R13): the gate judges conditioning from eigenvalues of the
  **already-fp32-corrupted** Gram — prove on CUDA whether a silent-wrong band exists
  where corruption yields a plausible-but-wrong λ that passes the gate.
- **R18 classification:** because the exposure is a **gated, opt-in, off-default** path,
  any imperfection found is **gather-level** (candidate for a bundled 4.4.1), **not** a
  brakes-slam — UNLESS the CUDA test shows a **default-reachable uncaught wrong answer**,
  which would be a showstopper (R16). Clear or downgrade during the 4.4.0 restart.
- **Status:** OPEN — to be cleared/downgraded during the 4.4.0 multivariate validation
  (CUDA boundary test on the gated Gram path).

---

## Cleared
_(none yet)_
