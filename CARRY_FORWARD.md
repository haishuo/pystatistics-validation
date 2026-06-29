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
- **Status:** OPEN — to be cleared by the multivariate chip.

---

## Cleared
_(none yet)_
