# Validation rigor — the rules every module report must satisfy

Binding methodology for the pystatistics validation program. Every per-module
validation (and the chip that performs it) MUST follow these. Companion to
`ARCHITECTURE.md` (where evidence/reports live) and the library's
`CONVENTIONS.md` (the API constitution). Treat every module **as if you were
publishing a paper on it** — that discipline is what surfaces the defects that
"just getting it written" hides.

## R1 — Parity with the reference means SPEED too, not just numbers

Numerical agreement with R/SAS is necessary but **not sufficient**. Every module
must also compare its **performance and asymptotic complexity** against the
reference across a **range of problem sizes** — never a single size, which hides
the slope.

- **Measure the empirical complexity.** Run the fit across a spread of `n` (and
  `p` where relevant) wide enough to reveal the scaling slope; observe/estimate
  the empirical exponent and compare it to the reference's. A **complexity-class
  gap** (e.g. pystatistics O(n²) while R is O(n)) is a **DEFECT**, full stop —
  investigate it, do not record the slower number and move on.
- **A large constant-factor lag is also a red flag.** If pystatistics trails the
  reference by a wide margin at scale, that is something to investigate as *not
  right*, not a fact to report neutrally.

> **Canonical incident (why this rule exists):** pystatistics `coxph` was O(n²)
> while R's `survival::coxph` is O(n). It nearly shipped — the validation session
> would have blown right past it — because performance wasn't checked across
> scales. It was caught by hand. **That must never depend on luck again.**

## R2 — A legitimate slowdown is DOCUMENTED, never swept under the rug

pystatistics is sometimes slower for a *real reason* — it computes something the
reference skips (e.g. a full Hessian / standard errors R does not return), uses a
more robust or more general algorithm, or pays for a correctness guarantee. When
so:

- **Name the extra work explicitly** in the report (the performance / limitations
  section): what pystatistics computes that the reference does not, and why.
- **Quantify it** — how much of the gap the extra work accounts for.
- **Flag it for discussion** if the cost is large; do not bury it. A documented,
  justified slowdown is acceptable; an unexplained one is a defect under R1.

The test for every speed gap: *can we state, in one sentence a reviewer would
accept, why we are slower?* If not, it's R1 (a defect), not R2 (a justified cost).

## R3 — Scaling study is mandatory, calibrated to the module

Each report includes a scaling study sized to reveal R1:

- **GPU-warranted modules** (per `CONVENTIONS.md`'s "when to add a GPU backend"):
  CPU vs MPS vs CUDA device pivots across `n`/`p`, plus pystatistics-vs-R.
- **CPU-only modules** (the constitution deliberately gives some none — e.g.
  `coxph`, `anova`, `timeseries.ets`): the scaling study is **pystatistics CPU vs
  R across `n`** — this is precisely where complexity gaps like the coxph one
  surface. Do **not** manufacture a GPU showcase where the constitution says there
  is no GPU path; an honest "no GPU path, and why" is the correct result.

## R4 — Constitutional-compliance audit

Verify the module's public surface conforms to `CONVENTIONS.md`: parameter names
and the selector taxonomy (`backend`/`family`/`link`/`method`/`solver`), the
`backend=` device+precision encoding (no `use_fp64`), `…Solution` result objects
and uniform accessors (e.g. `.z_values` vs `.t_values` per Amendment A3),
`core.exceptions` error types. Report any deviation as a finding to fix under the
consistency standard.

## R5 — PyPI-only, render-from-artifacts, version-pinned

Validate the **PyPI-released** version (never a local checkout; `require_pypi`
enforces). Numbers come only from frozen artifacts via `render_report.py` — never
hand-typed. A report is immutable per library version.

## R6 — Baseline → optimize → re-validate

When validation finds an inefficiency (R1) or a justified-but-improvable cost
(R2) and a fix is shipped, the fix is **released to PyPI** and the module is
**re-validated at the new version**, with the report documenting baseline→optimized
(as `regression` did: v3.18.0 baseline → v3.20.0 optimized). Library changes obey
`CONVENTIONS.md` + the Coding Bible + `.release/UNRELEASED.md`, and respect any
release-hold coordination in `ROADMAP.md`.

## R8 — Cross-module consistency discoveries: pause + library-wide minor release

Validating one module routinely surfaces issues that are not local to it — a naming
or API inconsistency, a shared `core/compute` defect, a convention drift — that
affect **more than one module**. When that happens:

- **Pause** the current module's validation work.
- **Fix it library-wide and ship a consistency minor release** for the whole library
  (not buried inside one module's optimization). This is exactly what **4.1.0** was:
  a multi-module consistency fix that the survival work surfaced, spun out as its own
  release so survival could then continue toward 4.2.0.
- **If the fix is breaking, STOP and discuss with the user first.** A breaking change
  is not a quiet minor release — it is a deliberate, discussed decision (cf. the 4.0
  consistency release).
- A defect local to the single module under validation is handled in-module (R1/R6),
  not via a library-wide release.

This keeps module work from silently absorbing cross-cutting fixes, and keeps the
library coherent (one standard, applied everywhere) rather than patched per-module.

## R9 — fp32 GPU non-convergence is a false-negative until proven otherwise

CPU paths are fp64; MPS/CUDA fp32 paths are not. A convergence criterion tuned to
R's fp64 default (e.g. **absolute** `|Δβ| < 1e-8`) is **unreachable in fp32** — not
because the fit is wrong, but because fp32 has a noise floor that scales with
coefficient magnitude. A correct fp32 fit can converge to the right coefficients in
a handful of iterations and then *plateau* at that floor (e.g. `max|Δβ| ≈ 1e-3` for
coefficients of magnitude ~12), burn all remaining iterations, and falsely declare
non-convergence.

**Before recording any GPU/MPS path as failing, numerically unstable, or
"ill-conditioned," rule out the false-negative convergence test:**

- Compare the fp32 coefficients against the fp64 fit. If they agree to the **fp32
  tier** (e.g. `max_rel ~1e-6`), the fit is **correct** — the convergence *test* is
  wrong, not the math.
- A convergence check must be evaluated at a tolerance appropriate to the **working
  precision** (relative, fp32-tier), not R's absolute fp64 tolerance.
- An error message like "GPU Cholesky is unstable" on such a fit is a
  **misdiagnosis** — report it as the convergence-logic defect it is, not as a
  numerical-stability problem.

**Do not blame the GPU backend without cause.** "It doesn't converge on MPS" is a
hypothesis to test against the fp64 reference first, not a conclusion. This is GPU
GLM convergence logic in general — not specific to any one module or model type.

## R7 — The seven questions

Every report answers them (see `ARCHITECTURE.md` / the template): procedure,
algorithms, comparison to reference, tolerances, benchmarks (incl. the R1/R3
scaling + complexity analysis), known limitations (incl. any R2 documented
slowdown), and the exact version validated.
