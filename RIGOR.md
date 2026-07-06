# Validation rigor — the rules every module report must satisfy

Binding methodology for the pystatistics validation program. Every per-module
validation (and the chip that performs it) MUST follow these. Companion to
`ARCHITECTURE.md` (where evidence/reports live) and the library's
`CONVENTIONS.md` (the API constitution). Treat every module **as if you were
publishing a paper on it** — that discipline is what surfaces the defects that
"just getting it written" hides.

## What we validate — the three foundational guarantees, in strict order

PyStatistics makes the user three promises — **Correctness, Fidelity, Performance** — in
this order of importance. **A failure high in this list is not bought off by success
lower down.** Everything else in this document (the red-team, the scaling studies, the
GPU bar, R1–R18) is *method* — the machinery that proves or enforces these three.

### Guarantee 1 — Correctness
**Never return a result that falls outside the documented correctness contract. If
correctness cannot be guaranteed, fail loudly rather than returning a result.**

The contract is per-module: agreement with the mathematical reference (R) to a *stated
tolerance* (the `tolerances` contract). Floating point need not be **exact** — it must
**satisfy the contract**. "Looks close" is not the bar; meeting the documented level is.
A result that violates the contract — or is wrong while looking right — is the gravest
failure there is; when correctness cannot be guaranteed, the only acceptable output is a
**loud failure, never a number**. A contract-violating result that reaches a user is a
showstopper (R16): stop, fix, re-release, restart. The red-team (R10) and the
no-silent-wrong proofs (R9/R12/R13) stress this where it is most likely to break.

### Guarantee 2 — Fidelity
**Never satisfy Guarantee 1 by silently solving a different problem.**

Do not silently change the **estimator, backend, precision, solver, or approximation**
the user asked for. If the user requests `backend="mps"` and MPS cannot satisfy
Guarantee 1, do **not** quietly fall back to CPU to produce a number — **raise**. Getting
"the right answer" by quietly computing a *different* thing than the user requested is the
violation this guarantee names (the A6 "PyStatistics does not help" amendment, elevated
to a foundational guarantee). The one non-violation: a **default call where the user
expressed no preference** — there the library may auto-select the best
validated-equivalent path, provided the choice is **disclosed** (e.g.
`solution.backend_name`). Default-selection-when-disclosed is not silent substitution;
overriding or swapping an *explicit* request is.

### Guarantee 3 — Performance
**Subject to Guarantees 1 and 2, make every reasonable effort to match or exceed the
reference implementation on equivalent hardware using the same statistical estimator.**

"Equivalent hardware, same estimator" is the apples-to-apples frame — an M2 Max is not
expected to beat an H100, and a CPU is not measured against a GPU. If R fits an LM in
0.01 ms and PyStatistics takes 0.1 ms on the same hardware with the same estimator, that
gap **must be justified by a material benefit** — a more accurate or robust result,
quantities R does not compute, a correctness/stability guarantee. **If the benefit is
"nothing," it is a defect** (R1/R2/R6). Proven with a **speed comparison across problem
sizes**, not a single point (a complexity-class gap, R1, is its most severe form). A
*different, faster* estimator or *different* hardware (GPU) is a separate legitimate win,
not a violation — and a faster *validated* implementation (even on a heavier dependency,
per `CONVENTIONS.md`'s dependency-tiering amendment) is preferred over a slower one,
never the reverse "for packaging reasons."

### The GPU corollary to Guarantee 3
The GPU is Guarantee 3 on *different hardware*, held to its own bar: Guarantees 1 & 2
still bind absolutely (it must never crash or be silently wrong — A6/R9/R12 — and an
explicit GPU request that cannot satisfy Guarantee 1 must **raise, not fall back**); it
must reach the claimed fp32 tier; and it must actually be **faster than the CPU in its
intended large-`n`/`p` regime** — a GPU that ties or loses there has no reason to exist
(flag it; fix or remove it). Small-`n`/narrow-`p` parity is expected (dispatch overhead)
and fine. We are not chasing peak FLOPs — just a non-embarrassing, genuinely useful
speedup; an honest "no GPU path, and why" is a correct result. Do not manufacture a showcase, and do not ship a
GPU path that never wins.

**CUDA is first-class; MPS is second-class — test CUDA FIRST, always.** The Mac is
MPS-only, but MPS is a developer convenience; **CUDA is the datacenter target the
regulated-buyer trust story rests on.** Any GPU-involved code MUST be validated on
CUDA (Forge) before MPS, in this strict order: **(1)** CUDA must work, period — a GPU
defect on CUDA is catastrophic and blocks the release; **(2)** once CUDA works, check
MPS; **(3)** if MPS does not work, make a best-effort fix; **(4)** if best-effort
doesn't do it, dive; **(5)** worst case, eliminate MPS. A pure-MPS failure is
acceptable (and sometimes an MPS path does not exist at all — e.g. no fp64 on Metal);
a CUDA failure is not. **Never bless a GPU path, or run the CF-1 fp32-Hessian check,
on MPS evidence alone** — MPS is suggestive, Forge/CUDA is the authoritative check
(cf. the gam GPU-investigation lesson: never substitute MPS/carried evidence for a
CUDA question). This ordering is why the ordinal/multinomial CF-1 showstopper (4.6.8→
4.6.9) was proven and fixed on CUDA first, with MPS following.

## R1 — Parity with the reference means SPEED too, not just numbers

Numerical agreement with R/SAS is necessary but **not sufficient**. Every module
must also compare its **performance and asymptotic complexity** against the
reference across a **range of problem sizes** — never a single size, which hides
the slope.

**Scope: this strict mandate is the CPU path (Guarantee 3) — pystatistics on CPU must
never lag R.** The GPU (the Guarantee-3 corollary) is held to its own bar: it need not beat R, but it
**must beat the CPU in its intended large-`n` regime** (a GPU that ties or loses to the
CPU has no reason to exist — it already gave up accuracy for speed), must reach claimed
fp32 accuracy, and must never be silently wrong. Read the rest of R1 as the CPU
contract.

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

Each report includes a scaling study sized to reveal R1.

**The CPU-vs-R-across-sizes study is MANDATORY for every module** (Guarantee 3): fit
across a spread of `n` (and `p` where relevant), from small `n` (where fixed Python
dispatch overhead bites — the place pystatistics is most likely to lag) through large
`n`, comparing **pystatistics CPU vs R**. A single-point speed comparison does not
satisfy this. Any size at which pystatistics lags R must be explained per Guarantee 3 /
R2 (what is the user buying?) or fixed.

- **GPU-warranted modules** (per `CONVENTIONS.md`'s "when to add a GPU backend"): add
  CPU vs MPS vs CUDA device pivots across `n`/`p` **on top of** the mandatory CPU-vs-R
  study. The GPU pivots exist to show the GPU bar (Guarantee-3 corollary): no crash, claimed accuracy,
  **and a genuine speedup over the CPU in the intended large-`n` regime** (a GPU that
  ties/loses to the CPU there is a finding, not an acceptable result) — but **not** to
  prove the GPU beats R. When you compare a GPU number against R, isolate precision from
  hardware (R11). Do not let a GPU speedup headline stand in for the CPU-vs-R evidence
  Guarantee 3 requires.
- **CPU-only modules** (the constitution deliberately gives some none — e.g.
  `coxph`, `anova`, `timeseries.ets`): the CPU-vs-R study **is** the scaling study —
  this is precisely where complexity gaps like the coxph one surface. Do **not**
  manufacture a GPU showcase where the constitution says there is no GPU path; an
  honest "no GPU path, and why" is the correct result.

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
hand-typed. A report is immutable per library version. **"Immutable per version"
governs how we record an *accepted* version's numbers — it is NOT licence to keep
validating a version you have found to be broken; see R16.**

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

**The whole library is in scope — modules are not silos.** The "this is the *X*
session, I shouldn't touch module *Y*" instinct confuses the **sibling-repo** boundary
(Coding Bible Rule 8 — *other projects*) with **intra-library modules**, which are all
fair game during any validation. A fix in shared code (`core/compute`, a GPU precision
pattern, a convention) that affects more than one module is **R8** — do not quietly fix
it in only the current module.

**Fix-now-or-log — never forget a cross-module implication.** When a fix is applied in
one module's context but the underlying defect is in shared code affecting others:
1. **Preferred: fix it library-wide now** (this rule).
2. **If it genuinely cannot be done now** (e.g. the affected module isn't validated yet
   and the fix needs its test bed), **record it in `CARRY_FORWARD.md` immediately.** Every
   new module's chip MUST read that ledger and clear any item targeting it. The canonical
   example: the 4.3.2 fp64-Gram GPU-SE fix landed in regression but PCA (eigendecomposition
   of the same Gram) likely needs it too — logged as CF-1.

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

> **Debugging heuristic (G3):** a *math* error and a *stop-criterion* error present
> **identically** (non-convergence / wrong-looking output) but have **opposite** fixes
> — one corrects the arithmetic, the other corrects/relaxes the convergence test.
> Diagnose which before fixing: the coxph/discrete-time false-negative was a
> stop-criterion bug masquerading as numerical instability.

## R10 — Correctness grids must include the hard cases, and match R's failures and warnings

A grid of well-behaved textbook datasets (a handful of coefficients, no degeneracy)
proves only "matches R when things are easy" — which is exactly what a skeptic or an
auditor will not probe. The claim worth earning is **"matches R when things are
hard, including matching R's own failures and warnings."** Every module's correctness
grid must therefore reach into the adversarial regime where R-agreement is most likely
to break *and* where R itself changes behaviour:

- **Near-collinear / ill-conditioned designs** — agree with R right up to the
  documented refusal boundary (e.g. the condition-number cutoff), then refuse loudly.
- **Degenerate likelihoods** (e.g. logistic separation) — replicate R's *behaviour*:
  the divergence **and** the warning it emits, not just the coefficients.
- **Factor / contrast coding** — tested in isolation, not entangled with another gap.
- **Weights and offsets.**
- **Rank-deficient / aliased designs** — match R's handling (column-drop → `NA`
  aliasing), not a different silent answer.

Matching R's *numbers* on easy data and matching R's *failures and warnings* on hard
data are two different claims. The report must make — and substantiate — the second.

## R11 — GPU benchmarks must isolate precision from hardware, and name the reference BLAS

A headline speedup that compares fp64-CPU-R against fp32-GPU-pystatistics confounds
two independent effects: the fp32-vs-fp64 win and the GPU-vs-CPU win. Such a number is
not wrong, but it is **not attributable** — and a reviewer will discount it.

- **Isolate the hardware effect** with a same-precision pivot: `gpu_fp64` vs
  `cpu_fp64` (on CUDA) measures hardware alone. Report it alongside the bundled number,
  not instead of it.
- **The CPU-vs-R column is already same-precision** (both fp64) — keep it as the clean
  comparison.
- **Name the BLAS R was linked against** (reference vs OpenBLAS/MKL). It moves the CPU
  baseline materially; an unstated BLAS makes the CPU-vs-R gap uninterpretable.

## R12 — Relaxing a fail-loud guarantee into a convergence claim demands a no-silent-wrong proof

This is the **complement to R9**. R9 guards against the *false negative* — refusing a
correct fp32 fit. The opposite failure is worse: a relaxed gate that **accepts a
biased fp32 fit** — a silently-wrong answer. fp32 can converge to a wrong optimum that
still passes a loose Newton-decrement / tolerance test, and **"converges on the tested
grid" does not prove "never converges-wrong-without-failing-loud" off it.**

Whenever a path is changed from fail-loud to "converges to ~X" (as the plain fp32
log-link was at 4.2.3), the report must carry a **dedicated adversarial stress test
that the accept/refuse boundary is principled — that there is no silent-wrong zone in
the gap:**

- Probe designs straddling the precision floor (ill-conditioned, near-separation,
  large-coefficient) where the gate must decide.
- **Force the accepted fit and verify against fp64** — an accepted fit must be correct
  to the fp32 tier.
- **Confirm refusal fires *before* accuracy degrades** — there must be no band of
  inputs that are accepted yet wrong.

R9 and R12 together require the gate to be a **true classifier in both directions**:
never refuse a correct fit (R9), never accept a wrong one (R12). A speed win that
relaxes a guarantee is the single most likely place a correctness regression hides —
treat it as such.

## R13 — Guarantees are regime-conditional; do not globalize an inherited guarantee

A correctness guarantee (e.g. the R12 fp32 no-silent-wrong gate) is proven for the
*input regime* it was tested on — it is **not** a property of the code in the abstract.

- **Forwarded / inherited guarantees must be re-proven on the new regime.** When module
  A forwards to module B (`survival.discrete_time` → `regression` GLM), A does **not**
  inherit B's guarantee for free. discrete-time's regime — heavy low-weight
  interval-dummy blocks, a separation-prone default — is adversarial in ways B's generic
  validation never stressed. Re-run the R12-style boundary on A's *own* regime.
- **A novelty claim earns MORE scrutiny, not less.** "First discrete-time survival on
  GPU at scale" is the path to stress hardest, plus a prior-art trawl (arXiv / PyPI /
  GitHub) before it appears in a paper.
- What **does** generalize is the *methodology* — the correctness taxonomy, the
  debugging heuristics (G3), the principles — not the specific numeric guarantee. Be
  careful which lessons you globalize.

## R14 — Put the guarantee on the stable, version-independent layer

Separate the version-*sensitive* implementation from the version-*independent*
guarantee. The fp32/MPS solver is torch-version-sensitive; the host-fp64 acceptance
gate is not. The guarantee must live on the stable layer so it **survives a dependency
upgrade — your guarantee must not break when torch updates.** Every report must:

- **name the validated dependency version** (e.g. torch 2.12.1) for the
  version-sensitive path, and
- **state the invariant that holds across versions** (the fp64 gate keeps a wrong answer
  loud regardless of the torch build).

This is directly load-bearing for the regulated-buyer trust story: a guarantee that
silently lapses on a dependency bump is not a guarantee.

## R15 — Validate the DEFAULT invocation, not just the expert case

Validate the call a naive user actually makes — the **defaults** — not only the tuned
case you know is meaningful. A known-degenerate default (e.g.
`discrete_time(intervals=None)` → every unique event time → perfect separation on
continuous data) must **fail loud or warn**, never silently return separated garbage
with huge coefficients. "We validated the meaningful 5-bin case" does not cover the
footgun a user triggers on day one. The default's behaviour on the regime it will
actually meet is a first-class correctness claim (Guarantee 1).

## R16 — A severe (showstopper) bug found mid-run: STOP, fix, release, restart

Validation exists to protect the user from wrong answers; **completing a run is never
more important than that.** When a run surfaces a **severe** bug — by best judgement, a
correctness/showstopper defect: a silently-wrong ("quiet-wrong") result, numbers a user
would trust but shouldn't, a violated fail-loud guarantee (A6), data corruption — the
version under test is **dead on arrival.** Do not rationalize continuing; "the report
is frozen / immutable / PyPI-pinned" governs an *accepted* version (R5) and is never a
reason to bless a known-broken one.

**The procedure (slam the brakes):**
1. **STOP the validation run immediately** — no further validation proceeds on the
   doomed version.
2. **Surface it to the user at once** — the stop, the diagnosis, the severity call.
3. **Fix it as the top priority,** on a branch, with whatever testing the fix demands
   (library changes obey `CONVENTIONS.md` + the Coding Bible + `.release/UNRELEASED.md`).
4. **Cut a patch release** (e.g. 4.3.2 → **4.3.3**) and **publish to PyPI** — through the
   normal release authorization (publishing is outward/irreversible; the user authorizes
   it per `OPERATIONS.md`; never silent).
5. **Discard / mark superseded** the doomed version's partial artifacts — never commit a
   report that blesses a version you know is broken.
6. **RESTART validation from the new version.**

**Severity is a judgement call:**
- **Showstopper → stop-fix-release-restart:** anything in Guarantee-1 correctness that
  misleads a user (quiet-wrong, wrong-but-precise-looking, fail-loud bypassed). A
  complexity-class regression bad enough to make a path unusable also qualifies.
- **Not a showstopper → log + handle in the normal R6 cycle, continue:** a defect that
  already **fails loud** (no silent wrong), a performance nit with no correctness impact,
  missing-but-fails-loud functionality (e.g. unimplemented stratified Cox), cosmetics/docs.
- **When in doubt, treat it as severe and surface to the user.** A false stop costs
  time; a false continue ships wrong answers.

**Interactions:** if the fix is **breaking**, STOP and discuss — a breaking change is a
deliberate decision, not a quiet patch (cf. R8). If a **release-hold** is active, a
showstopper likely forces the issue — surface the conflict; the user decides. R16 is the
**severity** trigger; R8 is the **breadth** (multi-module) trigger; both can fire at
once. R6 is the routine baseline→optimize→re-validate flow; **R16 is its emergency-stop,
correctness-driven sibling.**

## R17 — Datasets: one centralized store, HDF5 only

Validation datasets live in **one place, in one format**: the centralized store
`Dev/datasets/` (Forge mirror `/mnt/data/pystatistics-datasets`), reached via
`MVNMLE_DATA_DIR`, as **HDF5 (`.h5`)** files written by `dataset_writer.py` and
documented in `SCHEMA.md`, with checksums in `MANIFEST.sha256`. Drivers **load from the
central store** — they do not carry their own copies.

- **No new CSV/parquet/ad-hoc copies in `drivers/*/data/`.** Any dataset a module needs
  is added to the central store as HDF5 (writer + SCHEMA entry + MANIFEST), then loaded
  from there. "Datasets spread across a dozen formats and directories" is the failure
  mode this rule exists to prevent.
- **Real reference data is curated, deterministic, and checksummed** (R6/determinism):
  the same bytes feed pystatistics, R, and every host.
- **Known stragglers to migrate (cleanup):** `drivers/regression/data/*.csv` and
  `drivers/survival/data/*.csv` predate this rule and duplicate `.h5` files already in the
  central store — fold them into the store and load from `MVNMLE_DATA_DIR` when those
  modules are next touched.

## R18 — Triage every finding: showstoppers slam the brakes, the rest are gathered into one bundled patch

R16 is the emergency stop; R6 is the routine fix-and-re-validate. R18 is the **triage
rule that decides which one applies** and prevents version-thrash from minor defects.

> ### Bless precondition (non-negotiable, no exceptions)
>
> **A report may be blessed only when its findings ledger contains no open fixable
> finding.** Every finding is either (a) **fixed** in the version the report blesses, or
> (b) a **true permanent documented limitation** whose rationale has been argued and
> accepted as such (see below) — there is no third state. **"Documented" is not a state a
> finding can rest in.** Documenting a bug does not fix it, and nothing on this Earth
> makes a report with a known, fixable defect blessable — not that the defect is "bounded,"
> not that it is "off the default path," not that it is "gather-class," not that it is
> "disclosed," not that it is "conservative," not that a note describes it. There is no
> "technically blessable with the findings documented." That sentence is forbidden. If you
> catch yourself reaching for it, you have already failed the rule: **stop, fix the
> finding, and only then bless the version that fixes it.**
>
> Severity (R16 vs gather) decides *when* a finding is fixed — now or in the bundle — never
> *whether*. The bless gate is the same for both: fixed, or accepted as a permanent
> limitation. Nothing else opens it.

Every defect found during a validation pass is classified at the moment it is found:

- **Showstopper → R16, immediately.** A correctness defect that would mislead a user:
  a **silent failure**, an **uncaught wrong answer** (wrong-but-trusted, wrong-but-
  precise-looking), a bypassed fail-loud guarantee (A6), data corruption, or a
  complexity-class regression bad enough to make a path unusable. The version under
  test is **dead on arrival**: STOP all validation, surface, fix, cut a patch, publish
  (with authorization), discard the doomed version's partial artifacts, restart from the
  new version. **No validation proceeds on a doomed version.** When in doubt, treat as
  showstopper (a false stop costs time; a false continue ships wrong answers).

- **Non-showstopper → GATHER, do not stop.** An inaccuracy or defect that **already
  fails loud** (no silent wrong), a numerical tolerance miss on a gated/opt-in path a
  default user does not reach, a documented-and-bounded approximation, a performance nit
  with no correctness impact, cosmetics/docs. These do **not** each trigger a stop or
  their own release. **Log each to the run's findings ledger and continue the pass.**

**"Gather" means DEFERRED, not EXCUSED.** Gather is a scheduling decision — *don't slam
the brakes for this one* — **not** a verdict that the finding is acceptable. Every gathered
item is a thing we have decided to **fix**; the only question R18 answers is *now (R16) or
in the bundle (R18)*. "We documented it" is **never** the remedy. A deviation-from-the-
reference that a disclosure note merely *describes* is still an open defect: the note buys
time to batch the fix, it does not close the finding. Writing "documented convention
difference" in a report and moving on — as if disclosure discharged the obligation — is the
exact failure this rule forbids. If a gathered item is genuinely *not* going to be fixed
(a true, permanent, justified divergence — e.g. a deliberate fail-loud choice R does not
make), it is not a "gather" at all; it is a **documented limitation** with its rationale,
and it must be argued as such, not smuggled in under the gather label.

**The bundle.** After the pass, the gathered non-showstoppers are **fixed** together and
shipped as **one patch release** (e.g. 4.4.0 → **4.4.1**) — not a release per defect, and
not "left documented." The published, blessed report is then the **bundled version**:
re-validate the affected surface at the new patch (R6's baseline→optimize→re-validate,
batched). A validation pass may legitimately *complete* on a version carrying known
non-showstopper findings, **provided** the final blessed report is the bundle that fixes
them — never a report that silently blesses known inaccuracies (Rule 9 / R5), and never one
that blesses them *loudly* either (a disclosed-but-unfixed deviation is still unfixed).

The line is correctness-to-the-user, not severity-of-effort: *would a user trust a wrong
number because nothing told them not to?* If yes → showstopper/R16. If the system already
refuses, warns, or the inaccuracy is bounded and off the default path → gather/R18 — **and
then the bundle fixes it.**

## R7 — The seven questions

Every report answers them (see `ARCHITECTURE.md` / the template): procedure,
algorithms, comparison to reference, tolerances, benchmarks (incl. the R1/R3
scaling + complexity analysis), known limitations (incl. any R2 documented
slowdown), and the exact version validated.
