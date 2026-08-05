# pystatistics close-out — chip brief (finish the library → 5.0 → whole-library bless)

**Self-contained brief for the session(s) that finish `pystatistics` after A0.** A0
(mvnmle + mice re-validation) is DONE and blessed at 4.6.13. This brief covers the rest:
implement the confirmed capability gaps, document the confirmed carve-outs, run the pre-launch
consistency sweep as **5.0**, and earn the single whole-library bless. **Read
`handoffs/final-bless-completeness-audit.md` first** — it holds the per-item detail, evidence
citations, and the VERIFY-sweep code findings; this brief is the execution wrapper.

## The governing rules (locked decisions)
- **Sequencing (user, 2026-07-08):** `pystatistics` is finished COMPLETELY — the implement list
  below + the 5.0 pre-launch sweep + the single whole-library bless — **before any downstream
  `pystats*` vertical (PyStatsBio, …) is validated.** No downstream package validates against a
  moving API. (Recorded in `pystatistics/CONVENTIONS.md` → "Versioning policy".)
- **Parity bar (the Prime Directive):** the **math must match R**; the **interface may** differ,
  and you may omit an R capability **only** with a GOOD reason that is EXPLAINED and DOCUMENTED
  (in `CONVENTIONS.md`). Undocumented absence is not allowed. Implement anything a working
  statistician routinely reaches for; JUSTIFY→B only if genuinely specialist / R-farms-it-out /
  superseded by a better path we offer; tie-break → implement; never leave named-but-broken.
- **RIGOR (`RIGOR.md`) binds every validation:** R1/R3 CPU-vs-R across sizes (CPU must match AND
  match/beat R's speed); R10 hard cases (match R's failures/warnings, not just easy numbers);
  R4 constitutional audit; R5 validate the PyPI release, render reports from frozen artifacts;
  R12/R13 fp32 no-silent-wrong gate on any GPU path, **proven on CUDA (Forge) FIRST**; R15
  default-invocation; R16 (showstopper → stop-fix-release-restart) vs R18 (gather → one bundled
  patch; bless precondition = no open fixable finding). Coding Bible + `CONVENTIONS.md` also bind.

## Scope — the locked implement/justify list (finalized 2026-07-08)

**IMPLEMENT (mainstream; each is its own implement → test → R-validate → release cycle):**
`A1+VA-8` survival cluster (stratified Cox/KM + `(start,stop)`/time-varying Cox + left-trunc KM +
`cox.zph` + robust/cluster SE) · `VA-1` gam tensor smooths `te()`/`ti()`/`s(x,z)` · `VA-4`
timeseries `xreg`+drift (`+VA-4b fixed=`) · `VA-2` gam `by=` · `VA-3` gam usable `nb()` (estimate
θ) · `A3` gam `cc`/`ps` bases · `A2` gam GLM-family analytic sp-gradient · `VA-5` regression
`link=` cloglog/cauchit/sqrt + families inverse.gaussian/quasipoisson/quasibinomial · `A4` glmm
offset/prior-weights/aggregated-binomial · `VA-6a` regression `anova`/`drop1` + diagnostics
(Cook's/hat/residuals) · `VA-6b` regression profile-likelihood CIs (`confint.glm`) · `A5` polr
`loglog`/`cauchit` links · `A6` glmm `is_singular` surface · `VA-7` glmm `||` uncorrelated RE ·
`VA-10a` FA factor scores (`scores=`) · `VA-11` anova ω²/Cohen's d + Games-Howell · `A8` survival
Breslow-ties validation row + owed prior-art trawl · `A9` timeseries 4.6.12 re-bless.

**MEASURE-THEN-DECIDE:** `A7` montecarlo BCa on balanced/parametric/stratified bootstrap —
measure the tail divergence vs R's regression `empinf`; implement `empinf` if material, else B.

**JUSTIFY→B (confirmed; write into a new `CONVENTIONS.md` "Capability scope" section, NO code):**
gam exotic bases `re`/`ds`/`gp`/`fs` (specialist; `cr`/`tp`/`cc`/`ps` cover mainstream) ·
regression non-treatment contrasts helmert/sum/poly (math-identical fitted values; coding-only) ·
regression fully-general `quasi(link,variance)` constructor (quasipoisson/quasibinomial ARE built
per VA-5; arbitrary-variance is specialist) · survival `exact` ties (Efron+Breslow cover practice;
`exact` rare+costly, fails loud) · montecarlo `h`/`hinv`/antithetic/blocked-permutation (advanced
variance-reduction; caller-applyable) · multivariate `prcomp tol` (`n_components=` is equivalent).

## Recommended phasing (chips)
1. **4.7.0 additive bundle (do FIRST — fastest ledger shrink, lowest risk):** the JUSTIFY-B
   "Capability scope" `CONVENTIONS.md` section (closes the whole justify side) + the small/med
   implements that don't need a big build: `A5`, `A6`, `VA-7`, `VA-10a`, `VA-11`, `VA-6a`,
   `VA-6b`, `VA-5`, `A4`, `A3`, `VA-2`, `VA-3`, `A2` + `A7` measure + `A8`/`A9`. (Split into 2–3
   chips if it gets unwieldy; several are medium.) Each item: implement + tests (Rule 7) + log to
   `.release/UNRELEASED.md` (Rule 10), then one bundled patch/minor + re-validate the touched
   surface (R6/R18) and refresh that module's report.
2. **Tier-A large builds (one release each):** `A1+VA-8` survival cluster (stratified Cox is the
   single most-demanded self-flagged gap in the corpus) · `VA-1` gam tensor smooths · `VA-4`
   timeseries `xreg`+drift. Each is a real statistical build → validate vs R (`survival::coxph
   strata()`, `mgcv te()`, `forecast::Arima xreg`) with R10 hard cases + (if it has a GPU path)
   the R12 CUDA-first fp32 gate.
3. **Pre-launch consistency sweep → 5.0:** hunt for remaining v1-regret API smells (park them in
   `docs/ROADMAP.md` → "Deprecations & scheduled removals" as you go); then cut **5.0** removing
   everything in that table (currently `mvnmle backend='cpu-reference'` → `solver='reference'`)
   plus the sweep's findings. The `.release/CHECKLIST.md` "Major releases only" gate enforces this.
4. **Single whole-library bless:** all corpus reports current at the 5.0 line, every finding
   ledger closed (R18), the B-appendix assembled. Then — and only then — PyStatsBio.

## Repos / env / flow
- **Library:** `Dev/pystatistics` (GitHub `sgcx-org/pystatistics`, branch main; releases on main).
  Constitution `pystatistics/CONVENTIONS.md`; roadmap+removals `docs/ROADMAP.md`; release flow
  `.release/CHECKLIST.md` (hand-written CHANGELOG+README, `release.py --bump`, tag, push, `gh
  release create` → `publish.yml` → PyPI). **Publishing a release needs the user's explicit go**
  (outward/irreversible, per `OPERATIONS.md`). Any breaking change → 5.0 punch-list, NOT a patch.
- **Validation:** `Dev/pystatistics-validation` (branch main; commit+push). Drivers `drivers/<mod>/`,
  prose `subsystems/<mod>/meta.json`, artifacts `artifacts/<mod>/v<ver>/`, render via
  `render_report.py`. Shared harness `pystatsval` lives in `pystatistics/validation/`.
- **Data:** `DATASETS_ROOT=Dev/datasets` (HDF5, R17). **R** available (`Rscript`).
- **CUDA:** any GPU claim proven on **Forge (CUDA) first** — standing allowance in `OPERATIONS.md`
  (`gpumice` env or a throwaway cloned env + scratch dir; yield the GPU via `nvidia-smi`; clean up
  after). MPS is second-class. See memory `cuda-first-gpu-ordering`, `forge-cuda-testing`.
- **A0 precedent:** memory `a0-mvnmle-mice-blessed-4.6.13` shows the full pattern (throwaway PyPI
  venv, CUDA+MPS gate proof, R-reproduce, bundle-fix→release→re-validate→render→bless→commit).
