# Validation program roadmap

The module-by-module plan for the pystatistics validation corpus. Tackled **one at
a time**, each to the rigor of `RIGOR.md` (treat as if publishing). The
**coordinator chip** owns this file: it advances the program one module at a time
and keeps the status table current.

**Version lineage.** 4.0
standardized the API to `CONVENTIONS.md`; **4.1.0** consistency sweep (R8); **4.2.0**
survival optimizations (coxph O(n²)→O(n), R1); **4.2.3** plain-fp32 GPU GLM convergence
fix (R9); **4.2.4** MPS fp32 GLM gated matrix-free CG (squaring-free, host-fp64 gate);
**4.3.0** added `weights=`/`offset=` support + Gamma AIC (and briefly regressed GPU-OLS
SEs); **4.3.1** exposed negbin `theta` + AIC/BIC counting it; **4.3.2** fixed three
correctness defects the 4.3.1 re-validation surfaced (Gamma/Gaussian BIC, binomial
deviance/AIC/BIC machine-eps clamp, and the 4.3.0 GPU-OLS-SE understatement); **4.3.3**
exposed `DiscreteTimeSolution.converged/.n_iter` (C5, additive — IRLS untouched, all
discrete_time numbers bit-identical); **4.4.0** added the multivariate randomized GPU
(MPS/CUDA) PCA path; **4.4.1** R18 bundle fixing the two factor-analysis findings F1
(varimax relative-convergence test) + F2 (Heywood `lower=` floor); **4.5.0** mixed/LMM
perf (F1/F2a/F2b) + the new `grm_lmm()` GPU model + `CONVENTIONS.md` A7 (dependency
tiering) + the Prime Directive reframe; **4.5.1** F3 fix (an extreme-variance-ratio
silent-wrong the mixed red-team caught — R16).

**Current library version:** pystatistics **4.6.8** (on PyPI). **montecarlo** first-time
validation drove an **R16 showstopper fix at 4.6.7** (GPU backends inferred the statistic
form from a single resample → could silently compute the wrong statistic → fixed via an
explicit fail-loud `gpu_statistic` opt-in) and an **R18 gather bundle at 4.6.8** (boot_ci
`norm.inter` quantile + regression-influence BCa acceleration → all 5 CI types now match
`boot.ci`; two-sided permutation p-value → 2·min-tail, correct for non-centered stats).
Prior — **timeseries** (the
largest module) first-time validation across the whole surface drove a fix cascade
**4.6.2→4.6.5** (STL loess rewrite — trend-leak; `ndiffs` default KPSS-not-ADF; ADF
p-value in the fail-to-reject region; **seasonal AIC counting free params not expanded
polynomial coeffs — was silently mis-driving `auto_arima` seasonal selection**) → blessed
whole-surface at 4.6.5; **4.6.6** ETS numba R6 perf cycle. `arima_batch` GPU path is a
batched Whittle spectral likelihood → **CF-1 N/A** (no normal-equations Gram), host-fp64
stationarity gate (R14). Prior — **gam** first-time
validation drove a full numerical rewrite: **4.6.0** fixed the H0 showstopper
(unconstrained smooth bases → exactly-singular design → silently-wrong smoothing
selection on the DEFAULT path, shipped unvalidated through 4.5.7) with mgcv-exact
constrained cr/tp bases + augmented-QR PIRLS + Laplace REML + real posterior SEs, and
**removed the gam fp32 GPU path** (H1 = CF-1 materialised: silent-wrong EDF on CUDA+MPS,
no gate); **4.6.1** added the Gaussian analytic sp-gradient (H4). Prior mixed arc:
**4.5.7** A.3 analytic θ-gradient (F4; torch rejected on benchmark); **4.5.2→4.5.6** the
glmm cascade (G1 nAGQ Laplace, G2 SE-transpose silent-wrong, G3 fail-loud, G4 PIRLS
overflow, G5 solver); **4.5.0/4.5.1** mixed LMM perf + `grm_lmm()` + `CONVENTIONS.md` A7 +
Prime Directive reframe + F3 fix.

**Regression (4.3.2), survival (4.3.3), and multivariate (4.4.1) are DONE and
red-teamed.** Multivariate: PCA machine-precision vs `prcomp` (R10 cond~1e8 recovers a
7e-9 singular value via SVD-of-X), a new MPS randomized PCA path that earns its keep,
**CF-1 CLEARED** (silent_wrong_count=0 on the gated `solver='gram'` path, R12/R13
re-proven MPS+CUDA), FA validated vs `factanal` after the R18 bundle (F1+F2); F3 (small-n
FA speed dip) is a documented R2 cost. **`mixed` (v4.5.7)** and **`gam` (v4.6.1)** are
also DONE + red-teamed — gam's first-time validation caught a DEFAULT-path silent-wrong
(H0) + the CF-1 GPU silent-wrong (H1), driving a full rewrite. **`timeseries` (v4.6.6)**
is also DONE + red-teamed — whole surface vs `stats`/`forecast`/`tseries`, several real
correctness defects caught (seasonal-AIC/auto_arima, STL trend-leak, ndiffs default).
**8 of 9 done. HOME STRETCH — `montecarlo` (#7) DONE + red-teamed (v4.6.8). 2 modules
left: ordinal+multinomial (#8, next NEW module), anova/descriptive/hypothesis (#9).**

## Order + status

Ordered by: foundation-first (the shared `core/compute` kernel) → highest-leverage
rideable methods → heaviest optimization headroom → correctness-dominant tail.

| # | Module | Status | Notes |
|---|---|---|---|
| 1 | regression (OLS + GLM families) | ✅ done + red-teamed (v4.3.2) | v3.18.0 → v3.20.0 → v4.2.x → hardened/red-teamed **v4.3.2**: R10 hard cases (weights/offset now supported), R11 precision/hardware isolation + BLAS, R12 extended to inference SEs, priority-3 CPU-vs-R across sizes (meets/beats R at every n incl. n=50), priority-4 GPU value (~18×/10×). Found+fixed 3 correctness defects (→4.3.2). |
| 2 | survival (KM / log-rank / coxph) | ✅ done + red-teamed (v4.3.3) | v4.0.0 → v4.2.x → red-teamed **v4.3.2/v4.3.3**: R15 default validated CORRECT (matches R ~1e-15, not a footgun), R13 fp32 no-silent-wrong gate re-proven on discrete-time's person-period regime (MPS+CUDA, +inference SEs), R8 bit-identical, C5 convergence accessors added (4.3.3). Stratified Cox = known gap (fail-loud, NotImplementedFeatureError). OWED: prior-art trawl before publication (novelty path). |
| 3 | multivariate (PCA + factor analysis) | ✅ done + red-teamed (v4.4.1) | first-time full validation + red-team. PCA machine-precision vs `prcomp`, R10 hard cases (cond~1e8 via SVD-of-X), new MPS/CUDA randomized PCA path earns its keep, **CF-1 cleared** (gated `solver='gram'`, silent_wrong_count=0, R12/R13). FA vs `factanal` validated at 4.4.1 after R18 bundle (F1 varimax relative-convergence + F2 Heywood `lower=`). F3 small-n FA dip = documented R2. |
| 4 | mixed (`lmm` + `glmm` + `grm_lmm`) | ✅ **DONE + red-teamed (v4.5.7)** — LMM + GLMM + GRM all validated; A.3 (analytic θ-gradient) closed F4 | `lmm` vs `lme4`/`lmerTest` + `grm_lmm()` vs `rrBLUP` (blessed 4.5.1, byte-identical at 4.5.6). **`glmm()` validated vs `lme4::glmer` (Laplace nAGQ=1)** on the two-tier contract across binomial/logit, poisson/log, correlated random slope, probit; R10 red-team + R15 default match glmer's behaviour. First-time glmm pass drove a fix cascade **4.5.2→4.5.6**: G1 nAGQ=0→Laplace + G2 SE transpose (silent-wrong for correlated predictors) @4.5.2; G3 fail-loud free-dispersion families @4.5.3; G4 PIRLS overflow + variance-collapse @4.5.4/completed @4.5.6; G5 structure-exploiting solver (removes O(#groups³) gap) @4.5.5. G6 (no glmm is_singular flag) + G7 (no aggregated-binomial/weights/offset) documented. F4 = documented R2 (autodiff/A.3). |
| 5 | gam | ✅ **DONE + red-teamed (v4.6.1)** — first-time validation drove a full numerical rewrite | vs `mgcv::gam`. No baseline → first-time run found the SHOWSTOPPER (H0): pre-4.6.0 smooths had no identifiability constraint → exactly-singular design → garbage EDF (`total_edf=−3.12`) → silently-wrong GCV/REML smoothing selection on the DEFAULT path (unvalidated module shipped through 4.5.7). Full rewrite at **4.6.0**: mgcv-exact constrained cr/tp bases + augmented-QR PIRLS + Laplace REML + real posterior SEs; two-tier contract met (tier-1 fixed-sp ≤1e-9, tier-2 free sp ~1e-5). **CF-1 materialised for gam** (fp32 GPU Gram band, silent-wrong on CUDA+MPS, negative EDF) → **GPU path REMOVED** (H1; no shippable GPU win per the CUDA-first investigation). H2 (scale-invariant GCV) + H3 (names mislabel) fixed in adversarial review pre-release. **H4 resolved at 4.6.1** — Gaussian analytic sp-gradient (mixed-A.3 pattern); GLM families keep finite-diff (documented). CPU-only. |
| 6 | timeseries (ARIMA/ETS/STL) | ✅ **DONE + red-teamed (v4.6.6)** — first-time whole-surface validation | vs `stats`/`forecast`/`tseries` (+ statsmodels as an independent 3rd reference). Whole surface: acf/pacf/diff/ndiffs, adf/kpss, arima/auto_arima/arima_batch, ets, stl/decompose. Deterministic pieces machine-precision (acf ~1e-16, stl ~1e-13, exact test stats); MLE fits (arima/ets/auto_arima) optimizer-tier, method-matched. Fix cascade **4.6.2→4.6.5**: STL loess rewrite (trend-leak), ndiffs default KPSS, ADF p-value, **seasonal AIC free-param count (was silently mis-driving auto_arima seasonal selection)**. 4.6.6 ETS numba R6. `arima_batch` GPU = batched Whittle spectral → **CF-1 N/A** (no Gram); host-fp64 stationarity gate (R14). |
| 7 | montecarlo (bootstrap/permutation) | ✅ **DONE + red-teamed (v4.6.8)** — first-time whole-surface validation + red-team; R18 bundle re-blessed at 4.6.8 | `boot`/`boot_ci`/`permutation_test` vs `boot::boot`/`boot.ci` + base-R exact permutation enumeration. Three-part STOCHASTIC contract: R6 seed determinism; TIGHT tier feeding R's genuine resample indices to pystatistics (statistic 1e-15, all 5 CI types machine-precision/documented-convention, BCa z0 exact ~1e-17); statistical-equivalence tier (independent-RNG large-B within MC error) + exact-enumeration permutation + known-DGP coverage study (BCa best-calibrated). **Drove one R16 showstopper**: GPU backends inferred the statistic form from a SINGLE resample → could silently compute the mean for a different statistic on `backend='gpu'`. Fixed **4.6.7** (published): explicit `gpu_statistic` opt-in, fail-loud, no silent substitution. GPU real win — MPS fp32 25–43× (drift ~1e-7 within tier), CUDA fp64 exact 59–158×. **CF-1 N/A** (no Gram). **R18 gather bundle → 4.6.8 (fixed, not documented-away):** boot_ci now uses R's `norm.inter` quantile (basic/perc/stud machine-precision vs boot.ci) + regression-influence BCa acceleration (BCa ~1e-5); `permutation_test` two-sided now 2·min-tail (correct for non-centered stats). All 5 CI types now match boot.ci on shared replicates. Remaining R2: tiny-n CPU lag reverses by n~1000 (3.4× at n=10000). |
| 8 | ordinal (polr) + multinomial | ⬜ next NEW module — FULL run + red-team in ONE chip | IRLS-family; de-risked by the optimized kernel. R refs `MASS::polr` (ordinal) / `nnet::multinom` (multinomial). **⚠️ CARRIES THE CF-1 FLAG** — the IRLS Gram `X'WX`; if it exposes an fp32 GPU path, that's a live CF-1 exposure (gam materialised it). MUST check, not assume. No baseline → first-time validation AND red-team. Starts on user go. |
| 9 | anova, descriptive, hypothesis | ⬜ pending | correctness-dominant, optimization-light; corpus completeness; batch last |

Done already (pre-order): mvnmle (v3.18.0), mice (v3.16.3 / v3.18.0).

## Open work

**`mixed` (v4.5.7), `gam` (v4.6.1), `timeseries` (v4.6.6), and `montecarlo` (v4.6.8) are
COMPLETE. HOME STRETCH: 2 modules left** — ordinal+multinomial (#8, next), anova/
descriptive/hypothesis (#9) — then the A7 packaging migration + tail items, then the
downstreams. The two `mixed` BLOCKER bullets below are kept for history.

- **⚠️ ordinal+multinomial (#8) carries the CF-1 flag.** It's the IRLS-family module — if
  it exposes a GPU path forming `X'WX` in fp32, that's a live CF-1 exposure (gam
  materialised it; SVD-immunity doesn't transfer). The ordinal chip MUST check it, not
  assume. (montecarlo/anova don't form a Gram → CF-1 N/A.)

- **CF-1 gam instance — CLOSED (gam GPU path removed at 4.6.0).** The gam fp32 GPU Gram
  band materialised (silent-wrong EDF on CUDA+MPS, no gate) and was closed by removing the
  path (no shippable GPU win). **Durable lesson (in `CARRY_FORWARD.md`): a module forming a
  normal-equations Gram `X'WX(+λS)` on an fp32 GPU path is a LIVE CF-1 exposure whenever an
  optimizer visits the ill-conditioned regime — the SVD-of-X immunity that saved PCA does
  NOT transfer.** Relevant flag for future modules with an IRLS/normal-equations GPU path
  (e.g. ordinal #8): check it, don't assume.
- **Minor (gam, non-blocking) — H4 GLM-family analytic gradient deferred.** The analytic
  sp-gradient closed H4 for the Gaussian-identity path (4.6.1); GLM families
  (Poisson/binomial UBRE, non-Gaussian REML) keep the finite-difference search (their IRLS
  weights depend on the coefficients). The full Wood-2011 implicit-derivative form is a
  documented future optimization, not a defect.

- **BLOCKER 1 — glmm() first-time validation — ✅ DONE (blessed mixed v4.5.6).** Validated
  `glmm()` (generalized LMM via Laplace/PIRLS) against **`lme4::glmer` (nAGQ=1)** on the
  two-tier contract (tight fixed effects + logLik/deviance/AIC/BIC; optimizer/Laplace-tier
  SEs/variance/BLUPs ~1-2%) across binomial/logit, poisson/log, correlated random slope,
  and probit; R10 red-team (quasi-separation, singular RE, unbalanced, fail-loud
  gaussian/gamma) + R15 default match glmer's behaviour. CPU-only (no GPU path, as
  expected). The pass drove the 4.5.2→4.5.6 fix cascade (G1-G5; G6/G7 documented). Report:
  `reports/mixed-v4.5.6.md`.
- **BLOCKER 2 — A.3 θ-gradient (implement + re-validate) — ✅ DONE (blessed mixed v4.5.7).** The F4 fix:
  multi-term random-slope LMM is ~2× slower than lmerTest at scale because
  `_optimizer.optimize_theta` runs L-BFGS-B with **finite-difference** gradients
  (`2·dim(θ)+1` deviance evals per step). Supplying a real θ-gradient cuts that to 1.
  **DECISION (evidence-based): numpy ANALYTIC gradient, NOT torch autodiff.** A benchmark
  of finite-diff vs numpy-analytic vs torch-CPU autograd (all landing at the identical
  optimum; analytic grad matches finite-diff to 1.8e-7) showed the numpy analytic gradient
  is **~2.3× faster than finite-diff at every size, dependency-free**, while **torch-CPU is
  SLOWER than numpy-analytic until n≈12k and only wins ≤1.3× at n≥24k** (~50 ms fits) —
  the "accelerator that rarely wins" anti-pattern. So torch is rejected here; the CPU path
  stays torch-free. Scope: the batched single-factor path (where F4 lives); crossed/sparse
  stays on finite-diff. **Requires re-validation** (R6): (a) analytic grad ≡ finite-diff to
  machine precision + estimates bit-identical to 4.5.6 + match `lme4`; (b) F4 gap closes
  (Performance/G3); (c) R8 bit-identical glmm/grm_lmm untouched. Release (patch, user
  authorizes) → re-validate LMM → then mixed (#4) is ✅ done and gam unblocked. See memory
  `a3-torch-loses-on-cpu` + `torch-dependency-policy`.
- **Minor (mixed, non-blocking) — G6/G7:** `glmm()` has no `is_singular` flag (G6; it
  matches glmer's isSingular *behaviour*, just doesn't surface it — a transparency gap like
  survival's C5) and no aggregated-binomial / weights / offset support (G7). Gather-level;
  a future additive `mixed` release, not a defect.
- **Owed (cross-program, before any paper):** prior-art trawl (arXiv/PyPI/GitHub) for the
  survival novelty claim "first discrete-time survival on GPU at scale" (R13). Not blocking
  the validation corpus; required before publication.
- **Minor (survival, non-blocking):** the 4.3.3 changelog's illustrative convergence
  examples ("all-censored → n_iter=0" / "separated → converged=False") describe narrower
  code paths than the validated behaviour (which tracks R bit-for-bit); a doc-wording
  tidy for a future release, not a defect.

## Standing coordination constraints

- **Release-hold: CLEARED.** The `pystatsbio`/`sgcbio` consistency releases have
  landed and pystatistics has since shipped through **4.6.8**. No active hold. Re-check
  before any new library release; reinstate this line if a downstream consistency
  release is mid-flight again.
- **Standing allowance — the validation testing env always tracks the latest PyPI
  release; do NOT ask each time.** Each chip builds a throwaway PyPI venv (`require_pypi`)
  at the version under validation, which is normally the CURRENT PyPI release. Installing
  or updating that env to the latest released pystatistics is **pre-authorized every
  session** (like the Forge-CUDA allowance) — it is local, throwaway, reversible env
  setup, not an outward/irreversible action, so a chip (or the coordinator) does NOT stop
  to ask permission to bump it. **R5 is unaffected:** each report still pins the exact
  version it validated; the env simply always has frictionless access to the current
  release. (Contrast: *publishing* a release still needs explicit authorization —
  *installing* one never does.)
- **Finding triage: showstopper vs gather (RIGOR R18).** Every defect found mid-run is
  classified when found: a user-misleading correctness defect (silent/uncaught wrong,
  fail-loud bypass) → R16 brakes-slam (stop-fix-release-restart); everything that already
  fails loud / is gated-off-default / bounded / cosmetic → GATHER into one bundled patch
  (e.g. 4.4.0 findings F1+F2 → bundled 4.4.1), and the blessed report is the bundle. Never
  bless a report that silently carries known inaccuracies.
- **One at a time.** Do not spawn the next module chip until the current one is done
  AND the user says go. The coordinator confirms before spawning.
  - **Why (the real rationale, not just caution).** The validation *output* barely
    conflicts — each run writes disjoint `drivers/<m>/`, `artifacts/<m>/`,
    `reports/<m>-*.md`, `subsystems/<m>/`. What forces serialization is **shared mutable
    state**: (1) **library source** — the moment a run finds a defect (CF-1, an R6/R8 fix,
    an R16 showstopper) it mutates `../pystatistics`, and the most-likely-touched code is
    the shared `core/compute` kernel where cross-module bugs live; (2) **the version line
    and releases** — pystatistics has ONE version lineage (… → 4.3.3 → 4.4.0 …) and two
    chips cannot both cut a PyPI release or race the version number (R5 version-pinning and
    R16 stop-fix-release-restart both assume a single coherent version at any instant);
    (3) **coordinator-owned shared files** — `ROADMAP.md`, `RIGOR.md`, `CARRY_FORWARD.md`,
    and the central `Dev/datasets` store. So sequential buys **a single, coherent,
    releasable library state** — not merely tidy merges. Every run is simultaneously a
    *consumer* and a potential *mutator* of the trunk; a large org parallelizes features
    against a stable trunk, which this is not.
  - **Only sanctioned parallel exception:** a batch of **pure-correctness validations on
    modules judged unlikely to need a library fix** (e.g. the correctness-dominant tail —
    anova / descriptive / hypothesis), under a HARD rule: **no library mutation; anything
    that finds a real defect STOPS and re-serializes.** Since you cannot predict which run
    trips a fix, even this is a gamble — default remains sequential.
- **Cross-module issues: fix-now-or-log; the whole library is in scope (RIGOR R8 +
  `CARRY_FORWARD.md`).** A shared-code fix that affects >1 module is R8 — fix it
  library-wide, or if it can't be done now, log it in `CARRY_FORWARD.md` so the affected
  module's chip clears it. Modules are not silos; the Rule-8 boundary is *sibling repos*,
  not intra-library modules. Open: **CF-1** (fp64-Gram GPU fix → PCA).
- **Datasets: centralized HDF5 only (RIGOR R17).** One store (`Dev/datasets` /
  `/mnt/data/pystatistics-datasets` via `MVNMLE_DATA_DIR`), HDF5 via `dataset_writer.py` +
  `SCHEMA.md` + `MANIFEST.sha256`. No new CSVs in drivers. Cleanup owed: migrate the
  existing `drivers/{regression,survival}/data/*.csv` stragglers into the store.
- **GPU must EARN its existence (RIGOR priority 4 / R1).** The GPU trades accuracy for
  speed, so it must be FASTER than the CPU in its intended large-`n` regime — a GPU that
  ties/loses to the CPU there is a finding, not an acceptable result. Not miracles; just
  no embarrassment. Small-`n`/narrow-`p` parity is expected and fine.
- **Regime-conditional guarantees; validate the default (RIGOR R13/R14/R15).** Inherited
  guarantees (a forwarding module riding another's gate) must be RE-PROVEN on the new
  regime; the guarantee lives on the version-independent layer (host fp64 gate, not the
  torch-sensitive solver — name the torch version); validate the DEFAULT invocation a
  naive user triggers, not just the expert case. Novelty claims get the most scrutiny.
- **Hard cases + match R's failures/warnings (RIGOR R10).** Correctness grids must
  reach the adversarial regime (collinearity to the refusal boundary, separation,
  factor coding, weights/offsets, rank-deficiency) and match R's failures and warnings
  — not just easy-data numbers.
- **Isolate precision from hardware in GPU benchmarks (RIGOR R11).** Report `gpu_fp64`
  vs `cpu_fp64` (hardware alone) alongside any bundled fp32-GPU-vs-fp64-CPU-R number,
  and name the BLAS R linked against.
- **Relaxing fail-loud → convergence needs a no-silent-wrong proof (RIGOR R12, the
  complement to R9).** Any fail-loud→"converges" relaxation must carry an adversarial
  stress test that the accept/refuse boundary is principled (no accepted-but-wrong
  band). R9 + R12: the gate must be a true classifier both directions.
- **fp32 GPU non-convergence is a false-negative until proven otherwise (RIGOR R9).**
  CPU is fp64; MPS/CUDA fp32 cannot meet R's absolute `|Δβ| < 1e-8`. Before recording
  any MPS/CUDA path as failing/unstable, compare its coefficients to the fp64 fit —
  agreement to the fp32 tier (`max_rel ~1e-6`) means the fit is correct and the
  convergence *test* is wrong. Do not blame the GPU backend without cause.
- **Cross-module consistency discoveries → pause + library-wide minor release** (see
  RIGOR.md R8). If, while validating a module, you find a consistency issue that
  affects **more than one module**, pause that module's work and ship a consistency
  minor release for the library as a whole (this is exactly what **4.1.0** was — a
  multi-module fix surfaced by survival, spun out so survival could continue). **If
  the fix is breaking, STOP and discuss with the user** (a breaking change is not a
  quiet minor release).
- **Severe (showstopper) bug mid-run → STOP, fix, release, restart (RIGOR R16).** A
  quiet-wrong result / wrong-but-precise-looking output / bypassed fail-loud guarantee
  means the version under test is dead — do NOT continue validating it under the
  "frozen/immutable" doctrine. Slam the brakes, surface to the user, fix as top
  priority, cut a patch release (e.g. 4.3.2 → 4.3.3) + publish to PyPI (user authorizes,
  never silent), discard the doomed version's partial artifacts, and RESTART validation
  from the new version. A bug that already fails loud is NOT a showstopper → log + R6
  next cycle. When in doubt, treat as severe.

## Per-module chip template (the coordinator fills `<<…>>` and embeds the rules)

**Two chip shapes — know which you're spawning:**
- **Red-team bolt-on** (what `regression` and `survival` got): the module was already
  validated/frozen, so the chip ADDS red-team evidence (R10/R12/R13/R15…) + any priority
  upgrades on top of existing frozen numbers. Historical — both are now done.
- **Full run + red-team in ONE chip** (every NEW module from `multivariate` onward): there
  is **no baseline** — the chip does the complete first-time validation **and** the
  red-team in a single pass. Do not split into baseline-now / red-team-later. A new module
  is not "done" until it has been red-teamed.

Each module chip is self-contained (the spawned session has no prior context) and:
1. Points at `ARCHITECTURE.md`, `RIGOR.md`, `CONVENTIONS.md`, **`CARRY_FORWARD.md`**, and
   the completed examples (`reports/regression-v4.3.2.md` — the current rigor bar, plus
   `survival-v4.3.3.md`, mvnmle, mice) + the harness `pystatsval` API + the salvageable
   R refs in `_archive/`. **Reads `CARRY_FORWARD.md` and clears any item targeting this
   module** (e.g. CF-1: the fp64-Gram GPU fix for PCA).
2. States the module, its R reference (the R package/function), the canonical
   dataset(s), and the target version (current PyPI release — **4.3.2**).
3. Mandates the full `RIGOR.md` deliverables, worked in **priority order** (the lead
   section): **(1) correctness** vs the promised tolerance, **incl. the R10 hard-case
   grid matching R's failures/warnings and the R15 default-invocation check**; **(2)**
   the hard-problem red-team; **(3) the mandatory CPU-vs-R speed study ACROSS SIZES**
   (small-n overhead → large-n; CPU must never lag R); **(4) GPU at its bar** — must be
   faster than CPU in its regime (R11 precision-vs-hardware isolation), never silently
   wrong (R12 no-silent-wrong proof; R13 don't inherit a guarantee — re-prove it on this
   module's regime; R14 guarantee on the version-independent layer). Plus the **R4
   constitutional audit** and an honest perf story (R2: document any justified slowdown;
   never sweep).
4. Repeats the current **release-hold** status, the Forge standing-CUDA-testing
   allowance (only if a GPU path is warranted per the constitution), the **standing
   allowance that the PyPI testing env always installs the current release without asking**
   (installing ≠ publishing), and the **R16 stop-fix-release-restart** rule: a severe
   (showstopper / quiet-wrong) bug halts the run — surface to the user, fix, cut+publish a
   patch release, restart from the new version; do not validate on past a known-broken
   version.
5. Deliverables: `drivers/<m>/`, `artifacts/<m>/v<current>/`, `subsystems/<m>/meta.json`,
   `reports/<m>-v<current>.md` (current PyPI version, which may advance mid-run under
   R16); commit to validation `main` and push. **Any new dataset → centralized HDF5 in
   `Dev/datasets` (writer + SCHEMA + MANIFEST), loaded via `MVNMLE_DATA_DIR` — no CSVs in
   `drivers/*/data/` (R17).**
6. Opens with discuss-before-acting: understanding + plan first.

(The `regression` v4.3.2 chip is the worked example of the full rigor bar; the
`survival` v4.3.2 chip is the worked example of the red-team bolt-on.)
