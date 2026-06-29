# Validation program roadmap

The module-by-module plan for the pystatistics validation corpus. Tackled **one at
a time**, each to the rigor of `RIGOR.md` (treat as if publishing). The
**coordinator chip** owns this file: it advances the program one module at a time
and keeps the status table current.

**Current library version:** pystatistics **4.3.3** (on PyPI). Lineage: 4.0
standardized the API to `CONVENTIONS.md`; **4.1.0** consistency sweep (R8); **4.2.0**
survival optimizations (coxph O(n²)→O(n), R1); **4.2.3** plain-fp32 GPU GLM convergence
fix (R9); **4.2.4** MPS fp32 GLM gated matrix-free CG (squaring-free, host-fp64 gate);
**4.3.0** added `weights=`/`offset=` support + Gamma AIC (and briefly regressed GPU-OLS
SEs); **4.3.1** exposed negbin `theta` + AIC/BIC counting it; **4.3.2** fixed three
correctness defects the 4.3.1 re-validation surfaced (Gamma/Gaussian BIC, binomial
deviance/AIC/BIC machine-eps clamp, and the 4.3.0 GPU-OLS-SE understatement); **4.3.3**
exposed `DiscreteTimeSolution.converged/.n_iter` (C5, additive — IRLS untouched, all
discrete_time numbers bit-identical).

**Regression is DONE and red-teamed at 4.3.2.** **Survival is DONE and red-teamed at
4.3.3** — R15 found the `intervals=None` default is actually CORRECT (matches R to
~1e-15, even at separation extreme — not the presumed footgun); R13 re-proved the fp32
no-silent-wrong gate on discrete-time's own person-period regime (MPS + CUDA, no
silent-wrong band, inference-SE extension); R8 confirmed discrete_time bit-identical
4.2.4→4.3.3; the red-team surfaced C5 (missing convergence accessors), fixed additively
in 4.3.3. **Next NEW module: multivariate (full run + red-team in one chip).**

## Order + status

Ordered by: foundation-first (the shared `core/compute` kernel) → highest-leverage
rideable methods → heaviest optimization headroom → correctness-dominant tail.

| # | Module | Status | Notes |
|---|---|---|---|
| 1 | regression (OLS + GLM families) | ✅ done + red-teamed (v4.3.2) | v3.18.0 → v3.20.0 → v4.2.x → hardened/red-teamed **v4.3.2**: R10 hard cases (weights/offset now supported), R11 precision/hardware isolation + BLAS, R12 extended to inference SEs, priority-3 CPU-vs-R across sizes (meets/beats R at every n incl. n=50), priority-4 GPU value (~18×/10×). Found+fixed 3 correctness defects (→4.3.2). |
| 2 | survival (KM / log-rank / coxph) | ✅ done + red-teamed (v4.3.3) | v4.0.0 → v4.2.x → red-teamed **v4.3.2/v4.3.3**: R15 default validated CORRECT (matches R ~1e-15, not a footgun), R13 fp32 no-silent-wrong gate re-proven on discrete-time's person-period regime (MPS+CUDA, +inference SEs), R8 bit-identical, C5 convergence accessors added (4.3.3). Stratified Cox = known gap (fail-loud, NotImplementedFeatureError). OWED: prior-art trawl before publication (novelty path). |
| 3 | multivariate (PCA + factor analysis) | ⬜ next NEW module — FULL run + red-team in ONE chip | SVD/eigendecomposition; the most GPU-amenable op; TCGA large-matrix scaling; self-contained. No baseline exists → first-time validation AND red-team together (not done until red-teamed). Starts after the survival red-team lands AND user says go. |
| 4 | mixed (LMM) | ⬜ pending | biggest optimization headroom (least GPU-touched); REML over variance components |
| 5 | gam | ⬜ pending | penalized IRLS + REML smoothing selection; rides the kernel |
| 6 | timeseries (ARIMA/ETS/STL) | ⬜ pending | largest module; Kalman/state-space + optimizer loops |
| 7 | montecarlo (bootstrap/permutation) | ⬜ pending | embarrassingly parallel → clean GPU story |
| 8 | ordinal (polr) + multinomial | ⬜ pending | IRLS-family; de-risked by the optimized kernel |
| 9 | anova, descriptive, hypothesis | ⬜ pending | correctness-dominant, optimization-light; corpus completeness; batch last |

Done already (pre-order): mvnmle (v3.18.0), mice (v3.16.3 / v3.18.0).

## Open work

- **Multivariate (#3) — CHIP SPAWNED (full run + red-team in ONE chip).** No baseline →
  first-time validation AND red-team together, priority order. PCA + factor analysis
  (SVD/eigendecomposition). R refs: `stats::prcomp`/`princomp` (PCA), `stats::factanal`
  (FA). Chip picks canonical datasets (added to the central HDF5 store per R17). Target
  PyPI **4.3.3**. Forge/CUDA allowance granted. **Must clear CARRY_FORWARD CF-1** —
  verify PCA's GPU path forms the covariance/Gram in fp64, not fp32 (the regression
  4.3.2 defect class); ill-conditioned R10 case on MPS + CUDA; library fix via R8/R6, or
  R16 if a showstopper silent-wrong.
- **Owed (cross-program, before any paper):** prior-art trawl (arXiv/PyPI/GitHub) for the
  survival novelty claim "first discrete-time survival on GPU at scale" (R13). Not blocking
  the validation corpus; required before publication.
- **Minor (survival, non-blocking):** the 4.3.3 changelog's illustrative convergence
  examples ("all-censored → n_iter=0" / "separated → converged=False") describe narrower
  code paths than the validated behaviour (which tracks R bit-for-bit); a doc-wording
  tidy for a future release, not a defect.

## Standing coordination constraints

- **Release-hold: CLEARED.** The `pystatsbio`/`sgcbio` consistency releases have
  landed and pystatistics has since shipped through **4.3.3**. No active hold. Re-check
  before any new library release; reinstate this line if a downstream consistency
  release is mid-flight again.
- **One at a time.** Do not spawn the next module chip until the current one is done
  AND the user says go. The coordinator confirms before spawning.
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
   allowance (only if a GPU path is warranted per the constitution), and the **R16
   stop-fix-release-restart** rule: a severe (showstopper / quiet-wrong) bug halts the
   run — surface to the user, fix, cut+publish a patch release, restart from the new
   version; do not validate on past a known-broken version.
5. Deliverables: `drivers/<m>/`, `artifacts/<m>/v<current>/`, `subsystems/<m>/meta.json`,
   `reports/<m>-v<current>.md` (current PyPI version, which may advance mid-run under
   R16); commit to validation `main` and push. **Any new dataset → centralized HDF5 in
   `Dev/datasets` (writer + SCHEMA + MANIFEST), loaded via `MVNMLE_DATA_DIR` — no CSVs in
   `drivers/*/data/` (R17).**
6. Opens with discuss-before-acting: understanding + plan first.

(The `regression` v4.3.2 chip is the worked example of the full rigor bar; the
`survival` v4.3.2 chip is the worked example of the red-team bolt-on.)
