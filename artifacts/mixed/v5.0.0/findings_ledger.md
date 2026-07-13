# mixed (LMM + GLMM + GRM) v5.0.0 — findings ledger (BLESS-CLEAN · module complete)

Validation of the **PyPI 5.0.0** build. Report: `reports/mixed-v5.0.0.md`. This is the
**R5 version-bump re-validation** at the 5.0.0 major release, carrying the blessed
4.5.7 state (`reports/mixed-v4.5.7.md`) forward. No behavioural change; no driver
edits required.

## Version arc

`4.5.1` (LMM + GRM blessed) -> glmm cascade `4.5.2-4.5.6` -> **`4.5.7`** A.3 analytic
theta-gradient (module complete) -> **`5.0.0`** major release (this re-validation).

## API / surface audit (R5 rename check)

- `lmm`, `glmm`, `grm_lmm` signatures are **identical** to 4.5.7. `family` still
  accepts a family string; `grm_lmm`'s third positional argument is `random_factor`,
  which the drivers pass positionally (`grm_lmm(y, X, W, backend=...)`) - no `W=`
  kwarg in play.
- The `.se -> .standard_errors` / `.ci -> .conf_int` field renames (migrated at 4.0)
  are the live spellings; the drivers already use them. **Zero Python-side edits.**
- `detect_install_source()` is no longer exported; PyPI provenance is asserted via
  `pystatsval.device.require_pypi(env_manifest(...))`, which the drivers already use.
  `install_source == pypi`, `__version__ == 5.0.0` confirmed.

## Re-validation evidence (all CPU studies re-run live on 5.0.0)

- **LMM correctness** vs lme4/lmerTest - TIGHT tier: beta to ~1e-16..3e-13, logLik ~1e-11.
  OPTIMIZER tier: SE/df/varcomp/BLUP <= 8e-5. Dyestuff2 singular on both engines.
  Reproduces 4.5.7.
- **LMM hard-cases (R10 + R15)** - singular diagnostics, extreme variance ratio,
  nested, and the matched reused-inner-label footgun all match lme4's behaviour;
  bare-default == explicit `(1|group)`. Reproduces 4.5.7.
- **GRM correctness** vs rrBLUP - beta <= 3.4e-7, variances <= 2e-5, h2 <= 9.3e-6,
  genetic-BLUP correlation = 1.000000. Reproduces 4.5.7.
- **GLMM correctness** vs glmer(nAGQ=1) - fixed effects + logLik ~1e-3; SE/z/varcomp/
  BLUP within the ~1-5% Laplace tier. Reproduces 4.5.7.
- **GLMM hard-cases (R10 + R15 + A6)** - quasi-separation ~4e-5; singular-RE variance
  collapses to 0 matching glmer isSingular; unbalanced ~2e-4; bare-default == explicit;
  **gaussian and gamma free-dispersion still FAIL LOUD** (ValidationError). Fail-loud
  and the singular classifier both intact.
- **LMM / GLMM CPU speed** - profiles reproduce 4.5.7 (LMM intercept ~3-5x over lme4;
  intercept+slope 2.1x at G=10 -> 0.55x at G=2000; GLMM 0.29x at G=25 -> 0.62x at
  G=400). No complexity-class regression. (Timing noise under concurrent load is a
  documented R2 caveat, not a showstopper.)

## GPU legs

- **MPS (live on 5.0.0)** - CF-1 no-silent-wrong gate holds: `silent_wrong_count = 0`;
  cond(W) <= 3e3 accepted (|dh2| <= 1.4e-3), cond(W) >= 1e4 refused loud. Speed 0.36-1.0x
  vs fp64 CPU - correctness/portability path, not a win (reported honestly).
- **CUDA / Forge** - carried forward from 4.5.7 (no CUDA host on this run). The live
  MPS fp32-Gram gate re-proves the same fp32 defect-class classifier behaviour on
  5.0.0.

## Findings - final disposition

- **F1-F4** (LMM/GRM) and **G1-G5** (GLMM) - RESOLVED (<= 4.5.7). **G6/G7** (GLMM,
  no `is_singular` flag / boundary warning) - DOCUMENTED, unchanged.
- **No new findings at 5.0.0.**

**`mixed` is complete and BLESS-CLEAN at 5.0.0.** No open findings; no behavioural
change vs 4.5.7.
