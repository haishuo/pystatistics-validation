# VA-2 / VA-3 gam validation — session brief

**Mission:** dedicated two-tier validation of the two gam surfaces that shipped in
4.7.0 with **no rendered validation report**: **VA-2 `s(x, by=)`** (varying-coefficient
/ by-variable smooths) and **VA-3 `nb()`** (negative-binomial theta estimation).
This is the **last open item gating the pystatistics whole-library bless** — the
v5.0.0 session is holding its release on this session's verdict.

**Requested by:** the v5.0.0 session (2026-07-12), executing the owed coordinator
item from `handoffs/v4.8.0-validation.md` §"Owed coordinator item".

---

## What is known (quick probes at 4.8.0, from the 4.8.0 handoff)

- **VA-2 `by=` (continuous):** approximately matches mgcv — edf 8.7245 vs 8.7416
  (~0.2%), fitted ~0.5% — but **NOT confirmed machine-precise**. The by-variable
  penalty-scaling / sp parametrization differs (py lambda 4.75 vs mgcv sp 0.0123).
  Possibly a GAM-2-style reparameterization convention (estimator exact, sp scale
  different) — but that is UNPROVEN for `by=`. Not confirmed silent-wrong; not
  confirmed exact.
- **VA-2 factor-by:** semantics unchecked. py's `by=` treats a numeric column as a
  continuous varying coefficient; mgcv's factor-`by` builds **per-level smooths**.
  Determine what py does with a factor/categorical by (per-level smooths? loud
  refusal? silent misinterpretation? — the last would be a Guarantee-2/A6
  showstopper).
- **VA-3 `nb()`:** approximately matches — py edf 6.9782 vs mgcv 7.0487 (~1%);
  the theta accessor is unverified.

## Required design (model on `drivers/gam/run_tensor.py` — the VA-1 pattern)

Two-tier vs **mgcv 1.9.3 / R 4.5.2**:

1. **TIER 1 — fixed-sp cross-feed:** feed mgcv's optimized sp into py (mind the
   parametrization question: establish the sp-scale mapping for `by=` smooths
   first, or demonstrate why the cross-feed is inapplicable, as VA-1's GAM-2 did
   for ti margins — with per-term edf/deviance/fitted evidence, not hand-waving).
   Goal: prove the **basis + penalty construction is mgcv-exact**.
2. **TIER 2 — free REML/GCV:** fitted values, per-term + total EDF, deviance,
   REML/GCV score, and for nb: the **estimated theta** vs `m$family$getTheta(TRUE)`.

R10 hard cases to cover (at minimum): by-variable with negative values and with
zero-variance stretch; by= combined with a second smooth; factor-by (if supported,
per-level exactness; if not, fail-loud proof); nb with small counts / large theta
/ theta near the Poisson limit. Fail-loud checks per the fidelity harness.

## Ground rules

- **Artifact:** validate the **PyPI** artifact `pystatistics==4.8.0` in a
  throwaway venv (R5). `by=`/`nb` shipped in 4.7.0 — validate them live at 4.8.0
  (no carry-forward; there is nothing to carry from).
- **RIGOR:** R16 — if `by=` (esp. factor-by) turns out silently-wrong, that is a
  SHOWSTOPPER: stop, report immediately (the 5.0 session must know before it
  cuts). R18 — non-showstopper findings gather into the ledger. The bless bar:
  nothing rests in "documented" that is actually fixable.
- Repos: validation `/Volumes/Archive/Documents/Dropbox/Dev/pystatistics-validation`,
  library `/Volumes/Archive/Documents/Dropbox/Dev/pystatistics` (read-only for
  this session; report findings, do not fix).
- Prior art in-repo: `drivers/gam/run_tensor.py`, `run_correctness.py`,
  `subsystems/gam/meta.json`, `reports/gam-v4.8.0.md`, and the 4.8.0 handoff.

## Deliverables

1. Driver(s) under `drivers/gam/` (e.g. `run_by_nb.py`), frozen artifacts under
   `artifacts/gam/`, rendered report (extend the gam report line).
2. **A verdict handoff at `handoffs/va2-va3-gam-validation.md`** stating, for the
   waiting v5.0.0 session: **VA-2 CLOSED-BLESSABLE / VA-3 CLOSED-BLESSABLE**, or
   the findings that block. One line at the top, evidence below.
3. `subsystems/gam/meta.json` updated with the new surfaces.
