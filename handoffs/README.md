# handoffs/

Cross-session hand-off and investigation documents — **not** version-pinned
validation reports. Keeping them here stops them clogging `reports/`, which holds
only the rendered, immutable `<subsystem>-v<X.Y.Z>.md` reports.

What belongs here:
- Recommendation / coordination notes a spun-off chip hands back to the blessing
  session (e.g. `mixed-recommendation.md`).
- Investigation write-ups and their prototype scripts (e.g.
  `mixed-gpu-investigation.md` + `mixed-gpu-investigation-scripts/`).
- Any other working document that informs a decision but is not itself a report.

What does NOT belong here:
- Rendered validation reports → `reports/`.
- Frozen measurement evidence + per-run findings ledgers → `artifacts/<subsystem>/`.

## Lifecycle

1. **Active** — lives at the top of `handoffs/` while its recommendations are
   still being acted on.
2. **Archived** — once the work it drove is complete (the relevant version is
   blessed and any spun-off chip is closed), move it to `handoffs/archive/`.
   Don't delete: the rationale is the historical record of *why* a decision (e.g.
   "`mixed` is CPU-only, no GPU backend") was made.

## Current

_(none active)_

## Archived

- `mixed-*` (recommendation, GPU investigation + scripts, implementation handback,
  F3-fix handback) — the full hand-off trail behind `mixed` **v4.5.1** (blessed
  2026-07-01: GPU verdict B — general LMM/GLMM CPU-only, a separate `grm_lmm()`
  GPU model; F1/F2a/F2b/F3 resolved, F4 documented). See
  `reports/mixed-v4.5.1.md` and `artifacts/mixed/v4.5.1/findings_ledger.md`.
