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

- `gam-gpu-investigation.md` (+ `gam-gpu-investigation-scripts/`) — GPU-feasibility
  verdict for `gam`, **measured on Forge CUDA** (RTX 5070 Ti, MKL CPU baseline).
  **Verdict: no meaningful win in the typical regime (n≲10k — sub-ms inner op);
  at large n the only safe wins are ~3× (fp32 augmented-QR) to ~5× (batched fp64
  multi-λ), in formulations the library doesn't have. The current `gpu_pirls`
  fp32 path is 43–78× fast but carries a CUDA-proven silent-wrong EDF band
  (33-DOF error / negative EDF, ungated CF-1); its fp64 remedy LOSES at typical p
  (0.67×).** Flags for the gam chip (task_31d5fd2b): R12 boundary sweep on the fp32
  path (MPS+CUDA), TF32 pin (R14), overstated cuSOLVER-divergence docstring.
  Informs the GPU-backend decision; archive once gam is blessed.

## Archived

- `mixed-*` (recommendation, GPU investigation + scripts, implementation handback,
  F3-fix handback) — the full hand-off trail behind `mixed` **v4.5.1** (blessed
  2026-07-01: GPU verdict B — general LMM/GLMM CPU-only, a separate `grm_lmm()`
  GPU model; F1/F2a/F2b/F3 resolved, F4 documented). See
  `reports/mixed-v4.5.1.md` and `artifacts/mixed/v4.5.1/findings_ledger.md`.
