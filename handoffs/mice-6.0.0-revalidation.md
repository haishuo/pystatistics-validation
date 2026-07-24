# mice — 6.0.0 re-validation (handoff)

Parallel track to the mvnmle/paper work (not paper-coupled). Fresh red-team
re-validation of `pystatistics.mice` at **6.0.0** across CPU / Apple-Metal(MPS) / CUDA.

**Verdict: CLEAN — reproduces the 5.0.0 bless.** mice carries no Cython `.pyx` (the 6.0.0
migration was timeseries/survival only), so it is behaviorally unchanged; every gate
reproduces, including a *fresh* CUDA leg that the 5.0.0 report could only carry forward
from 4.6.13.

Artifacts: `artifacts/mice/v6.0.0/runs/` (mice_gate MPS, mice_gate CUDA, compare_r).
Generated via `drivers/mice/{mice_gate,compare_r_reproduce}.py` vs PyPI `pystatistics==6.0.0`,
R `mice` 3.19.0. Paper repos untouched.

## Results (6.0.0 vs 5.0.0 report)

| Check | 6.0.0 | 5.0.0 |
|---|---|---|
| CPU-vs-R fidelity (pmm/logreg/polr) | \|Δmean\|≤0.053, W1≤0.053 — **byte-identical** | same |
| fp32 gate (norm/pmm) covers truth | yes (MPS+CUDA, incl gpu_fp64) | yes |
| Rubin coverage cpu/gpu/gpu_fp64 | 0.955 / 0.935(CUDA),0.945(MPS) / 0.920 | identical |
| Separation (perfect-sep column) | frac1≈0.486, **no silent collapse** | no collapse |
| Default `mice(data, seed=0)` | ok | ok |

## Key safety properties confirmed at 6.0.0
- **No silent collapse on separation** — the CF-1 mitigation (fp64 discrete fits for
  logreg/polyreg/polr on CUDA) holds; a perfectly-separated binary column imputes to a
  non-degenerate ~0.49 class-1 fraction on cpu/gpu/gpu_fp64 alike.
- **Rubin coverage ~0.95** preserved under fp32.
- **fp32 imputations cover the known truth** (no fp32 bias).

## Not done / optional
- `generate_scaling.py` end-to-end scaling sweep (perf, not correctness) — skipped; the
  5.0.0 scaling story is unaffected by a no-op migration.
- No new findings; no code change indicated. mice needs no 6.0.x fix.

Reusable env: same persistent Forge `pystatistics-test` conda env (6.0.0 + cu128 torch).
