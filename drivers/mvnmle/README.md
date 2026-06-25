# drivers/mvnmle — canonical mvnmle evidence generators

These scripts regenerate the frozen artifacts under `artifacts/mvnmle/v<X.Y.Z>/`.
They were migrated out of `forward-cholesky-mvnmle-paper/code/` and re-based on the
shared harness `pystatsval`; the paper is now a downstream consumer (see its
`replication/vendor/` snapshot).

| script | job |
|---|---|
| `run_pystatistics.py` | fit + time `pystatistics.mvnmle.mlest` (cpu/gpu) → canonical record |
| `run_r_mvnmle.py` + `r_mvnmle_timing.R` | fit + time R `mvnmle::mlest` (the reference; p ≤ 50) |
| `bench_endtoend.py` | end-to-end GPU fits across surveys × p (Table 1 / CUDA + MPS) |
| `factorial_ablation.py` | trace × gradient per-evaluation factorial (Fig. 2) |
| `trace_backend_compare.py` | per-pattern trace-term microbenchmark (Fig. 3) |
| (shared) `../_shared/survey_io.py`, `curate.py` | survey `.h5` → curated MVN problem |

## Requirements (PyPI only — never a local pystatistics checkout)

```bash
pip install "pystatistics==3.18.0"          # the version being validated, from PyPI
pip install -e ../../../pystatistics/validation   # pystatsval harness
export MVNMLE_DATA_DIR=/path/to/survey/h5    # wvs.h5, gss.h5 (see paper data/PROVENANCE.md)
```

CPU + MPS run on a Mac; **CUDA numbers require Forge** (per Dev `OPERATIONS.md`).
On macOS set `KMP_DUPLICATE_LIB_OK=TRUE` (libomp double-init).

## Verified

`run_pystatistics` (cpu) vs `run_r_mvnmle` on WVS p=5 (pystatistics 3.18.0 from
PyPI): |Δloglik| = 3.3e-6, |Δσ_fro| = 1.4e-5, 56× faster — confirming the migrated
driver reproduces the agreement claim through `pystatsval`.
