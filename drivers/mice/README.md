# drivers/mice — canonical mice evidence generators

Native generators for the `mice` subsystem, built on the shared harness `pystatsval`.
They emit the canonical `validation-run/v1` schema directly (superseding the
`drivers/_bootstrap/convert_mice.py` bridge, which only flattened the paper's legacy
result format). The gpu-mice paper is a downstream consumer of what these produce.

| script | job |
|---|---|
| `run_pystatistics.py` | fit + time `pystatistics.mice.mice` (cpu/gpu) → canonical record, incl. peak memory (via `pystatsval.measure`) |
| `run_r_mice.py` + `_r/mice_run.R` | fit + time R `mice` (the reference); returns timing record + per-column imputed values for fidelity |
| `generate_scaling.py` | end-to-end: synthetic MCAR sweep × backend → a serialized run file |
| (shared) `../_shared/dgp.py` | seeded data-generating processes (known-truth linear + mixed-type) and MCAR/MAR missingness |

## Requirements (PyPI only — never a local pystatistics checkout)

```bash
pip install "pystatistics==3.16.3"          # the HEADLINE version, from PyPI
pip install -e ../../../pystatistics/validation   # pystatsval harness
# R: install.packages("mice")  (reference pinned at mice 3.19.0)
```

The headline mice report pins **3.16.3**. `generate_scaling.py` stamps the run with
whatever version is imported and refuses a non-PyPI install (`--allow-editable` for
smoke tests only). CPU + MPS run on a Mac; **CUDA requires Forge** (Dev `OPERATIONS.md`).
On macOS set `KMP_DUPLICATE_LIB_OK=TRUE`.

```bash
python generate_scaling.py --out ../../artifacts/mice/v3.16.3/runs/scaling_cpu.json \
    --n 2000,5000,20000 --backends cpu,gpu --m 100 --maxit 10
```

## Verified

Against the locally-installed PyPI pystatistics (3.18.0) on synthetic data:
`run_pystatistics` (cpu) fits + times mice and captures peak memory; `run_r_mice`
shells out to R mice 3.19.0 and returns imputed values + elapsed — both emit valid
`validation-run/v1` records. Regenerating the canonical **3.16.3** artifacts (and the
GPU rows) needs a PyPI 3.16.3 env / Forge; the driver is ready for it.

## Studies still on the bootstrap bridge

`generate_scaling.py` covers the scaling study. The other studies (r-fidelity,
validity/coverage, categorical, vs-miceforest) currently reach the canonical schema
via `drivers/_bootstrap/convert_mice.py`; porting each to a native `generate_*.py`
on these primitives is follow-on work (the fidelity/coverage reductions live in the
vendored paper drivers `gpu-mice-paper/replication/vendor/code/bench_*.py`).
