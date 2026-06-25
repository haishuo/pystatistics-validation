# Artifact schema

Frozen evidence lives under `artifacts/<subsystem>/v<X.Y.Z>/`. Three kinds of file:

```
artifacts/<subsystem>/v<X.Y.Z>/
  manifest.json        # canonical index for this (subsystem, version) — REQUIRED
  runs/<run>.json      # one benchmark run: env + config + records[]
  runs/<run>.csv       # optional flat summary of the same run (one row per record)
```

The **manifest** is the contract the renderer reads. It names every study, which
device(s) it covers, which run file(s) back it, the reference compared against, and
the provenance/host/version facts. The renderer never discovers files by globbing —
it reads the manifest, so an artifact that isn't listed is not part of the report.

## `manifest.json` — `validation-artifact-manifest/v1`

| field | meaning |
|---|---|
| `schema` | `"validation-artifact-manifest/v1"` |
| `subsystem` | e.g. `"mvnmle"`, `"mice"` |
| `pystatistics_version` | exact version validated — MUST match the directory `v<X.Y.Z>` |
| `install_source` | `"pypi"` (canonical) or `"editable"` (flagged as non-canonical) |
| `frozen_utc` | date the evidence was frozen |
| `provenance` | where the artifacts came from; any caveats (e.g. stale embedded version labels) |
| `hosts` | map of host id → {role, device, precision} |
| `reference` | the comparison baseline (R package, SAS, CPU, prior version) |
| `studies[]` | each study: `id`, `title`, `device[]`, `run`/`runs[]`, `summary`/`summaries[]`, `claim`, `host` |
| `record_fields` | human glossary of the per-record fields used in this subsystem |

`claim` is one of `agreement` (matches a reference numerically), `speed`
(performance/benchmark), `coverage` (statistical validity study), `stability`
(determinism / numerical robustness), or a `+`-joined combination.

## `runs/<run>.json` — `validation-run/v1`

A run is the unit the harness emits. Canonical shape:

```json
{
  "schema": "validation-run/v1",
  "env": { ... },        // see below — the reproducibility manifest
  "config": { ... },     // run parameters (surveys, p grid, reps, tol, ...)
  "records": [ { ... } ] // one per (engine x problem) cell
}
```

### `env` — the reproducibility manifest (REQUIRED for newly-generated runs)

Captured by `pystatistics/validation/device.py`. The version MUST be read from the
**imported module** (`pystatistics.__version__`), never from `importlib.metadata`
(an editable install freezes that at install time — the documented stale-label bug).

| field | meaning |
|---|---|
| `pystatistics_version` | from the imported module |
| `install_source` | `pypi` / `editable` (refuse to render a canonical report unless `pypi`) |
| `python`, `numpy`, `torch` | versions |
| `blas` | BLAS/LAPACK backend string |
| `os`, `cpu` | platform identifiers |
| `device` | `cpu` / `mps` / `cuda` |
| `gpu_name`, `cuda_version` | when device is a GPU |
| `host` | logical host id (matches `manifest.hosts`) |

> Bootstrap snapshots vendored from the papers predate this block; their provenance
> and true version are recorded in the manifest instead. New runs must carry `env`.

### `record` — one comparable measurement (`record.py` schema)

Flat dict so pystatistics (CPU/GPU) and R/reference rows compare field-for-field.
Core fields: `engine`, `backend_name`, `precision`, `parameterization`, problem
descriptors (`survey`/dataset, `p`, `n`, `n_patterns`, `missing_frac`), outcome
(`loglik`, `n_iter`, `converged`), timing (`wall_median_s`, `wall_min_s`,
`wall_max_s`, `wall_times_s`), and estimate summaries (`mu`, `sigma_diag`,
`sigma_fro`, `sigma_logdet`, `sigma_full` for small p). Subsystem-specific extras
(e.g. `n_function_evals`) are allowed; the manifest's `record_fields` documents the
ones a given subsystem relies on.

The JSON-Schema files in this directory (`manifest.schema.json`, `run.schema.json`)
are the machine-checkable version of the above; they are intentionally permissive on
subsystem-specific `record` extras.
