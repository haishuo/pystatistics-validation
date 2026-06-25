# drivers/

Per-subsystem scripts that (re)generate the frozen artifacts under
`artifacts/<subsystem>/v<X.Y.Z>/` by driving the validation harness.

Each driver MUST:
1. Install/verify pystatistics is the intended version **from PyPI** (fail loud if
   the import resolves to an editable/local checkout — see the `install_source`
   guard in `pystatistics/validation/device.py`).
2. Run the benchmarks/comparisons via `pystatistics.validation`.
3. Serialize results to the artifact schema (`artifacts/schema/`), writing
   `manifest.json` + `runs/` for the version.

A driver never writes a report — `render_report.py` does that from the artifacts.

Status: the harness core (`pystatistics/validation/`) is being extracted from the
paper `code/` dirs; until it lands, `artifacts/mvnmle/v3.18.0/` is a vendored
bootstrap snapshot (see its `manifest.json` provenance). `run_mvnmle.py` and
`run_mice.py` will be added as the harness migrates.
