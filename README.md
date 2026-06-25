# pystatistics-validation

Canonical home for **version-pinned validation evidence and reports** for
[pystatistics](https://pypi.org/project/pystatistics/). Each report states
*"this validates pystatistics `<subsystem>` v`<X.Y.Z>`"*, is rendered from frozen
measurement artifacts (never hand-written), and is immutable for that library
version.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full design (the canonical flow,
the ownership rule, and how papers consume this evidence).

## Layout

```
artifacts/<subsystem>/v<X.Y.Z>/   frozen evidence (manifest.json + runs/), keyed
                                  subsystem × version × device
subsystems/<subsystem>/meta.json  per-subsystem prose (procedure, algorithms, ...)
reports/<subsystem>-v<X.Y.Z>.md   rendered, version-pinned reports
render_report.py + render/        the generation path
templates/report-template.md      the seven-question report structure
drivers/                          scripts that (re)generate artifacts via the harness
artifacts/schema/                 the artifact JSON schema + docs
_archive/                         the original sanity-check suite, kept inert
```

## Rendering a report

Reports are generated from frozen artifacts — never edited by hand:

```bash
python render_report.py mvnmle 3.18.0      # -> reports/mvnmle-v3.18.0.md
python render_report.py --all              # render every artifact set present
```

## Generating artifacts (the harness)

The harness that produces the frozen artifacts lives **with the library** in
`../pystatistics/pystatistics/validation/`, and runs against a **PyPI-installed**
pystatistics of the exact version being validated (never a local checkout). The
per-subsystem `drivers/` here invoke it. *(Harness migration in progress — see the
roadmap in ARCHITECTURE.md.)*

## Status

- **mvnmle v3.18.0** — report rendered from a bootstrap snapshot of the
  forward-Cholesky paper's results; harness regeneration pending.
- **mice** — next.
- Broad correctness corpus (lm/glm/anova/pca/…) and pystatsbio — later waves.

## Tests

```bash
pytest          # renderer unit tests (tests/)
```
