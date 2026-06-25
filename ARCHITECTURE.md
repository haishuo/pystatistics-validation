# pystatistics validation — architecture

This repo is the **canonical home for version-pinned validation evidence and
reports** for `pystatistics`. It is not a release gate and not a build artifact:
regenerating or editing a report never requires cutting a library release, and the
two evolve on independent cadences.

## The canonical flow

```
pystatistics code (a released version)
        │
        ▼
validation harness            ← lives in  ../pystatistics/pystatistics/validation/
        │                        (the machinery belongs with the code it validates)
        ▼
frozen result artifacts       ← lives HERE  artifacts/<subsystem>/v<X.Y.Z>/
        │
        ▼
canonical validation report   ← lives HERE  reports/<subsystem>-v<X.Y.Z>.md
        │
        ├──────────────► library users
        └──────────────► journal paper (a derived scholarly adaptation)
```

The paper is **not** in the middle. The validation report is the canonical
engineering artifact; a paper packages a *subset* of it together with novelty
framing and scholarly context, and **vendors a frozen snapshot** of the evidence it
relied on (plus a tagged validation commit) so a later harness rerun cannot silently
change a published figure.

## The ownership rule (the tie-breaker for "where does this live?")

> **If regenerating it could change the scientific claim about the implementation,
> it belongs to the validation system.**

| Artifact | Canonical owner |
|---|---|
| Benchmark harness, timing/record/serialization, R-comparison logic, dataset curation | **Validation** (harness in `pystatistics/validation/`) |
| Raw timing JSON, R-comparison outputs, coverage experiments, regression datasets | **Validation** (`artifacts/` here) |
| Plots used in a paper | Regenerated **from** validation evidence |
| LaTeX figure styling, related-work, novelty framing, narrative prose | **Paper** |

## Three layers / locations

1. **`../pystatistics` repo — harness home.** Reusable machinery under
   `pystatistics/validation/`: `timing` (warmup+repeats, median), `record` (the
   uniform flat-dict result schema), `device` (the env/reproducibility manifest;
   reads the version from the *imported module*, never `importlib.metadata`),
   `serialize` (freeze to the artifact schema), `rrunner` (R subprocess bridge),
   and dataset curation/IO where generally useful. It runs against a **PyPI-installed**
   pystatistics of the same version — never the local checkout. *(Migration of this
   core out of the paper `code/` dirs is in progress.)*

2. **This repo — evidence + reports.**
   - `artifacts/<subsystem>/v<X.Y.Z>/` — frozen evidence, keyed **subsystem × version ×
     device**. `manifest.json` indexes every study; `runs/` holds the run files. The
     directory name pins the **library version**; **device** (cpu/mps/cuda) is an
     internal axis, not a separate report. Schema: `artifacts/schema/`.
   - `subsystems/<subsystem>/meta.json` — the stable, non-numeric prose (procedure,
     algorithms, reference description, tolerance policy, limitations). Hand-authored,
     version-controlled; merged with artifacts at render time.
   - `reports/<subsystem>-v<X.Y.Z>.md` — the rendered, version-pinned report. **One per
     subsystem**, never hand-edited (numbers come only from artifacts).
   - `render_report.py` + `render/` — the generation path. `templates/report-template.md`
     documents the seven-question structure every report answers.
   - `drivers/` — per-subsystem scripts that invoke the harness to (re)generate artifacts.
   - `_archive/` — the original sanity-check suite, kept inert for its salvageable R
     reference values (second-wave reports), not run.

3. **`../../UMassD/Papers/*` — derived papers.** Consume a vendored snapshot of
   `artifacts/` + a canonical-commit pointer. Never live-reference mutable artifacts.

## Every report answers seven questions

1. What statistical procedure is implemented? 2. Which algorithms are used? 3. How
does it compare to R/SAS/another reference? 4. What numerical tolerances are
expected? 5. What benchmarks were run? 6. What are the known limitations? 7. Which
version of pystatistics do these results apply to?

Questions 1, 2, 4, 6 are prose (`subsystems/<s>/meta.json`); 3, 5, 7 are rendered
from frozen artifacts. Numbers are never authored in a report.

## Roadmap

- **Wave 1 (in progress):** `mvnmle` (v3.18.0) and `mice` — they have mature
  harnesses, manuscripts, and frozen results; they prove the pipeline end to end.
- **Wave 2:** rebuild the broad correctness corpus (lm, glm, anova, pca, survival,
  descriptive, …) under this architecture, salvaging the R references in `_archive/`.
- **Wave 3:** `pystatsbio` gets its own analogous validation architecture (harness
  in the pystatsbio repo, reports for its subsystems).

## Invariants

- **PyPI only.** Reports validate a released version. A canonical report must not be
  rendered from an editable/local install. If the code installs from a local
  checkout, the code is wrong.
- **Render, never hand-write.** A number in a report must be traceable to a file in
  `artifacts/`. Fix the artifact or the renderer; never type a number into a report.
- **Versioned and immutable.** A report is immutable for a given library version. A
  new version gets a new `artifacts/<s>/v<new>/` and a new `reports/<s>-v<new>.md`.
