"""Assemble a validation report (markdown) from a Loaded artifact set.

One job: compose the seven-question report (see templates/report-template.md) by
merging subsystem prose (meta) with tables derived from frozen run files. No claim
numbers are authored here — only read from artifacts.
"""

from __future__ import annotations

from typing import Any

from . import tables
from .load import Loaded, read_csv, read_records

# Render at most this many raw record rows per study (loud-noted when exceeded).
MAX_ROWS = 40

# Trimmed column set for the standard wall-clock summary CSVs.
_SUMMARY_COLS = ["survey", "p", "n", "engine", "converged", "n_iter",
                 "loglik", "wall_median_s", "backend_name"]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) + "\n"


def _generic_records_table(records: list[dict[str, Any]]) -> str:
    """Render records with non-standard fields: union of keys minus bulky arrays."""
    drop = {"mu", "sigma_diag", "sigma_full", "wall_times_s"}
    keys: list[str] = []
    for r in records:
        for k, v in r.items():
            # Drop bulky/structured fields and any list-valued field (it would
            # explode a table cell); summaries should be scalars in the artifact.
            if k not in keys and k not in drop and not isinstance(v, (list, dict)):
                keys.append(k)
    # Drop columns that are empty for every record (keeps the table legible).
    keys = [k for k in keys
            if any(r.get(k) not in (None, "") for r in records[:MAX_ROWS])]
    rows = ([r.get(k) for k in keys] for r in records[:MAX_ROWS])
    return tables.md_table(keys, rows)


def _study_tables(study: dict[str, Any], L: Loaded) -> str:
    """Render the evidence table(s) for one study."""
    # An explicit render hint (e.g. device_pivot) takes precedence over a summary CSV.
    # Prefer a flat summary CSV when the manifest provides one and no hint is set.
    summaries = study.get("summaries") or ([study["summary"]] if study.get("summary") else [])
    if summaries and not study.get("render"):
        # A study may override the default column set (e.g. a coefficient-agreement
        # table whose columns are nothing like the wall-clock summary's).
        cols = study.get("summary_cols", _SUMMARY_COLS)
        out = []
        for rel in summaries:
            header, rows = read_csv(L.version_dir, rel)
            out.append(tables.table_from_csv(header, rows, cols))
        return "\n".join(out)

    runs = study.get("runs") or ([study["run"]] if study.get("run") else [])
    # device_pivot pairs CPU vs accelerator by problem key, so it must see records
    # from ALL of a study's run files at once (device is an internal axis spread
    # across one run file per device). Concatenate, then pivot a single time.
    if study.get("render") == "device_pivot":
        allrec: list[dict[str, Any]] = []
        for rel in runs:
            allrec += read_records(L.version_dir, rel)
        return tables.device_pivot(allrec)

    out = []
    for rel in runs:
        if rel.endswith(".csv"):
            header, rows = read_csv(L.version_dir, rel)
            out.append(tables.table_from_csv(header, rows))
            continue
        records = read_records(L.version_dir, rel)
        if any(r.get("loglik") is not None for r in records):
            note = (f"\n_Showing first {MAX_ROWS} of {len(records)} records._\n"
                    if len(records) > MAX_ROWS else "")
            out.append(tables.table_from_records(records[:MAX_ROWS]) + note)
        else:
            out.append(_generic_records_table(records))
    return "\n".join(out)


def _filter_engine(records: list[dict[str, Any]], engine: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("engine") == engine]


def _agreement_table(study: dict[str, Any], L: Loaded) -> str:
    """Render the SUT-vs-reference agreement pivot for a study with an `agreement` block."""
    spec = study["agreement"]
    ref_records: list[dict[str, Any]] = []
    for rel in (study.get("runs") or ([study["run"]] if study.get("run") else [])):
        ref_records += _filter_engine(read_records(L.version_dir, rel), spec["reference_engine"])
    sut_records = _filter_engine(read_records(L.version_dir, spec["sut_run"]), spec["sut_engine"])
    return tables.agreement_pivot(
        ref_records, sut_records,
        key=tuple(spec.get("pair_key", ("survey", "p"))),
        ref_label=spec["reference_engine"].split(":")[-1],
        sut_label=spec["sut_engine"].split(":")[-1],
    )


def _reference_block(meta: dict[str, Any]) -> str:
    ref = meta.get("reference", {})
    lines = [ref.get("what", "")]
    also = ref.get("also_compared")
    if also:
        lines.append(f"\nAlso compared against: {', '.join(also)}.")
    return "\n".join(lines) + "\n"


def _tolerances_block(meta: dict[str, Any]) -> str:
    tol = meta.get("tolerances", {})
    return _bullets([f"**{k}:** {v}" for k, v in tol.items()])


def build(L: Loaded, rendered_utc: str) -> str:
    """Return the full markdown report for the loaded (subsystem, version)."""
    m, meta = L.manifest, L.meta
    ref = m.get("reference", {})
    hosts = "; ".join(f"**{h}** — {d.get('device', d.get('role', '?'))}"
                      for h, d in (m.get("hosts") or {}).items())
    prov = (m.get("provenance") or {}).get("note", "")

    # Q3: reference prose + the head-to-head study(ies). Agreement studies render a
    # SUT-vs-reference pivot here and are NOT repeated in Q5.
    agree_studies = [s for s in m.get("studies", []) if "agreement" in s.get("claim", "")]
    agree_md = ""
    for s in agree_studies:
        agree_md += f"\n**{s['title']}**\n\n"
        agree_md += (_agreement_table(s, L) if s.get("agreement")
                     else _study_tables(s, L)) + "\n"

    # Q5: one subsection per non-agreement study (agreement ones live in Q3).
    bench = []
    for study in m.get("studies", []):
        if "agreement" in study.get("claim", ""):
            continue
        dev = ", ".join(study.get("device", []))
        head = (f"### {study['title']}\n"
                f"_claim: {study.get('claim', '?')} · device: {dev}"
                + (f" · host: {study['host']}" if study.get("host") else "") + "_\n")
        body = _study_tables(study, L)
        note = f"\n> {study['note']}\n" if study.get("note") else ""
        bench.append(head + "\n" + body + note)

    return f"""# Validation report — pystatistics `{m['subsystem']}` v{m['pystatistics_version']}

> This report validates **pystatistics {m['subsystem']} v{m['pystatistics_version']}**, \
installed from **{m.get('install_source', '?')}**. Evidence lives in \
`artifacts/{m['subsystem']}/v{m['pystatistics_version']}/`. It was generated by \
`render_report.py` from frozen artifacts — **do not edit numbers by hand**; fix the \
artifact or the renderer and re-render.

**Subsystem:** {meta.get('title', m['subsystem'])} (`{meta.get('module', '')}`)

## 1. What statistical procedure is implemented?

{meta.get('procedure', '_TODO_')}

## 2. Which algorithms are used?

{_bullets(meta.get('algorithms', []))}
## 3. How does it compare to {ref.get('name', 'the reference')}?

{_reference_block(meta)}
{agree_md}
## 4. What numerical tolerances are expected?

{_tolerances_block(meta)}
## 5. What benchmarks were run?

{chr(10).join(bench)}
## 6. What are the known limitations?

{_bullets(meta.get('limitations', []))}
## 7. Which version of pystatistics do these results apply to?

- **Library version:** {m['pystatistics_version']}  ·  **Install source:** {m.get('install_source', '?')}
- **Frozen:** {m.get('frozen_utc', '?')}  ·  **Reference:** {ref.get('name', '?')}
- **Hosts / devices:** {hosts}
- **Provenance:** {prov}

---
_Rendered {rendered_utc} from `artifacts/{m['subsystem']}/v{m['pystatistics_version']}/manifest.json`._
"""
