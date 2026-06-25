"""Report renderer: turn frozen artifacts into a version-pinned validation report.

One job: read `artifacts/<subsystem>/v<version>/manifest.json` plus the per-subsystem
prose in `subsystems/<subsystem>/meta.json`, and emit a markdown report. Numbers are
NEVER authored here — they are read from the frozen run files. Prose is read from the
subsystem meta. See ARCHITECTURE.md and templates/report-template.md.
"""
