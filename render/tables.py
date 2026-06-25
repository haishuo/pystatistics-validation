"""Build markdown tables from records / summary rows.

One job: pure formatting. No file I/O, no claim logic — turn lists of dicts into
markdown table strings, with a couple of subsystem-agnostic derivations (numeric
formatting, CPU-vs-accelerator pairing for scaling studies).
"""

from __future__ import annotations

from typing import Any, Iterable


def _fmt(v: Any) -> str:
    """Format a cell: compact floats, pass through everything else."""
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return ""
    if abs(f) >= 1000 or (abs(f) < 1e-3 and f != 0):
        return f"{f:.4g}"
    return f"{f:.4f}".rstrip("0").rstrip(".") if "." in f"{f:.4f}" else f"{f:.0f}"


def _sortkey(t: tuple) -> tuple:
    """Robust sort key for a problem-key tuple of any length: numbers sort
    numerically and before strings, component by component."""
    out = []
    for x in t:
        try:
            out.append((0, float(x), ""))
        except (TypeError, ValueError):
            out.append((1, 0.0, str(x)))
    return tuple(out)


def _esc(s: str) -> str:
    """Escape pipes so cell/header content can't break the markdown table."""
    return s.replace("|", "\\|")


def md_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    """Render a markdown table. Empty rows → an italic 'no rows' line."""
    rows = [[_esc(_fmt(c)) for c in r] for r in rows]
    if not rows:
        return "_(no rows)_\n"
    head = "| " + " | ".join(_esc(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}\n"


def table_from_csv(header: list[str], rows: list[dict[str, str]],
                   columns: list[str] | None = None) -> str:
    """Render selected columns of a summary CSV as markdown."""
    cols = columns or header
    cols = [c for c in cols if c in header]
    return md_table(cols, ([row.get(c, "") for c in cols] for row in rows))


# Columns we surface from a generic `record` when a study has no summary CSV.
_RECORD_COLS = [
    ("dataset", lambda r: r.get("survey") or r.get("dataset")),
    ("p", lambda r: r.get("p")),
    ("n", lambda r: r.get("n")),
    ("engine", lambda r: r.get("backend_name") or r.get("engine")),
    ("converged", lambda r: r.get("converged")),
    ("n_iter", lambda r: r.get("n_iter")),
    ("loglik", lambda r: r.get("loglik")),
    ("wall_median_s", lambda r: r.get("wall_median_s")),
]


def table_from_records(records: list[dict[str, Any]]) -> str:
    """Render a compact view of raw records (the columns most subsystems share)."""
    present = [(name, fn) for name, fn in _RECORD_COLS
               if any(fn(r) is not None for r in records)]
    headers = [name for name, _ in present]
    rows = ([fn(r) for _, fn in present] for r in records)
    return md_table(headers, rows)


def agreement_pivot(ref_records: list[dict[str, Any]],
                    sut_records: list[dict[str, Any]],
                    key=("survey", "p"),
                    ref_label="R", sut_label="pystatistics") -> str:
    """Pair a system-under-test against a reference by problem key.

    Shows numerical agreement (|Δloglik|, relative Δloglik, Frobenius-norm of the
    covariance estimate) and the speed ratio (reference / SUT wall-clock). Rows the
    two engines do not share are dropped (only matched problems can be compared).
    """
    def kof(r: dict[str, Any]) -> tuple:
        return tuple(r.get(k) for k in key)

    ref = {kof(r): r for r in ref_records}
    sut = {kof(r): r for r in sut_records}
    paired = sorted(set(ref) & set(sut), key=_sortkey)

    # Show only the agreement columns the data actually carries: log-likelihood
    # (MLE subsystems like mvnmle) and/or a fidelity distance (stochastic
    # subsystems like mice). Speed is always shown.
    has_ll = any(s.get("loglik") is not None for s in sut_records)
    has_fro = any(s.get("sigma_fro") is not None for s in sut_records)
    has_fid = any(s.get("fidelity_max_diff") is not None for s in sut_records)

    headers = ["problem", "p", "n"]
    if has_ll:
        headers += [f"loglik ({sut_label})", f"loglik ({ref_label})", "|Δloglik|", "relΔ"]
    if has_fro:
        headers += [f"σ_fro ({sut_label})", f"σ_fro ({ref_label})"]
    if has_fid:
        headers += [f"fidelity max dist vs {ref_label}"]
    headers += [f"wall {sut_label} (s)", f"wall {ref_label} (s)", "speedup"]

    rows = []
    for k in paired:
        r, s = ref[k], sut[k]
        row = [s.get("survey") or s.get("dataset"), s.get("p"), s.get("n")]
        if has_ll:
            lr, ls = r.get("loglik"), s.get("loglik")
            dll = abs(ls - lr) if (lr is not None and ls is not None) else None
            rel = (dll / abs(lr)) if (dll is not None and lr) else None
            row += [ls, lr, dll, rel]
        if has_fro:
            row += [s.get("sigma_fro"), r.get("sigma_fro")]
        if has_fid:
            row += [s.get("fidelity_max_diff")]
        wr, ws = r.get("wall_median_s"), s.get("wall_median_s")
        spd = (wr / ws) if (wr and ws) else None
        row += [ws, wr, (f"{spd:.0f}x" if spd else "")]
        rows.append(row)
    return md_table(headers, rows)


def device_pivot(records: list[dict[str, Any]],
                 key=None, cpu_engine="cpu", acc_engine="gpu") -> str:
    """Pivot CPU vs accelerator rows by problem key → device-axis comparison.

    Demonstrates 'device as an internal axis': pairs the CPU and accelerator rows
    for each problem and shows wall-clock + speedup, plus |Δloglik| agreement and
    peak memory *only when those fields are present* (so it serves both mvnmle —
    which has loglik — and mice — which has memory but no loglik).
    """
    if not records:
        return md_table(["problem"], [])
    if key is None:
        base = "survey" if "survey" in records[0] else "dataset"
        key = (base, "p")

    def kof(r: dict[str, Any]) -> tuple:
        return tuple(r.get(k) for k in key)

    cpu = {kof(r): r for r in records if r.get("engine", "").endswith(cpu_engine)}
    acc = {kof(r): r for r in records if r.get("engine", "").endswith(acc_engine)}
    has_ll = any(r.get("loglik") is not None for r in records)
    has_mem = any(r.get("peak_mem_mb") is not None for r in records)

    headers = ["problem", "p", "n"]
    if has_ll:
        headers += ["loglik (cpu)", "loglik (acc)", "|Δloglik|"]
    headers += ["wall cpu (s)", "wall acc (s)", "speedup"]
    if has_mem:
        headers += ["mem cpu (MB)", "mem acc (MB)"]

    rows = []
    for k in sorted(set(cpu) & set(acc), key=_sortkey):
        c, a = cpu[k], acc[k]
        wc, wa = c.get("wall_median_s"), a.get("wall_median_s")
        spd = (wc / wa) if (wc and wa) else None
        row = [c.get("survey") or c.get("dataset"), c.get("p"), c.get("n")]
        if has_ll:
            lc, la = c.get("loglik"), a.get("loglik")
            dll = abs(lc - la) if (lc is not None and la is not None) else None
            row += [lc, la, dll]
        row += [wc, wa, (f"{spd:.1f}x" if spd else "")]
        if has_mem:
            row += [c.get("peak_mem_mb"), a.get("peak_mem_mb")]
        rows.append(row)
    return md_table(headers, rows)
