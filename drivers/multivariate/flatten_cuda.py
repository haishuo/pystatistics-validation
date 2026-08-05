"""Flatten the self-contained Forge CUDA artifact into renderer-ready CSVs.

One job: deterministically derive flat summary CSVs (one row per measurement) from
``cuda_forge.json`` so the renderer can table them. No new numbers are computed --
this only reshapes frozen CUDA results into the flat form the report renderer reads
(render-from-artifacts).

    python -m drivers.multivariate.flatten_cuda artifacts/multivariate/v4.4.0/runs
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_root import artifact_root  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def flatten(runs_dir: Path) -> None:
    data = json.loads((runs_dir / "cuda_forge.json").read_text())

    # CF-1 gate boundary: one row per (method, cond).
    cf1_rows = []
    for method, rows in data["cf1_gate"]["sweep"].items():
        for r in rows:
            cf1_rows.append({
                "method": method,
                "cond_X": r.get("cond_X"),
                "outcome": r.get("outcome"),
                "subspace_angle_deg": r.get("subspace_angle_deg"),
                "sdev_max_rel": r.get("sdev_max_rel"),
                "accepted_correct": r.get("accepted_correct"),
                "force_bypasses": r.get("force_bypasses"),
            })
    _write_csv(runs_dir / "cuda_cf1_gate_summary.csv", cf1_rows)

    _write_csv(runs_dir / "cuda_r11_isolation_summary.csv",
               data["r11_isolation"]["shapes"])
    _write_csv(runs_dir / "cuda_randomized_summary.csv",
               data["randomized"]["shapes"])
    print(f"flattened CUDA artifact -> 3 CSVs in {runs_dir} "
          f"(cf1 rows={len(cf1_rows)}, silent_wrong={data['cf1_gate']['silent_wrong_count']})")


def main() -> None:
    # Default is relative to the current directory, as it always was; the only
    # change is that it now honours VALIDATION_ARTIFACT_ROOT like every other
    # artifact path. An explicit argv[1] still wins.
    runs_dir = (Path(sys.argv[1]) if len(sys.argv) > 1
                else artifact_root(Path.cwd()) / "multivariate/v4.4.0/runs")
    flatten(runs_dir)


if __name__ == "__main__":
    main()
