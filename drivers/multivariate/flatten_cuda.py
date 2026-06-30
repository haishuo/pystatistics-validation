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
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "artifacts/multivariate/v4.4.0/runs")
    flatten(runs_dir)


if __name__ == "__main__":
    main()
