"""Fold the self-contained Forge CUDA GRM study JSON into the v<ver> artifact set.

The CUDA GRM study runs on Forge via a self-contained script (no pystatsval on the
box); this flattens its raw JSON output into the same artifact shape the local
studies produce — a run envelope + CF-1-gate and speed summary CSVs — under
``artifacts/mixed/v<ver>/runs/``. Provenance (torch build, GPU, compute cap) is
carried from the raw JSON's ``env``, not synthesised locally.

    python -m drivers.mixed.flatten_grm_cuda <raw_cuda.json>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    raw = json.loads(Path(sys.argv[1]).read_text())
    ver = raw["pystatistics_version"]
    env = raw["env"]
    out_dir = _REPO / "artifacts" / "mixed" / f"v{ver}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = {
        "schema": "validation-run/v1",
        "env": {"pystatistics_version": ver, "device": env["device"],
                "host": "forge", "torch": env["torch"], "gpu": env["gpu"],
                "compute_cap": env["compute_cap"], "install_source": "pypi"},
        "config": {"study": "grm_gpu", "device": "cuda",
                   "reference": "self-contained Forge CUDA study "
                                "(drivers/mixed/generate_grm_gpu.py logic, run in a "
                                "throwaway venv layered on the gpumice torch); "
                                "silent_wrong_count must be 0; GPU must beat CPU",
                   "silent_wrong_count": raw["silent_wrong_count"]},
        "cf1_gate": raw["cf1_gate"],
        "speed": raw["speed"],
    }
    run_path = out_dir / "grm_gpu_cuda_forge.json"
    run_path.write_text(json.dumps(run, indent=2))
    _write_csv(out_dir / "grm_cf1_gate_cuda_forge_summary.csv",
               [{"study": "cf1_gate", "device": "cuda", **r} for r in raw["cf1_gate"]])
    _write_csv(out_dir / "grm_speed_cuda_forge_summary.csv",
               [{"study": "speed", "device": "cuda", **r} for r in raw["speed"]])
    print(f"wrote {run_path} (silent_wrong_count={raw['silent_wrong_count']})")


if __name__ == "__main__":
    main()
