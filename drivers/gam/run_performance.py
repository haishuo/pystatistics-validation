"""GAM performance validation: pystatistics CPU vs mgcv::gam (G3).

Times both engines' fit (compute only, best-of-reps) across problem size and
number of smooths, on identical Gaussian data, and records the empirical
scaling exponent. Emits artifacts/gam/v<ver>/runs/performance.json.

G3 frame (RIGOR): same estimator (penalised regression spline + GCV), same
machine, CPU-vs-CPU. mgcv's inner loop is compiled C; the honest target is
"within striking distance, explained", not "beat C".
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run
# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

import pystatistics
from pystatistics.gam import gam, s

from artifact_root import artifact_root  # noqa: E402

_HERE = Path(__file__).resolve().parent
_PERF_R = _HERE / "perf_reference.R"
_VER = pystatistics.__version__
_ARTIFACT = artifact_root(_HERE.parents[1]) / f"gam/v{_VER}/runs/performance.json"

_REPS = 5


def _make_data(n: int, m: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cols = {f"x{j}": np.sort(rng.uniform(0, 1, n)) for j in range(m)}
    eta = sum(np.sin(2 * np.pi * (j + 1) * cols[f"x{j}"]) for j in range(m))
    y = eta + rng.normal(0, 0.5, n)
    return {"y": y, **cols}


def _time_py(y, smooth_data, smooths) -> float:
    best = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(_REPS):
            t0 = time.perf_counter()
            gam(y, smooths=smooths, smooth_data=smooth_data, method="GCV")
            best = min(best, time.perf_counter() - t0)
    return best


def _time_r(data: dict[str, np.ndarray], m: int, td: Path) -> dict[str, Any]:
    csv = td / "d.csv"
    names = list(data)
    with open(csv, "w", newline="") as fh:
        fh.write(",".join(names) + "\n")
        for row in zip(*[data[nm] for nm in names]):
            fh.write(",".join(repr(float(v)) for v in row) + "\n")
    formula = "y ~ " + " + ".join(f"s(x{j},bs='cr',k=10)" for j in range(m))
    spec = {"data_csv": str(csv), "formula": formula, "family": "gaussian",
            "method": "GCV", "reps": _REPS}
    spec_path = td / "spec.json"
    out_path = td / "out.json"
    spec_path.write_text(json.dumps(spec))
    proc = subprocess.run(
        ["Rscript", str(_PERF_R), str(spec_path), str(out_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"R perf failed: {proc.stderr[:300]}")
    return json.loads(out_path.read_text())


def main() -> None:
    env = env_manifest(device="cpu")
    require_pypi(env)

    records: list[dict[str, Any]] = []
    # Axis 1: n scaling, single smooth.
    for n in (200, 1000, 5000, 20000, 50000):
        data = _make_data(n, 1, seed=n)
        py = _time_py(data["y"], {"x0": data["x0"]}, [s("x0", k=10)])
        with tempfile.TemporaryDirectory() as td:
            r = _time_r(data, 1, Path(td))
        records.append({"axis": "n", "n": n, "n_smooths": 1, "k": 10,
                        "py_s": py, "r_s": r["min_s"],
                        "ratio_py_over_r": py / r["min_s"]})
    # Axis 2: number of smooths at fixed n.
    for m in (1, 2, 4, 6):
        data = _make_data(2000, m, seed=1000 + m)
        py = _time_py(data["y"], {f"x{j}": data[f"x{j}"] for j in range(m)},
                      [s(f"x{j}", k=10) for j in range(m)])
        with tempfile.TemporaryDirectory() as td:
            r = _time_r(data, m, Path(td))
        records.append({"axis": "n_smooths", "n": 2000, "n_smooths": m,
                        "k": 10, "py_s": py, "r_s": r["min_s"],
                        "ratio_py_over_r": py / r["min_s"]})

    # Empirical scaling exponent (n axis, log-log slope).
    n_rows = [r for r in records if r["axis"] == "n"]
    logn = np.log([r["n"] for r in n_rows])
    slope_py = float(np.polyfit(logn, np.log([r["py_s"] for r in n_rows]), 1)[0])
    slope_r = float(np.polyfit(logn, np.log([r["r_s"] for r in n_rows]), 1)[0])

    run = build_run(
        env=env,
        config={"suite": "gam-performance", "reference": "mgcv::gam",
                "reps": _REPS, "estimator": "penalised regression spline + GCV",
                "empirical_exponent_py": slope_py,
                "empirical_exponent_r": slope_r},
        records=records,
    )
    write_run(_ARTIFACT, run)
    print(f"wrote {_ARTIFACT}")
    print(f"{'axis':<12}{'n':>8}{'m':>4}{'py_ms':>10}{'r_ms':>10}{'py/r':>8}")
    for r in records:
        print(f"{r['axis']:<12}{r['n']:>8}{r['n_smooths']:>4}"
              f"{r['py_s']*1e3:>10.1f}{r['r_s']*1e3:>10.1f}"
              f"{r['ratio_py_over_r']:>8.2f}")
    print(f"empirical exponent  py={slope_py:.2f}  r={slope_r:.2f}")


if __name__ == "__main__":
    main()
