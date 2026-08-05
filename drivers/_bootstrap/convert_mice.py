#!/usr/bin/env python3
"""One-shot bootstrap: flatten the gpu-mice-paper legacy result JSONs into the
canonical `validation-run/v1` schema + a manifest for mice v3.16.3.

The mice paper stored results as lists of `{study, config, metrics, env}` envelopes
with nested per-engine metrics — a different shape from the flat `records[]` schema
the validation renderer consumes. This converter is a TEMPORARY bridge: it lets the
canonical renderer work on the existing evidence. Going forward, `drivers/mice/` will
emit the canonical schema directly via `pystatsval`, and this script is retired.

It reads the paper results READ-ONLY and writes into this repo's
`artifacts/mice/v3.16.3/`. Run: `python drivers/_bootstrap/convert_mice.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

# Guard artifact writes: refuse to clobber evidence committed to git unless
# PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1. See drivers/_shared/artifact_guard.py.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import guard_artifact_path  # noqa: E402

from artifact_root import artifact_root  # noqa: E402


REPO = Path(__file__).resolve().parent.parent.parent
PAPER = Path("/Volumes/Archive/Documents/Dropbox/UMassD/Papers/gpu-mice-paper/results")
OUT = artifact_root(REPO) / "mice" / "v3.16.3"
RUNS = OUT / "runs"
LEGACY = RUNS / "legacy"


def _device(env: dict, backend: str) -> str:
    if backend == "cpu":
        return "cpu"
    return "cuda" if env.get("cuda") else ("mps" if env.get("mps") else "cpu")


def _env_block(envs: list[dict], device: str, host: str) -> dict:
    e = envs[0].get("env", {})
    return {
        "pystatistics_version": e.get("pystatistics_version"),
        "install_source": "editable",  # honest: paper ran editable installs (stale-label era)
        "python": e.get("python"),
        "torch": e.get("torch"),
        "os": e.get("platform"),
        "device": device,
        "host": host,
        "note": "bootstrap snapshot flattened from gpu-mice-paper; editable-install provenance",
    }


def _rec(engine, dataset, *, n=None, p=None, backend_name=None, median_s=None,
         min_s=None, **extra):
    r = {"engine": engine, "dataset": dataset, "n": n, "p": p,
         "backend_name": backend_name, "loglik": None, "n_iter": None,
         "converged": None, "wall_median_s": median_s, "wall_min_s": min_s,
         "wall_max_s": None, "wall_times_s": None}
    r.update({k: v for k, v in extra.items() if v is not None})
    return r


def conv_scaling(envs):
    recs = []
    for e in envs:
        c, m = e["config"], e["metrics"]
        b = c["backend"]
        recs.append(_rec(
            f"pystatistics:{b}", f"n{c['n']}", n=c["n"], p=c["p"],
            backend_name=f"mice_{b}", median_s=m["median_s"], min_s=m.get("min_s"),
            peak_mem_mb=round(m["peak_mem_bytes"] / 1e6, 1) if m.get("peak_mem_bytes") else None,
            m_imputations=c.get("m"), maxit=c.get("maxit"), method=c.get("method")))
    return recs


def conv_fidelity(envs, *, kind="numeric"):
    recs = []
    for e in envs:
        m = e["metrics"]
        n, p, method = m["n"], m["p"], m.get("method", "mixed")
        ds = f"n{n}_{method}"
        ps = m["pystatistics"]
        # agreement scalar: max |mean diff| (numeric) or max wasserstein (mixed)
        fid = m.get("fidelity", {})
        agree = None
        if kind == "numeric":
            diffs = [abs(v["pystatistics"]["mean"] - v["r_mice"]["mean"])
                     for v in fid.values() if "pystatistics" in v and "r_mice" in v]
            agree = max(diffs) if diffs else None
        else:
            ws = [v.get("wasserstein") for v in fid.values() if v.get("wasserstein") is not None]
            agree = max(ws) if ws else None
        recs.append(_rec("pystatistics:cpu", ds, n=n, p=p, backend_name="mice_cpu",
                         median_s=ps["cpu"]["median_s"], min_s=ps["cpu"].get("min_s"),
                         method=method, fidelity_max_diff=agree,
                         speedup_cpu_over_r=m.get("speedup_cpu_over_r")))
        recs.append(_rec("pystatistics:gpu", ds, n=n, p=p, backend_name="mice_gpu",
                         median_s=ps["gpu"]["median_s"], min_s=ps["gpu"].get("min_s"),
                         method=method, speedup_gpu_over_r=m.get("speedup_gpu_over_r")))
        recs.append(_rec("r:mice", ds, n=n, p=p, backend_name="r_mice",
                         median_s=m["r_mice"]["elapsed_s"],
                         r_version=m["r_mice"].get("version")))
    return recs


def conv_validity(envs):
    recs = []
    for e in envs:
        c, m = e["config"], e["metrics"]
        fmi = m.get("mean_fmi")
        if isinstance(fmi, list):  # per-coefficient → scalar mean (ignoring NaN)
            vals = [x for x in fmi if isinstance(x, (int, float)) and x == x]
            fmi = round(sum(vals) / len(vals), 4) if vals else None
        recs.append(_rec(
            f"arm:{c['arm']}", f"{c['mechanism']}_{c['method']}", n=c["n"], p=c["p"],
            backend_name=c["arm"], mechanism=c["mechanism"], method=c["method"],
            coverage_slopes_mean=m.get("coverage_slopes_mean"),
            mean_fmi=fmi, n_replications=m.get("n_replications")))
    return recs


def conv_miceforest(envs):
    recs = []
    for e in envs:
        m = e["metrics"]
        n, ds = m["n"], f"n{m['n']}"
        for key, eng in [("pystat_gpu", "pystatistics:gpu"),
                         ("pystat_cpu", "pystatistics:cpu"),
                         ("miceforest_gpu", "miceforest:gpu"),
                         ("miceforest_cpu", "miceforest:cpu")]:
            blk = m.get(key) or {}
            recs.append(_rec(eng, ds, n=n, p=m.get("p"), backend_name=key,
                             median_s=blk.get("elapsed_s"),
                             error=blk.get("failed")))
    return recs


# (source file, study id, converter, study meta) ------------------------------
JOBS = [
    ("scaling_mps_v3153.json", "scaling_mps", conv_scaling,
     dict(title="Numeric scaling: CPU vs MPS (vary n at m=100)", device=["cpu", "mps"],
          claim="speed", render="device_pivot", host="mac")),
    ("scaling_cuda_v315.json", "scaling_cuda", conv_scaling,
     dict(title="Numeric scaling: CPU vs CUDA (vary n at m=100)", device=["cpu", "cuda"],
          claim="speed", render="device_pivot", host="forge")),
    ("r_fidelity_cuda.json", "r_fidelity_cuda",
     lambda e: conv_fidelity(e, kind="numeric"),
     dict(title="Speed & fidelity vs R mice (CUDA, pmm/norm)", device=["cpu", "cuda"],
          claim="agreement+speed", host="forge",
          agreement=dict(reference_engine="r:mice", sut_run="runs/mice_r_fidelity_cuda.json",
                         sut_engine="pystatistics:cpu", pair_key=["dataset"]))),
    ("r_fidelity_mps_v315.json", "r_fidelity_mps",
     lambda e: conv_fidelity(e, kind="numeric"),
     dict(title="Speed & fidelity vs R mice (MPS, pmm/norm)", device=["cpu", "mps"],
          claim="agreement+speed", host="mac",
          agreement=dict(reference_engine="r:mice", sut_run="runs/mice_r_fidelity_mps.json",
                         sut_engine="pystatistics:cpu", pair_key=["dataset"]))),
    ("validity.json", "validity", conv_validity,
     dict(title="Rubin coverage (MCAR/MAR; MICE vs naive arms)", device=["cpu", "mps"],
          claim="coverage", host="mac")),
    ("categorical_cuda.json", "categorical_cuda",
     lambda e: conv_fidelity(e, kind="mixed"),
     dict(title="Mixed-type imputation vs R mice (CUDA)", device=["cpu", "cuda"],
          claim="agreement+speed", host="forge",
          agreement=dict(reference_engine="r:mice", sut_run="runs/mice_categorical_cuda.json",
                         sut_engine="pystatistics:cpu", pair_key=["dataset"]))),
    ("vs_miceforest_cuda.json", "vs_miceforest", conv_miceforest,
     dict(title="Scale comparison vs miceforest (ML imputer)", device=["cpu", "cuda"],
          claim="speed", host="forge")),
]


def main() -> int:
    LEGACY.mkdir(parents=True, exist_ok=True)
    studies = []
    for src, sid, conv, meta in JOBS:
        envs = json.loads((PAPER / src).read_text())
        recs = conv(envs)
        device = meta["device"][-1]  # representative non-cpu device
        host = meta.get("host", "")
        run = {
            "schema": "validation-run/v1",
            "env": _env_block(envs, device, host),
            "config": {"source_file": src, "study": sid, "n_envelopes": len(envs)},
            "records": recs,
        }
        run_name = f"mice_{sid}.json"
        guard_artifact_path((RUNS / run_name))
        (RUNS / run_name).write_text(json.dumps(run, indent=2))
        guard_artifact_path((LEGACY / src))
        (LEGACY / src).write_text(json.dumps(envs, indent=2))  # vendor original
        study = {"id": sid, "run": f"runs/{run_name}", **meta}
        studies.append(study)
        print(f"  {src} -> runs/{run_name} ({len(recs)} records)")

    manifest = {
        "schema": "validation-artifact-manifest/v1",
        "subsystem": "mice",
        "pystatistics_version": "3.16.3",
        "install_source": "pypi",
        "evidence_state": "bootstrap-snapshot",
        "frozen_utc": "2026-06-25",
        "provenance": {
            "snapshotted_from": "UMassD/Papers/gpu-mice-paper/results",
            "note": ("Bootstrap: legacy {study,config,metrics,env} envelopes flattened into "
                     "validation-run/v1 by drivers/_bootstrap/convert_mice.py. Originals vendored "
                     "under runs/legacy/. Headline version 3.16.3 (paper MANIFEST 'current library'); "
                     "individual studies span 3.13.0-3.16.3 (recorded in each run's env). Going "
                     "forward, drivers/mice/ regenerates these via pystatsval against a PyPI install."),
            "version_span": "3.13.0 (validity, version-stable), 3.15.0/3.15.3 (scaling), 3.16.2/3.16.3 (categorical, miceforest)",
        },
        "hosts": {
            "mac": {"role": "MPS + CPU", "device": "Apple Silicon / Metal (MPS)"},
            "forge": {"role": "CUDA + CPU", "device": "NVIDIA RTX 5070 Ti (CUDA)"},
            "r": {"role": "reference", "engine": "R mice 3.19.0"},
        },
        "reference": {
            "name": "R mice",
            "kind": "cran-package",
            "what": "The CRAN mice package (van Buuren & Groothuis-Oudshoorn) — multiple imputation by chained equations. pystatistics mirrors its defaults (m=5, maxit=5, pmm for numeric) and methods (pmm/norm/logreg/polyreg/polr).",
        },
        "studies": studies,
        "record_fields": {
            "wall_median_s": "median wall-clock over timed repeats",
            "peak_mem_mb": "peak device/host memory (MB)",
            "fidelity_max_diff": "max per-variable distribution distance vs R (mean diff for numeric, Wasserstein for mixed)",
            "coverage_slopes_mean": "Rubin CI coverage averaged over slope coefficients (target ~0.95)",
            "speedup_cpu_over_r / speedup_gpu_over_r": "pystatistics speedup over R mice",
        },
    }
    guard_artifact_path((OUT / "manifest.json"))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT/'manifest.json'} with {len(studies)} studies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
