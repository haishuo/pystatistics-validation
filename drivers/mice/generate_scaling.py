#!/usr/bin/env python3
"""Generate a canonical mice scaling run (CPU vs accelerator) from synthetic data.

Produces one ``validation-run/v1`` file: for each n in the grid, fit + time
pystatistics MICE on a seeded MCAR problem with each requested backend, and record
wall-clock + peak memory. This is the harness-generated counterpart to the bootstrap
``scaling_*`` studies; it emits the SAME schema the renderer consumes.

PyPI-only: the env manifest's install-source guard refuses a non-PyPI pystatistics
unless --allow-editable is passed (for local smoke tests). The headline mice version
is 3.16.3 — run this against a PyPI 3.16.3 install to refresh the canonical artifact;
running against another installed version writes to that version's directory instead.

Usage:
  python drivers/mice/generate_scaling.py --out <run.json> --n 2000,5000 \
      --backends cpu --m 5 --maxit 5 --seed 20260617
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make the sibling driver + shared modules importable when run as a script
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "_shared")]

from dgp import impose_mcar, make_complete            # noqa: E402
from run_pystatistics import run_mice_record           # noqa: E402

from pystatsval import device, serialize               # noqa: E402


def _resolve_device(backends: list[str]) -> str:
    """One env device label for the run (a scaling run pairs cpu vs one accelerator)."""
    if "gpu" not in backends:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", default="2000,5000",
                    help="comma-separated row counts")
    ap.add_argument("--backends", default="cpu",
                    help="comma-separated: cpu,gpu")
    ap.add_argument("--p", type=int, default=20, help="predictors (matrix has p+1 cols)")
    ap.add_argument("--rate", type=float, default=0.2)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--maxit", type=int, default=5)
    ap.add_argument("--method", default="pmm")
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--host", default=None)
    ap.add_argument("--allow-editable", action="store_true",
                    help="permit a non-PyPI install (smoke tests only)")
    args = ap.parse_args(argv)

    ns = [int(x) for x in args.n.split(",")]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    dev = _resolve_device(backends)

    env = device.env_manifest(device=dev, host=args.host)
    if not args.allow_editable:
        device.require_pypi(env)

    records = []
    for n in ns:
        complete = make_complete(n, args.p, seed=args.seed)
        incomplete = impose_mcar(complete.matrix, rate=args.rate, seed=args.seed)
        for backend in backends:
            rec = run_mice_record(
                incomplete, backend=backend, dataset=f"n{n}",
                m=args.m, maxit=args.maxit, method=args.method, seed=args.seed,
                repeats=args.repeats, warmup=args.warmup)
            records.append(rec)
            wall = rec.get("wall_median_s")
            print(f"  n={n} {backend}: "
                  + (f"{wall:.3f}s" if wall is not None else f"FAILED ({rec.get('error')})"))

    config = {"study": "scaling", "slice": "vary_n", "n": ns, "p": args.p,
              "rate": args.rate, "m": args.m, "maxit": args.maxit,
              "method": args.method, "seed": args.seed,
              "backends": backends, "dgp": "make_complete+impose_mcar (synthetic, seeded)"}
    run = serialize.build_run(env=env, config=config, records=records)
    out = serialize.write_run(args.out, run)
    print(f"wrote {out} ({len(records)} records, pystatistics "
          f"{env['pystatistics_version']} / {env['install_source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
