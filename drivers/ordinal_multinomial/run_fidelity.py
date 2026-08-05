"""G2 fidelity — fail-loud, no silent substitution (both surfaces).

RIGOR A6 / Guarantee 2: neither polr nor multinom may silently solve a different
problem. Every unsupported request must RAISE, never quietly substitute (e.g. an
unsupported link must not fall back to logistic; an explicit CPU-with-GPU-tensor
must not silently migrate; l2 on the GPU must not be silently dropped). This
runner asserts the raise contract for the whole boundary.

Emits artifacts/ordinal_multinomial/v<ver>/runs/fidelity.json.
Run: python drivers/ordinal_multinomial/run_fidelity.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omcompare import to_native  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run  # noqa: E402
# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402
from pystatistics.ordinal import polr  # noqa: E402
from pystatistics.multinomial import multinom  # noqa: E402
from pystatistics.core.exceptions import ValidationError  # noqa: E402

_ART = (Path(__file__).resolve().parents[2]
        / "artifacts/ordinal_multinomial/v{ver}/runs/fidelity.json")


def _expect_raise(key, surface, desc, fn, exc=(ValidationError, RuntimeError,
                                               ValueError)) -> dict:
    """Record whether ``fn`` raised one of the expected fail-loud exceptions."""
    got, msg = "no-raise", ""
    try:
        fn()
    except exc as e:
        got, msg = type(e).__name__, str(e)[:120]
    except Exception as e:  # a raise, but not the fail-loud type we want
        got, msg = f"UNEXPECTED:{type(e).__name__}", str(e)[:120]
    passed = got not in ("no-raise",) and not got.startswith("UNEXPECTED")
    return {"key": key, "surface": surface, "desc": desc,
            "expected": "raise", "got": got, "message": msg, "pass": passed}


def run_checks() -> list[dict]:
    rng = np.random.default_rng(0)
    n = 120
    Xf = rng.standard_normal((n, 2))
    y = rng.integers(0, 3, n)
    Xm = np.column_stack([np.ones(n), Xf])
    ygap = np.where(y == 1, 2, y)
    Xnan = Xf.copy(); Xnan[0, 0] = np.nan

    checks = [
        # polr — no silent link substitution. NOTE: 'cauchit' and 'loglog' are
        # SUPPORTED links as of 5.0 (validated in run_newlinks), so they are no
        # longer fail-loud negatives here. The unknown-link contract is exercised
        # by a genuinely unsupported name below.
        _expect_raise("polr_link_unknown", "polr",
                      "an unknown link name must raise (no silent logistic fallback)",
                      lambda: polr(y, Xf, link="banana")),
        # polr — l2 only on CPU, never silently dropped on GPU
        _expect_raise("polr_l2_on_gpu", "polr",
                      "l2>0 with backend='gpu' must raise (not silently dropped)",
                      lambda: polr(y, Xf, l2=1.0, backend="gpu")),
        # polr — backend hygiene
        _expect_raise("polr_unknown_backend", "polr",
                      "unknown backend must raise",
                      lambda: polr(y, Xf, backend="tpu")),
        _expect_raise("polr_gpu_fp64_non_cuda", "polr",
                      "gpu_fp64 without CUDA must raise (MPS has no fp64)",
                      lambda: polr(y, Xf, backend="gpu_fp64")),
        # polr — input contract
        _expect_raise("polr_y_gap", "polr", "y with a missing level must raise",
                      lambda: polr(ygap, Xf)),
        _expect_raise("polr_y_nonzero_start", "polr",
                      "y not starting at 0 must raise", lambda: polr(y + 1, Xf)),
        _expect_raise("polr_y_noninteger", "polr", "non-integer y must raise",
                      lambda: polr(y + 0.5, Xf)),
        _expect_raise("polr_X_nonfinite", "polr", "non-finite X must raise",
                      lambda: polr(y, Xnan)),
        # multinom — mirror the boundary
        _expect_raise("multinom_unknown_backend", "multinom",
                      "unknown backend must raise",
                      lambda: multinom(y, Xm, backend="tpu")),
        _expect_raise("multinom_gpu_fp64_non_cuda", "multinom",
                      "gpu_fp64 without CUDA must raise",
                      lambda: multinom(y, Xm, backend="gpu_fp64")),
        _expect_raise("multinom_y_gap", "multinom",
                      "y with a missing class must raise",
                      lambda: multinom(ygap, Xm)),
        _expect_raise("multinom_X_nonfinite", "multinom",
                      "non-finite X must raise",
                      lambda: multinom(y, np.column_stack([np.ones(n), Xnan]))),
    ]
    return checks


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    checks = run_checks()
    run = build_run(
        env=env,
        config={"suite": "ordinal-multinomial-fidelity",
                "contract": "fail-loud, no silent substitution (RIGOR A6 / G2)"},
        records=to_native([{"key": "fidelity", "checks": checks}]),
    )
    out = Path(str(_ART).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    print(f"  fidelity {sum(bool(c['pass']) for c in checks)}/{len(checks)} pass")
    for c in checks:
        if not c["pass"]:
            print(f"   FAIL {c['key']}: got {c['got']}")


if __name__ == "__main__":
    main()
