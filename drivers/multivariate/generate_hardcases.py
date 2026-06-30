"""Generate the PCA R10 hard-case + R15 default-invocation artifacts (CPU).

One job: probe PCA in the adversarial regime where R-agreement is most likely to
break AND where R itself changes behaviour, and demand pystatistics MATCH R --
including R's failures (RIGOR R10). Each case is a deterministic synthetic design;
both engines analyse the identical numbers. Two outcome types:

  - **numeric agreement** — both decompose; we sign-align and reduce sdev /
    rotation / scores to agreement metrics (eigenvalues are well-defined even when
    near-degenerate eigenvectors are not; both call LAPACK SVD on identical bytes).
  - **matched refusal** — both must FAIL on the same design (e.g. a constant
    column under scale=TRUE: R's prcomp errors "cannot rescale a constant/zero
    column"; pystatistics raises ValidationError). We record both messages.

R15: the default invocation ``pca(X)`` (center=True, scale=False, all components)
is included as its own case, not only tuned calls.

    python -m drivers.multivariate.generate_hardcases --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from drivers.multivariate.run_pystatistics import (
    run_pca_record, strip_arrays as _strip_arrays)
from drivers.multivariate.run_r_multivariate import run_r_pca_record

_REPO = Path(__file__).resolve().parent.parent.parent
_SEED = 20260630  # documented deterministic seed for the synthetic hard cases


# ── deterministic adversarial designs ─────────────────────────────────────────

def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def dgp_near_collinear() -> tuple[np.ndarray, list[str], str]:
    """n=200, p=5; col4 = col3 + 1e-8*noise -> cond(X) ~ 1e8, one tiny eigenvalue."""
    rng = _rng()
    n = 200
    base = rng.standard_normal((n, 4))
    near = base[:, 3] + 1e-8 * rng.standard_normal(n)
    X = np.column_stack([base, near])
    return X, [f"x{i}" for i in range(5)], "near-collinear (cond~1e8)"


def dgp_rank_deficient() -> tuple[np.ndarray, list[str], str]:
    """n=200, p=5; col4 = exact linear combo of cols 0..2 -> exact zero eigenvalue."""
    rng = _rng()
    n = 200
    base = rng.standard_normal((n, 4))
    dep = 2.0 * base[:, 0] - 1.5 * base[:, 1] + 0.5 * base[:, 2]
    X = np.column_stack([base, dep])
    return X, [f"x{i}" for i in range(5)], "rank-deficient (exact dependency)"


def dgp_wide_p_gt_n() -> tuple[np.ndarray, list[str], str]:
    """n=20, p=50: more variables than observations (rank <= n-1 after centering)."""
    rng = _rng()
    X = rng.standard_normal((20, 50))
    return X, [f"x{i}" for i in range(50)], "wide p>>n (n=20, p=50)"


def dgp_constant_col() -> tuple[np.ndarray, list[str], str]:
    """n=100, p=4 with col1 constant: scale=TRUE must refuse on both engines."""
    rng = _rng()
    X = rng.standard_normal((100, 4))
    X[:, 1] = 3.0  # zero-variance column
    return X, [f"x{i}" for i in range(4)], "constant column"


# ── outcome reductions ────────────────────────────────────────────────────────

def _sign_align(Vp, Vr):
    Vp = np.asarray(Vp, float)
    Vr = np.asarray(Vr, float)
    signs = np.sign((Vp * Vr).sum(axis=0))
    signs[signs == 0] = 1.0
    return Vr * signs[np.newaxis, :], signs


def _agreement_case(case: str, sut: dict[str, Any], ref: dict[str, Any],
                    scale: bool) -> dict[str, Any]:
    sdev_s = np.asarray(sut["sdev"], float)
    sdev_r = np.asarray(ref["sdev"], float)
    k = min(sdev_s.size, sdev_r.size)
    sdev_s, sdev_r = sdev_s[:k], sdev_r[:k]
    # max_abs over all components (trailing ~0); max_rel over the resolvable ones.
    big = sdev_r > 1e-6 * sdev_r[0]
    sdev_max_abs = float(np.max(np.abs(sdev_s - sdev_r)))
    sdev_max_rel_big = float(np.max(np.abs(sdev_s[big] - sdev_r[big])
                                    / sdev_r[big])) if big.any() else 0.0
    rot_s = np.asarray(sut["rotation"], float)[:, :k]
    rot_r = np.asarray(ref["rotation"], float)[:, :k]
    rot_r_aligned, _ = _sign_align(rot_s, rot_r)
    rot_col_maxabs = np.max(np.abs(rot_s - rot_r_aligned), axis=0)
    # Eigenvectors of a (near-)zero singular value span an arbitrary basis of the
    # null space -- numpy and R legitimately pick different vectors there. Compare
    # rotation ONLY over RESOLVABLE components (sdev > 1e-6*sdev[0]); report the
    # unresolvable (null-space) count + its disagreement separately as EXPECTED
    # ambiguity, not a defect (R10: match R's behaviour, not an undefined choice).
    rot_max_abs_resolvable = float(rot_col_maxabs[big].max()) if big.any() else 0.0
    n_nullspace = int((~big).sum())
    nullspace_disagreement = float(rot_col_maxabs[~big].max()) if (~big).any() else 0.0
    return {
        "case": case,
        "outcome": "agreement",
        "scaling": "correlation" if scale else "covariance",
        "n": sut.get("n"), "p": sut.get("p"), "k_compared": k,
        "k_pystat": int(sdev_s.size), "k_r": int(np.asarray(ref["sdev"]).size),
        "sdev_max_abs": sdev_max_abs,
        "sdev_max_rel_resolvable": sdev_max_rel_big,
        "rotation_max_abs_resolvable": rot_max_abs_resolvable,
        "n_nullspace_components": n_nullspace,
        "nullspace_eigvec_disagreement_expected": nullspace_disagreement,
        "smallest_sdev_pystat": float(sdev_s.min()),
        "smallest_sdev_r": float(sdev_r.min()),
    }


def _refusal_case(case: str, py_err: str | None, r_err: str | None,
                  scale: bool) -> dict[str, Any]:
    return {
        "case": case,
        "outcome": "matched_refusal",
        "scaling": "correlation" if scale else "covariance",
        "pystat_refused": py_err is not None,
        "r_refused": r_err is not None,
        "matched": (py_err is not None) and (r_err is not None),
        "pystat_error": (py_err or "")[:300],
        "r_error": (r_err or "")[:300],
    }


def _run_r_safe(X, names, *, center, scale, reps):
    """Run R prcomp, returning (record, None) on success or (None, errmsg) on failure."""
    try:
        rec, _raw = run_r_pca_record(X, names, dataset="hardcase",
                                     center=center, scale=scale, reps=reps)
        return rec, None
    except RuntimeError as exc:
        return None, str(exc)


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    # Agreement cases (default scale=False = the R15 default invocation).
    agreement_dgps: list[tuple[Callable, bool]] = [
        (dgp_near_collinear, False),
        (dgp_rank_deficient, False),
        (dgp_wide_p_gt_n, False),
    ]
    for dgp, scale in agreement_dgps:
        X, names, label = dgp()
        sut = run_pca_record(X, backend="cpu", dataset="hardcase", scale=scale,
                             repeats=repeats, warmup=1)
        if sut.get("error"):
            raise RuntimeError(f"pystatistics PCA unexpectedly failed on "
                               f"'{label}': {sut['error']}")
        ref, _raw = run_r_pca_record(X, names, dataset="hardcase", center=True,
                                     scale=scale, reps=repeats)
        records.extend([sut, ref])
        row = _agreement_case(label, sut, ref, scale)
        rows.append(row)
        print(f"  [agree]  {label:32s}  sdev_abs={row['sdev_max_abs']:.2e}  "
              f"rot_abs(resolv)={row['rotation_max_abs_resolvable']:.2e}  "
              f"k(py/R)={row['k_pystat']}/{row['k_r']}  "
              f"nullspace={row['n_nullspace_components']}  "
              f"min_sdev={row['smallest_sdev_pystat']:.2e}")

    # Matched-refusal case: constant column under scale=TRUE.
    X, names, label = dgp_constant_col()
    py = run_pca_record(X, backend="cpu", dataset="hardcase", scale=True,
                        repeats=2, warmup=0)
    py_err = py.get("error")
    _r_rec, r_err = _run_r_safe(X, names, center=True, scale=True, reps=2)
    row = _refusal_case(label + " + scale=TRUE", py_err, r_err, True)
    rows.append(row)
    print(f"  [refuse] {label + ' + scale=TRUE':32s}  "
          f"py_refused={row['pystat_refused']}  r_refused={row['r_refused']}  "
          f"matched={row['matched']}")

    # Same constant column under scale=FALSE must SUCCEED on both (zero-variance
    # column simply contributes nothing) -- confirms the refusal is scale-specific.
    sut = run_pca_record(X, backend="cpu", dataset="hardcase", scale=False,
                         repeats=repeats, warmup=1)
    if sut.get("error"):
        raise RuntimeError(f"pystatistics PCA should accept a constant column "
                           f"under scale=False but failed: {sut['error']}")
    ref, _raw = run_r_pca_record(X, names, dataset="hardcase", center=True,
                                 scale=False, reps=repeats)
    records.extend([sut, ref])
    row = _agreement_case("constant column + scale=FALSE", sut, ref, False)
    rows.append(row)
    print(f"  [agree]  {'constant column + scale=FALSE':32s}  "
          f"sdev_abs={row['sdev_max_abs']:.2e}  "
          f"rot_abs(resolv)={row['rotation_max_abs_resolvable']:.2e}")

    config = {
        "study": "hardcases_r10_r15",
        "seed": _SEED,
        "reference": "R stats::prcomp",
        "cases": [r["case"] for r in rows],
        "note": "R15 default invocation = pca(X) center=True scale=False, "
                "covered by the scale=False agreement cases.",
    }
    records = _strip_arrays(records)  # keep scalar agreement metrics, drop arrays
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "multivariate" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"hardcases_cpu_{host}.json", run)
    _write_summary_csv(out_dir / f"hardcases_cpu_{host}_summary.csv", rows)
    print(f"\nwrote {run_path}")
    return run_path


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    generate(args.host, repeats=args.repeats)


if __name__ == "__main__":
    main()
