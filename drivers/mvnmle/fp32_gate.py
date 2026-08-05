"""fp32 GPU no-silent-wrong gate proof for mvnmle (RIGOR R12/R13).

THE deliverable of the A0 re-validation. The v3.18.0 report §5 tabulated
CPU-vs-fp32-GPU ``|Δloglik|`` up to ~146 at p=20..50 and treated it as "fp32
tolerance" with **no accept/refuse gate proven** — the exact CF-1 / R12 exposure
class that produced silent-wrong showstoppers in ordinal/multinomial and gam.

The mvnmle ``direct`` fp32 GPU path has no fp32-precision gate at all: it runs
L-BFGS-B at ``tol=1e-3`` and returns whatever it reaches (``converged`` is the
optimiser's own flag), plus two precision-agnostic conditioning guards
(``get_initial_parameters`` raises ``NumericalError`` on an ill-conditioned
sample covariance; ``_guard_degeneracy`` raises ``SingularMatrixError`` on a
rank-deficient fit). So the question is sharp:

    Is the returned fp32 (muhat, Sigmahat) ALWAYS correct to the fp32 tier
    (→ "always accept" is safe, like fp32 PCA), or is there a high-p /
    ill-conditioned regime where a BIASED optimum is returned silently (→ R16)?

DECISIVE EXPERIMENT. The §5 ``|Δloglik|`` conflates two effects: (a) the fp32
optimiser stopping at a different point than fp64, and (b) fp32 *evaluation*
noise from summing the loglik over ~75k observations. We disentangle them by
re-scoring EVERY fit's (mu, Sigma) under one shared fp64 numpy evaluator:

    L_star      = loglik_fp64(mu*,   Sigma*)     # fp64 optimum (the maximum)
    L_hat_fp64  = loglik_fp64(muhat, Sigmahat)   # fp32 estimate, re-scored fp64
    deficit     = L_star - L_hat_fp64            # >= 0 ; the TRUE quality gap
    report_gap  = L_star - loglik_hat            # the §5-style conflated number
    eval_noise  = loglik_hat - L_hat_fp64        # pure fp32 summation error

If ``deficit`` is fp32-tier tiny while ``report_gap`` is large, the §5 number was
benign evaluation noise on a CORRECT optimum → no silent-wrong. If ``deficit`` is
a large fraction of ``report_gap`` and the path returned ``converged=True`` with
no warning → silent-wrong band → R16.

Two problem families:
  * ``synthetic`` — MVN draws with a DIALED condition number kappa, the clean
    R10 instrument for locating any fp32 breakdown boundary and checking that the
    conditioning guards fire BEFORE accuracy degrades (R12).
  * ``survey``   — real GSS/WVS problems at p=15..50, both the screened
    (``|corr|<=0.99``, reproduces §5) and the deliberately unscreened
    (``max_abs_corr=None``, real ill-conditioned) regimes.

Device is chosen by the installed torch (MPS on Mac, CUDA on Forge). CUDA is the
authoritative check (RIGOR: never bless a GPU path on MPS evidence alone).

Usage:
    DATASETS_ROOT=/path/to/h5 python fp32_gate.py [tag] [--family both|synthetic|survey]
"""

from __future__ import annotations

import argparse
import json
import sys
import platform
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path[:0] = [str(_HERE.parent / "_shared")]
from survey_io import build_mvn_problem  # noqa: E402
from curate import standardize_columns  # noqa: E402

import pystatistics as _ps  # noqa: E402
_OUTDIR = _REPO / "artifacts" / "mvnmle" / f"v{_ps.__version__}" / "runs"
from store_io import SURVEY_NAMESPACE, store_root  # noqa: E402

# Guard artifact writes: refuse to clobber evidence committed to git unless
# PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1. See drivers/_shared/artifact_guard.py.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import guard_artifact_path  # noqa: E402


_DATA = store_root() / SURVEY_NAMESPACE


# --------------------------------------------------------------------------- #
# fp64 re-scoring: the library's own exact observed-data loglik evaluator.
# --------------------------------------------------------------------------- #
def make_fp64_evaluator(X: np.ndarray):
    """Return ``f(mu, sigma) -> float``: the exact observed-data loglik in fp64.

    Uses the library's numpy batched evaluator (the same one the closed-form
    monotone solver scores with), so the re-scored number is on the library's
    own fp64 scale, not a re-derivation.
    """
    from pystatistics.mvnmle._objectives.base import MLEObjectiveBase
    from pystatistics.mvnmle.backends._em_batched import (
        build_pattern_index, compute_loglik_batched_np,
    )
    obj = MLEObjectiveBase(np.asarray(X, dtype=np.float64), skip_validation=True)
    index = build_pattern_index(obj.patterns, X.shape[1])

    def evaluate(mu, sigma) -> float:
        return float(compute_loglik_batched_np(
            np.asarray(mu, dtype=np.float64),
            np.asarray(sigma, dtype=np.float64),
            obj.patterns, index))

    return evaluate


# --------------------------------------------------------------------------- #
# Synthetic MVN with a dialed condition number (R10 hard-case instrument).
# --------------------------------------------------------------------------- #
def make_conditioned_mvn(n: int, p: int, kappa: float, *, seed: int,
                         missing: float = 0.15) -> np.ndarray:
    """Draw ``n`` rows from ``N(0, Sigma)`` with ``cond(Sigma)=kappa``, MCAR-masked.

    Sigma is built from a random orthogonal basis and a geometric eigenvalue
    spectrum spanning ``[1/sqrt(kappa), sqrt(kappa)]`` so its 2-norm condition
    number is exactly ``kappa``. Deterministic given ``seed`` (Rule 6). Cells are
    dropped completely at random at rate ``missing``; no row is left fully
    missing.
    """
    if p < 2:
        raise ValueError(f"need p >= 2, got {p}")
    if kappa < 1.0:
        raise ValueError(f"kappa must be >= 1, got {kappa}")
    rng = np.random.default_rng(seed)
    # Geometric eigenvalue spectrum with condition number exactly kappa.
    eig = np.geomspace(1.0 / np.sqrt(kappa), np.sqrt(kappa), p)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Sigma = (Q * eig) @ Q.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, p)) @ L.T
    # MCAR mask, guarding against a fully-missing row.
    mask = rng.random((n, p)) < missing
    full = mask.all(axis=1)
    mask[full, 0] = False
    X = X.astype(np.float64)
    X[mask] = np.nan
    return X


# --------------------------------------------------------------------------- #
# One gate measurement on a single problem.
# --------------------------------------------------------------------------- #
def _default_gate_from_fit(sigmahat) -> str:
    """The DEFAULT (force=False) gate decision implied by a forced fit's
    covariance: 'accepted' iff the shared degeneracy guard would NOT raise.

    Derived from the fit (no extra optimisation) — the ``force`` flag only
    controls whether ``_guard_degeneracy`` raises on that same fitted
    covariance, so the default decision is a pure function of it.
    """
    from pystatistics.mvnmle._degeneracy import (
        DEFAULT_COLLINEARITY_TOL, check_fitted_covariance,
    )
    try:
        check_fitted_covariance(np.asarray(sigmahat), tol=DEFAULT_COLLINEARITY_TOL,
                                force=False)
        return "accepted"
    except Exception as exc:  # noqa: BLE001
        return f"refused:{type(exc).__name__}"


def gate_measure(X: np.ndarray, *, label: str, **meta) -> dict:
    """Fit fp64 (forward-Cholesky) + fp32 GPU, re-score both in fp64, classify.

    Design (RIGOR R12 "force the accepted fit and verify against fp64"):
      * Estimates are taken with ``force=True`` so BOTH paths always return
        numbers — even in the degenerate regime the default gate refuses — so the
        fp32 bias can be measured everywhere, including past the refusal boundary.
      * The DEFAULT gate decision (``force=False``) is recorded separately for
        each path (``ref_gate`` / ``gpu_gate``). A true classifier: wherever the
        gate ``accepts``, the forced fit is already correct to the fp32 tier;
        wherever the forced fit is degraded, the gate ``refuses`` first.

    Ground truth is ``backend='cpu'`` — the fp64 forward-Cholesky estimator, the
    SAME parameterization as the GPU path, so the only difference measured is
    precision (fp32 vs fp64), isolating the hardware/precision effect (R11).
    """
    from pystatistics.mvnmle import mlest

    n, p = X.shape
    rec = {"label": label, "n": int(n), "p": int(p),
           "missing_frac": float(np.isnan(X).mean()), **meta}

    ev = make_fp64_evaluator(X)

    # fp64 ground truth: the fast torch forward-Cholesky estimator (same
    # parameterization as the GPU path, so only precision differs — R11), run
    # tightly-converged (tol=1e-7, max_iter=500) so L_star is a genuine maximum
    # rather than a stalled point. The primary estimate-error metric is
    # ``sigma_rel_fro`` (covariance agreement, independent of which fit is "the
    # max"); ``deficit`` is reported honestly alongside (a small negative value
    # just means both fits sit on the flat optimum plateau). A pre-optimisation
    # NumericalError is not bypassed by force=True, so it can raise here too — a
    # fail-loud refusal.
    try:
        r = mlest(X, backend="cpu", method="direct", force=True,
                  tol=1e-7, max_iter=500)
    except Exception as exc:  # noqa: BLE001
        rec.update(ref_backend=None, ref_gate=f"refused:{type(exc).__name__}",
                   ref_forced_error=f"{type(exc).__name__}: {exc}", gpu_gate=None)
        return rec
    mu_star, Sig_star = np.asarray(r.muhat), np.asarray(r.sigmahat)
    ref_bk, ref_conv = r.backend_name, bool(r.converged)
    L_star = ev(mu_star, Sig_star)
    scale = abs(L_star) if L_star != 0 else 1.0
    sig_fro = float(np.linalg.norm(Sig_star))
    rec.update(ref_backend=ref_bk, ref_converged=ref_conv,
               ref_gate=_default_gate_from_fit(Sig_star), L_star=L_star)

    # fp32 GPU path, forced (device = MPS on Mac, CUDA on Forge).
    try:
        hat = mlest(X, backend="gpu", method="direct", force=True)
    except Exception as exc:  # noqa: BLE001 — an fp32-path exception is a valid, non-silent outcome
        rec.update(gpu_backend=None, gpu_gate=f"refused:{type(exc).__name__}",
                   gpu_forced_error=f"{type(exc).__name__}: {exc}")
        return rec

    mu_hat, Sig_hat = np.asarray(hat.muhat), np.asarray(hat.sigmahat)
    L_hat_fp64 = ev(mu_hat, Sig_hat)
    deficit = L_star - L_hat_fp64
    report_gap = L_star - float(hat.loglik)

    rec.update(
        gpu_backend=hat.backend_name,
        gpu_gate=_default_gate_from_fit(Sig_hat),
        gpu_converged=bool(hat.converged),
        gpu_warnings="; ".join(hat.warnings) if hat.warnings else "",
        gpu_loglik=float(hat.loglik),
        L_hat_fp64=L_hat_fp64,
        deficit=deficit,
        deficit_rel=deficit / scale,
        deficit_per_obs=deficit / n,
        report_gap=report_gap,
        eval_noise=report_gap - deficit,
        sigma_rel_fro=float(np.linalg.norm(Sig_hat - Sig_star) / sig_fro),
        sigma_max_abs=float(np.max(np.abs(Sig_hat - Sig_star))),
        mu_max_abs=float(np.max(np.abs(mu_hat - mu_star))),
    )
    return rec


# --------------------------------------------------------------------------- #
# Problem families.
# --------------------------------------------------------------------------- #
def _log(rec) -> None:
    print(f"  {rec['label']:26s} ref_gate={str(rec.get('ref_gate')):>22s} "
          f"gpu_gate={str(rec.get('gpu_gate')):>26s} "
          f"deficit={rec.get('deficit')!r} rel_fro={rec.get('sigma_rel_fro')!r}"
          + (f" FORCED_ERR={rec.get('gpu_forced_error')}"
             if rec.get('gpu_forced_error') else ""), flush=True)


def synthetic_records() -> list[dict]:
    """Conditioning sweep: fixed n,p; kappa dialed across the fp32 danger zone,
    finely around the refusal boundary where a silent-wrong band would hide."""
    n, p = 8_000, 20
    out = []
    for kappa in (1e1, 1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7, 1e9):
        X = make_conditioned_mvn(n, p, kappa, seed=20260707)
        out.append(gate_measure(
            X, label=f"synthetic_k{kappa:.0e}", family="synthetic",
            kappa=float(kappa), seed=20260707))
        _log(out[-1])
    return out


def survey_records() -> list[dict]:
    """Real GSS/WVS problems in the §5 benchmark regime: screened (|corr|<=0.99)
    AND standardized (unit-variance columns — the paper pipeline). This is the
    exact regime whose CPU-vs-fp32-GPU ``|Δloglik|`` of 113–146 the A0 audit
    flagged; ``report_gap`` reproduces that number, ``deficit`` decomposes it into
    the real optimisation bias vs benign fp32 evaluation noise."""
    out = []
    grid = [("gss", p) for p in (15, 20, 25, 50)] + \
           [("wvs", p) for p in (15, 20, 25, 50)]
    for survey, p in grid:
        path = _DATA / f"{survey}.h5"
        if not path.exists():
            print(f"{survey}: missing {path}", file=sys.stderr)
            continue
        try:
            prob = build_mvn_problem(path, p, max_abs_corr=0.99)
            X = np.asarray(standardize_columns(prob.X), dtype=np.float64)
            out.append(gate_measure(
                X, label=f"{survey}_p{p}_std", family="survey",
                survey=survey, screening="corr<=0.99", standardized=True))
            _log(out[-1])
        except Exception as exc:  # noqa: BLE001
            print(f"{survey} p={p} build failed: {exc}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", nargs="?", default="fp32_gate")
    ap.add_argument("--family", choices=("both", "synthetic", "survey"),
                    default="both")
    args = ap.parse_args()

    records: list[dict] = []
    if args.family in ("both", "synthetic"):
        records += synthetic_records()
    if args.family in ("both", "survey"):
        records += survey_records()

    # Host / device provenance for the manifest.
    try:
        import torch
        if torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.get_device_name(0)}"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        torch_ver = torch.__version__
    except Exception:  # noqa: BLE001
        dev, torch_ver = "unknown", None

    import pystatistics
    payload = {
        "study": "fp32_gate",
        "kind": "r12-no-silent-wrong-gate",
        "pystatistics_version": pystatistics.__version__,
        "device": dev,
        "torch_version": torch_ver,
        "host": platform.node(),
        "records": records,
    }
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    out = _OUTDIR / f"{args.tag}.json"
    guard_artifact_path(out)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({len(records)} records, device={dev})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
