"""CF-1 red-team — fp32 Hessian/Gram silent-wrong probe for polr + multinom GPU.

Both GPU paths invert the model Hessian in the working dtype (fp32 on
``backend='gpu'``): multinom forms ``H = X'WX`` block-by-block and inverts it in
fp32; polr takes an autograd Hessian and inverts it in fp32. That is the exact
CF-1 exposure (a normal-equations Gram inverted in single precision) that
materialised silent-wrong in gam. This script proves — on CUDA first, then MPS —
whether an ill-conditioned design yields a silently-wrong variance/SE while the
fit reports convergence.

Two levels of evidence per (surface, conditioning) case:

  BLACK-BOX (what a user sees): fit ``backend='gpu'`` (fp32) and ``backend='cpu'``
    (fp64); compare coefficients and standard errors; flag any NEGATIVE variance
    (an invalid vcov a user would trust). On CUDA, ``backend='gpu_fp64'`` isolates
    hardware from precision (R11).

  WHITE-BOX (isolates the fp32 Gram inversion, R12): at the SAME CPU-fp64 optimum
    parameters, invert the Hessian in fp32 vs fp64 via the library's own GPU
    likelihood classes, and compare the variance diagonals directly — no optimizer
    confound. A large fp32-vs-fp64 gap here is the CF-1 signature.

Conditioning sweep: X built by SVD with a target condition number spanning the
fp32-safe boundary (~1e7). ``X'WX`` inherits ~cond^2, so cond>=1e4 already
stresses an fp32 Gram inverse.

Self-contained: emits one JSON object to stdout (the caller redirects it to
artifacts/ordinal_multinomial/v<ver>/runs/cf1_<device>.json). Run on a CUDA host
(Forge) FIRST, then MPS.

Run:  PYTHONPATH=~/scratch_om/lib python run_cf1_gpu.py [cuda|mps]
"""

from __future__ import annotations

import json
import sys

import numpy as np


# --------------------------------------------------------------------------
# design generation (self-contained; no driver imports so it runs on Forge)
# --------------------------------------------------------------------------
def _conditioned_X(n: int, p: int, cond: float, seed: int) -> np.ndarray:
    """Mean-zero design of shape (n, p) with the target condition number ``cond``
    (singular values geometrically spaced in [1, cond]); columns then centered."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, p))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.geomspace(1.0, cond, p)
    X = (U * s) @ Vt
    X = X - X.mean(0)
    X = X / X.std(0)              # unit-scale columns; conditioning lives in the
    # cross-column correlation structure that SVD imposed (near-collinearity).
    # Re-impose the condition number after standardizing:
    U2, _, Vt2 = np.linalg.svd(X, full_matrices=False)
    return (U2 * np.geomspace(1.0, cond, p)) @ Vt2


def _make_ordinal(n: int, p: int, cond: float, seed: int = 0):
    """Ill-conditioned design, but a well-SPREAD 4-level response: the linear
    predictor is standardized to unit scale and the thresholds are placed at its
    quartiles, so all categories are populated regardless of cond (the
    ill-conditioning lives in X, not in a collapsed response)."""
    X = _conditioned_X(n, p, cond, seed)
    beta = np.linspace(0.8, -0.8, p)
    eta = X @ beta
    eta = eta / (eta.std() + 1e-12)            # moderate signal, keep X ill-cond
    u = np.random.default_rng(seed + 1).logistic(size=n)
    latent = eta + u
    zeta = np.quantile(latent, [0.25, 0.5, 0.75])
    y = np.searchsorted(zeta, latent)
    return y.astype(int), X, 4


def _make_multinom(n: int, p: int, cond: float, seed: int = 0):
    """Ill-conditioned design, balanced 3-class response (softmax logits
    standardized to moderate scale so no class is empty at high cond)."""
    Xf = _conditioned_X(n, p, cond, seed)
    X = np.column_stack([np.ones(n), Xf])
    K = 3
    B = np.linspace(0.8, -0.8, (K - 1) * p).reshape(K - 1, p)
    cols = [Xf @ B[j] for j in range(K - 1)]
    cols = [c / (c.std() + 1e-12) for c in cols]      # moderate logits
    eta = np.column_stack(cols + [np.zeros(n)])
    P = np.exp(eta - eta.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    u = np.random.default_rng(seed + 1).uniform(size=n)
    y = (u[:, None] > np.cumsum(P, 1)).sum(1).astype(int)
    return y, X, K


def _relerr(a, b) -> float:
    """Max relative error, scaled by the SIZE of b (robust to near-zero entries):
    max|a-b| / max_j|b_j|. Appropriate for coefficient/variance vectors where
    individual entries pass through zero."""
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    if a.shape != b.shape:
        return float("nan")
    scale = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / scale)


# --------------------------------------------------------------------------
# polr cases
# --------------------------------------------------------------------------
def _polr_raw_from_natural(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Invert the natural-threshold -> raw (log-gap) map polr optimizes in."""
    raw = np.empty(len(alpha))
    raw[0] = alpha[0]
    raw[1:] = np.log(np.maximum(np.diff(alpha), 1e-12))
    return np.concatenate([raw, np.asarray(beta, float)])


def polr_case(device: str, cond: float, has_fp64: bool) -> dict:
    from pystatistics.ordinal import polr
    from pystatistics.ordinal.backends.gpu_likelihood import PolrGPULikelihood
    from pystatistics.core.exceptions import ConvergenceError

    y, X, K = _make_ordinal(4000, 6, cond, seed=int(round(np.log10(cond))))
    out = {"surface": "polr", "cond": cond, "n": len(y), "p": X.shape[1], "K": K,
           "xtx_cond": float(np.linalg.cond(X.T @ X))}
    try:
        cpu = polr(y, X, link="logistic", backend="cpu")
    except ConvergenceError as e:
        out["cpu_error"] = str(e)[:120]
        out["note"] = ("CPU fp64 itself refuses (design beyond fp64) — not a CF-1 "
                       "case; both precisions correctly fail here.")
        return out
    out["cpu_converged"] = bool(cpu.converged)

    # BLACK-BOX: gpu fp32 vs cpu fp64
    try:
        gpu = polr(y, X, link="logistic", backend="gpu")
        out["gpu_fp32_converged"] = bool(gpu.converged)
        out["gpu_fp32_coef_relerr"] = _relerr(gpu.coefficients, cpu.coefficients)
        out["gpu_fp32_se_relerr"] = _relerr(gpu.standard_errors,
                                            cpu.standard_errors)
        out["gpu_fp32_neg_var"] = bool(np.any(np.asarray(gpu.standard_errors) <= 0)
                                       or np.any(~np.isfinite(gpu.standard_errors)))
    except ConvergenceError as e:
        out["gpu_fp32_error"] = str(e)[:120]

    if has_fp64:
        try:
            g64 = polr(y, X, link="logistic", backend="gpu_fp64")
            out["gpu_fp64_coef_relerr"] = _relerr(g64.coefficients, cpu.coefficients)
            out["gpu_fp64_se_relerr"] = _relerr(g64.standard_errors,
                                                cpu.standard_errors)
        except ConvergenceError as e:
            out["gpu_fp64_error"] = str(e)[:120]

    # WHITE-BOX: at the SAME cpu params, invert the Hessian in fp32 vs fp64.
    raw = _polr_raw_from_natural(cpu.threshold_values, cpu.coefficients)
    like32 = PolrGPULikelihood(X, y, K, link_name="logistic", device=device,
                               use_fp64=False)
    v32 = np.asarray(like32.compute_vcov(raw))
    ref = None
    if has_fp64:
        like64 = PolrGPULikelihood(X, y, K, link_name="logistic", device=device,
                                   use_fp64=True)
        ref = np.asarray(like64.compute_vcov(raw))
    else:
        # MPS has no fp64: use the CPU analytic observed-information vcov (raw).
        from pystatistics.ordinal._information import observed_information
        from pystatistics.regression.families import LogitLink
        H = observed_information(raw, y, X, LogitLink(), K)
        ref = np.linalg.inv(H)
    out["wb_var_relerr"] = _relerr(np.diag(v32), np.diag(ref))
    out["wb_fp32_neg_var"] = bool(np.any(np.diag(v32) <= 0))
    out["wb_ref_kind"] = "gpu_fp64" if has_fp64 else "cpu_fp64_analytic"
    return out


# --------------------------------------------------------------------------
# multinom cases
# --------------------------------------------------------------------------
def multinom_case(device: str, cond: float, has_fp64: bool) -> dict:
    from pystatistics.multinomial import multinom
    from pystatistics.multinomial.backends.gpu_likelihood import (
        MultinomialGPULikelihood)
    from pystatistics.core.exceptions import ConvergenceError

    y, X, K = _make_multinom(4000, 6, cond, seed=int(round(np.log10(cond))))
    out = {"surface": "multinom", "cond": cond, "n": len(y), "p": X.shape[1],
           "K": K, "xtx_cond": float(np.linalg.cond(X.T @ X))}
    try:
        cpu = multinom(y, X, backend="cpu", max_iter=2000)
    except ConvergenceError as e:
        out["cpu_error"] = str(e)[:120]
        out["note"] = ("CPU fp64 itself refuses (design beyond fp64) — not a CF-1 "
                       "case.")
        return out
    out["cpu_converged"] = bool(cpu.converged)

    try:
        gpu = multinom(y, X, backend="gpu", max_iter=2000)
        out["gpu_fp32_converged"] = bool(gpu.converged)
        out["gpu_fp32_coef_relerr"] = _relerr(gpu.coefficient_matrix,
                                              cpu.coefficient_matrix)
        out["gpu_fp32_se_relerr"] = _relerr(gpu.standard_errors,
                                            cpu.standard_errors)
        out["gpu_fp32_neg_var"] = bool(np.any(np.asarray(gpu.standard_errors) <= 0)
                                       or np.any(~np.isfinite(gpu.standard_errors)))
    except ConvergenceError as e:
        out["gpu_fp32_error"] = str(e)[:120]

    if has_fp64:
        try:
            g64 = multinom(y, X, backend="gpu_fp64", max_iter=2000)
            out["gpu_fp64_coef_relerr"] = _relerr(g64.coefficient_matrix,
                                                  cpu.coefficient_matrix)
            out["gpu_fp64_se_relerr"] = _relerr(g64.standard_errors,
                                                cpu.standard_errors)
        except ConvergenceError as e:
            out["gpu_fp64_error"] = str(e)[:120]

    # WHITE-BOX: X'WX inverted in fp32 vs fp64 at the SAME cpu params.
    params = np.asarray(cpu.coefficient_matrix, float).ravel()
    like32 = MultinomialGPULikelihood(X, y, K, device=device, use_fp64=False)
    v32 = np.asarray(like32.compute_vcov(params))
    if has_fp64:
        like64 = MultinomialGPULikelihood(X, y, K, device=device, use_fp64=True)
        ref = np.asarray(like64.compute_vcov(params))
        out["wb_ref_kind"] = "gpu_fp64"
    else:
        # CPU fp64 analytic X'WX reference (same closed form, numpy fp64).
        ref = _multinom_xwx_inv_fp64(X, y, K, params)
        out["wb_ref_kind"] = "cpu_fp64_analytic"
    out["wb_var_relerr"] = _relerr(np.diag(v32), np.diag(ref))
    out["wb_fp32_neg_var"] = bool(np.any(np.diag(v32) <= 0))
    return out


def _multinom_xwx_inv_fp64(X, y, K, params) -> np.ndarray:
    """Closed-form softmax Hessian X'WX in numpy fp64, inverted — the reference the
    GPU fp32 block-Hessian must match."""
    n, p = X.shape
    nnr = K - 1
    beta = params.reshape(nnr, p)
    eta = np.column_stack([X @ beta[j] for j in range(nnr)] + [np.zeros(n)])
    P = np.exp(eta - eta.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    Pn = P[:, :nnr]
    H = np.zeros((nnr * p, nnr * p))
    for j in range(nnr):
        for k in range(j, nnr):
            w = Pn[:, j] * ((1.0 if j == k else 0.0) - Pn[:, k])
            block = (X * w[:, None]).T @ X
            H[j*p:(j+1)*p, k*p:(k+1)*p] = block
            if k != j:
                H[k*p:(k+1)*p, j*p:(j+1)*p] = block.T
    return np.linalg.inv(H)


def main() -> None:
    import torch
    dev_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if dev_arg:
        device = dev_arg
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        print(json.dumps({"error": "no GPU device available"})); return
    has_fp64 = (device == "cuda")

    conds = [1e1, 1e2, 1e3, 1e4]
    polr_rows = [polr_case(device, c, has_fp64) for c in conds]
    multinom_rows = [multinom_case(device, c, has_fp64) for c in conds]

    result = {
        "device": device, "has_fp64": has_fp64,
        "torch_version": torch.__version__,
        "gpu_name": (torch.cuda.get_device_name(0) if device == "cuda" else "mps"),
        "polr": polr_rows, "multinom": multinom_rows,
    }
    print(json.dumps(result, indent=2, default=lambda o: (
        float(o) if isinstance(o, (np.floating,)) else
        int(o) if isinstance(o, (np.integer,)) else
        bool(o) if isinstance(o, (np.bool_,)) else str(o))))


if __name__ == "__main__":
    main()
