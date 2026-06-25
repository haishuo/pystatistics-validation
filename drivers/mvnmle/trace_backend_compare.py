"""Cross-backend comparison of per-pattern trace-term methods (torch-only).

Times the three ways to compute tr(Sigma_k^{-1} M_k) over a batch of patterns:
  - solve2     : two batched triangular solves (the portable default)
  - chol_solve : torch.cholesky_solve (fast on CUDA; NOT implemented on MPS)
  - blocked    : matmul-only blocked inversion of the Cholesky factor (our method)

The scientific question is the WITHIN-backend ratio (blocked vs the solve family)
on CUDA versus on MPS — that isolates the algorithm from the hardware. Runs the
identical code on any device; depends only on torch.

Usage:  python trace_backend_compare.py [cuda|mps|cpu]
"""

import sys
import time

import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else (
    "cuda" if torch.cuda.is_available() else
    ("mps" if torch.backends.mps.is_available() else "cpu"))


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def tri_inv_blocked(L):
    """Matmul-only inverse of a batched lower-triangular matrix."""
    v = L.shape[-1]
    if v == 1:
        return 1.0 / L
    if v == 2:
        ia = 1.0 / L[..., 0, 0]
        ic = 1.0 / L[..., 1, 1]
        off = -L[..., 1, 0] * ia * ic
        z = torch.zeros_like(ia)
        return torch.stack([torch.stack([ia, z], -1),
                            torch.stack([off, ic], -1)], -2)
    h = v // 2
    A = L[..., :h, :h].contiguous()
    C = L[..., h:, h:].contiguous()
    B = L[..., h:, :h].contiguous()
    Ai = tri_inv_blocked(A)
    Ci = tri_inv_blocked(C)
    BL = -Ci @ (B @ Ai)
    z = torch.zeros(L.shape[:-2] + (h, v - h), device=L.device, dtype=L.dtype)
    return torch.cat([torch.cat([Ai, z], -1), torch.cat([BL, Ci], -1)], -2)


def m_solve2(L, M):
    Y = torch.linalg.solve_triangular(L, M, upper=False)
    X = torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True)
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)


def m_cholsolve(L, M):
    X = torch.cholesky_solve(M, L)
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)


def m_blocked(L, M):
    W = tri_inv_blocked(L)
    return ((W.transpose(-1, -2) @ W) * M).sum((-2, -1))


def bench(P, v, reps=5):
    g = torch.Generator(device="cpu").manual_seed(0)
    A = torch.randn(P, v, v, generator=g)
    S = (A @ A.transpose(-1, -2) + v * torch.eye(v)).to(DEV)
    Bm = torch.randn(P, v, v, generator=g)
    M = (Bm @ Bm.transpose(-1, -2)).to(DEV)
    L = torch.linalg.cholesky(S)
    sync()
    ref = m_solve2(L, M)
    sync()
    out = {}
    for nm, fn in [("solve2", m_solve2), ("chol_solve", m_cholsolve),
                   ("blocked", m_blocked)]:
        try:
            fn(L, M)
            sync()
            t = time.perf_counter()
            for _ in range(reps):
                r = fn(L, M)
            sync()
            ms = (time.perf_counter() - t) / reps * 1000.0
            rel = ((r - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
            out[nm] = (ms, rel)
        except Exception as e:  # noqa: BLE001
            out[nm] = (None, f"{type(e).__name__}: {str(e)[:50]}")
    return out


def main():
    import csv
    from pathlib import Path
    print(f"device={DEV} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if DEV=='cuda' else DEV}")
    csv_rows = []
    for P, v in [(20522, 25), (20522, 50), (20522, 100), (5000, 100)]:
        o = bench(P, v)
        parts = []
        for nm, (ms, info) in o.items():
            parts.append(f"{nm}={ms:.1f}ms(rel{info:.0e})" if isinstance(ms, float)
                         else f"{nm}=FAIL[{info}]")
        # ratio of solve-family best vs blocked
        solve_ms = [o[k][0] for k in ("solve2", "chol_solve") if isinstance(o[k][0], float)]
        blk = o["blocked"][0]
        ratio = (f"{min(solve_ms)/blk:.1f}x" if solve_ms and isinstance(blk, float) else "n/a")
        print(f"P={P} v={v:3d}: " + "  ".join(parts) + f"  | best_solve/blocked={ratio}")
        # the figure uses the ~20k-pattern cases
        if P == 20522:
            csv_rows.append({"v": v, "backend": DEV,
                             "solve_ms": (min(solve_ms) if solve_ms else ""),
                             "blocked_ms": (blk if isinstance(blk, float) else "")})
    out = Path(__file__).resolve().parents[2] / "results" / f"trace_backend_{DEV}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["v", "backend", "solve_ms", "blocked_ms"])
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print("wrote", out)


if __name__ == "__main__":
    main()
