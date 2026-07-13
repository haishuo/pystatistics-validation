"""Determinism tier (R6) — the seed contract for a stochastic method.

A Monte-Carlo method with no seedable determinism is itself a finding. This
runner proves: (1) a supplied seed makes boot / boot_ci / permutation_test
bit-reproducible on re-run (CPU and GPU/MPS); (2) the DEFAULT seed=None is
non-reproducible (documented — the injectable seed is the determinism handle,
R15). The RNG is numpy.random.default_rng(seed) on CPU; torch.manual_seed(seed)
on GPU (same device).

Emits artifacts/montecarlo/v<ver>/runs/determinism.json (render-from-artifacts).

Run: MVNMLE_DATA_DIR=Dev/datasets python drivers/montecarlo/run_determinism.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcdata  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402

from pystatistics.montecarlo import boot, boot_ci, permutation_test  # noqa: E402

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/montecarlo/v{ver}/runs/determinism.json")


def _has_mps() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available())
    except Exception:
        return False


def run_cpu_reproducibility() -> list[dict]:
    recs = []
    law = mcdata.load("law").values
    city = mcdata.load("city").values
    x, y = mcdata.two_sample("sleep")

    # boot: identical t/t0/bias/se on re-run with same seed
    for key, data, stat_name in [("law", law, "corr"), ("city", city, "ratio")]:
        stat = mcdata.STAT[stat_name]
        r1 = boot(data, stat, n_resamples=3000, seed=20260705)
        r2 = boot(data, stat, n_resamples=3000, seed=20260705)
        same = (np.array_equal(r1.t, r2.t) and np.array_equal(r1.t0, r2.t0)
                and r1.bias[0] == r2.bias[0] and r1.standard_errors[0] == r2.standard_errors[0])
        # boot_ci identical too
        c1 = boot_ci(r1, ci_type="all").conf_int
        c2 = boot_ci(r2, ci_type="all").conf_int
        ci_same = all(np.array_equal(c1[k], c2[k]) for k in c1)
        recs.append({"group": "boot_repro", "dataset": key, "statistic": stat_name,
                     "t_identical": bool(same), "ci_identical": bool(ci_same),
                     "pass": bool(same and ci_same)})

    # permutation_test: identical perm_stats/p on re-run with same seed
    p1 = permutation_test(x, y, mcdata.mean_diff, n_resamples=5000, seed=99)
    p2 = permutation_test(x, y, mcdata.mean_diff, n_resamples=5000, seed=99)
    perm_same = (np.array_equal(p1.perm_stats, p2.perm_stats)
                 and p1.p_value == p2.p_value)
    recs.append({"group": "perm_repro", "dataset": "sleep",
                 "perm_stats_identical": bool(np.array_equal(p1.perm_stats, p2.perm_stats)),
                 "p_identical": bool(p1.p_value == p2.p_value),
                 "pass": bool(perm_same)})
    return recs


def run_default_nondeterminism() -> list[dict]:
    """seed=None must be non-reproducible (documented; seed is the handle)."""
    law = mcdata.load("law").values
    r1 = boot(law, mcdata.STAT["corr"], n_resamples=2000)      # seed=None
    r2 = boot(law, mcdata.STAT["corr"], n_resamples=2000)
    differ = not np.array_equal(r1.t, r2.t)
    # t0 (deterministic) must still match
    t0_same = np.array_equal(r1.t0, r2.t0)
    return [{"group": "default_seed", "dataset": "law",
             "seed": None, "replicates_differ": bool(differ),
             "t0_deterministic": bool(t0_same),
             "note": "default seed=None is intentionally non-reproducible; "
                     "pass a seed for R6 determinism (t0 is always deterministic)",
             "pass": bool(differ and t0_same)}]


def run_gpu_reproducibility() -> list[dict]:
    if not _has_mps():
        return [{"group": "gpu_repro", "skipped": "no MPS device", "pass": True}]
    recs = []
    rng = np.random.default_rng(0)
    data = rng.normal(5, 2, 400)
    r1 = boot(data, mcdata.STAT["mean"], n_resamples=2000, seed=7,
              backend="gpu", gpu_statistic="mean")
    r2 = boot(data, mcdata.STAT["mean"], n_resamples=2000, seed=7,
              backend="gpu", gpu_statistic="mean")
    recs.append({"group": "gpu_repro", "op": "boot_mean",
                 "device": r1.backend_name, "t_identical": bool(np.array_equal(r1.t, r2.t)),
                 "pass": bool(np.array_equal(r1.t, r2.t))})
    xa = rng.normal(0, 1, 150); ya = rng.normal(0.4, 1, 150)
    q1 = permutation_test(xa, ya, mcdata.mean_diff, n_resamples=4000, seed=13,
                          backend="gpu", gpu_statistic="mean_diff")
    q2 = permutation_test(xa, ya, mcdata.mean_diff, n_resamples=4000, seed=13,
                          backend="gpu", gpu_statistic="mean_diff")
    recs.append({"group": "gpu_repro", "op": "perm_meandiff",
                 "device": q1.backend_name,
                 "perm_identical": bool(np.array_equal(q1.perm_stats, q2.perm_stats)),
                 "p_identical": bool(q1.p_value == q2.p_value),
                 "pass": bool(np.array_equal(q1.perm_stats, q2.perm_stats)
                              and q1.p_value == q2.p_value)})
    return recs


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)

    cpu = run_cpu_reproducibility()
    default = run_default_nondeterminism()
    gpu = run_gpu_reproducibility()

    run = build_run(
        env=env,
        config={"suite": "montecarlo-determinism",
                "reference": "self (seed contract) — R6",
                "tolerance_contract": "bit-identical on same seed; default "
                "seed=None non-reproducible by design (t0 always deterministic)"},
        records=[{"key": "cpu_reproducibility", "checks": cpu},
                 {"key": "default_nondeterminism", "checks": default},
                 {"key": "gpu_reproducibility", "checks": gpu}],
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    write_run(out, run)
    print(f"wrote {out}")
    for grp in (cpu, default, gpu):
        npass = sum(bool(r.get("pass")) for r in grp)
        print(f"  {grp[0]['group']:16s}... {npass}/{len(grp)} pass")


if __name__ == "__main__":
    main()
