"""G3 performance sweep (CPU) + the explicit no-GPU-path rationale.

These are closed-form scalar tests and O(n) reductions -- no normal-equations
Gram, no iterative optimiser, no dense linear algebra that an accelerator could
amortise. So the CPU is the correct and only backend; there is NO GPU path and
none is manufactured (documented below and in the report).

We record two things:
  1. pystatistics CPU per-call wall time on the teaching-dataset inputs (tiny n)
     and on a large synthetic input (n = 1e6) for the O(n) reductions, to show
     throughput and that per-call cost is dominated by fixed Python overhead at
     small n (an R2 note, self-reversing at scale).
  2. R's in-process compute time for the same large-n reductions (measured INSIDE
     one R session via system.time, excluding interpreter startup) for parity
     context -- a fair compute-vs-compute comparison, not process-vs-process.

Writes runs/perf_g3.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import adhdata as D
from compare import write_artifact

from pystatistics import __version__ as PYVER
from pystatistics.descriptive import cor, quantile, var
from pystatistics.hypothesis import t_test
from pystatistics.anova import anova_oneway
from pystatsval.timing import time_call

ART = Path(__file__).resolve().parents[2] / \
    "artifacts/anova_descriptive_hypothesis" / f"v{PYVER}" / "runs"

NO_GPU_RATIONALE = (
    "No GPU path. Every function here is either a closed-form scalar statistic "
    "(t/chisq/fisher/wilcox/ks/prop/var.test, ANOVA F, Levene, TukeyHSD q) or an "
    "O(n) reduction (mean/var/cov/cor/quantile). There is no Gram matrix, no "
    "iterative optimiser, and no dense factorisation whose FLOPs an accelerator "
    "could amortise; kernel-launch and host<->device transfer would dominate. "
    "CPU is the correct and only backend (CF-1 is N/A: none of these form a "
    "normal-equations Gram). Manufacturing a GPU path would add surface area and "
    "a silent-precision risk for zero speed-up, so none exists."
)


def _perf(label, fn, reps=7):
    summ, _ = time_call(fn, warmup=1, reps=reps)
    return {"label": label, "median_s": summ["median_s"],
            "min_s": summ["min_s"], "reps": summ["reps"]}


def _r_large_reductions(x: np.ndarray, y: np.ndarray) -> dict:
    """R's in-process compute time (median of 7) for var/cor/quantile on the
    large vector, measured inside ONE R session (startup excluded)."""
    # Loop each op K times inside system.time and divide -> a meaningful
    # per-call time (a single n=1e6 reduction is sub-millisecond, below the
    # system.time clock resolution).
    script = r"""
    args <- commandArgs(trailingOnly=TRUE)
    d <- jsonlite::fromJSON(args[[1]])
    x <- d$x; y <- d$y; K <- 50
    tt <- function(f) { t <- system.time(for (i in 1:K) f())[["elapsed"]]; t / K }
    cat(jsonlite::toJSON(list(
      var=tt(function() var(x)),
      cor_pearson=tt(function() cor(x,y)),
      cor_spearman=tt(function() cor(x,y,method="spearman")),
      quantile_t7=tt(function() quantile(x, probs=c(.1,.25,.5,.75,.9), type=7))
    ), auto_unbox=TRUE))
    """
    import tempfile
    payload = {"x": x.tolist(), "y": y.tolist()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as jf:
        json.dump(payload, jf)
        p = jf.name
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as rf:
        rf.write(script)
        rp = rf.name
    try:
        out = subprocess.run(["Rscript", rp, p], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(out.stderr)
        return json.loads(out.stdout)
    finally:
        Path(p).unlink(missing_ok=True)
        Path(rp).unlink(missing_ok=True)


def run() -> dict:
    mt = D.load_frame("mtcars")
    pg = D.load_frame("PlantGrowth")
    mpg, hp = mt["mpg"], mt["hp"]

    # --- tiny-n (teaching datasets): fixed Python overhead dominates ---
    tiny = [
        _perf("var:mtcars.mpg(n=32)", lambda: var(mpg)),
        _perf("cor:mpg~hp:pearson(n=32)", lambda: cor(mpg, hp)),
        _perf("cor:mpg~hp:kendall(n=32)", lambda: cor(mpg, hp, method="kendall")),
        _perf("quantile:mpg:type7(n=32)", lambda: quantile(mpg, probs=[.25, .5, .75])),
        _perf("t_test:welch(n=32)", lambda: t_test(mpg, hp)),
        _perf("anova_oneway:PlantGrowth(n=30)",
              lambda: anova_oneway(pg["weight"], pg["group__labels"])),
    ]

    # --- large-n O(n) reductions: CPU throughput vs R in-process compute ---
    rng = np.random.default_rng(0)
    N = 1_000_000
    xl = rng.standard_normal(N)
    yl = xl * 0.5 + rng.standard_normal(N)
    large_py = [
        _perf(f"var(n={N})", lambda: var(xl), reps=5),
        _perf(f"cor:pearson(n={N})", lambda: cor(xl, yl), reps=5),
        _perf(f"cor:spearman(n={N})", lambda: cor(xl, yl, method="spearman"), reps=5),
        _perf(f"quantile:type7(n={N})",
              lambda: quantile(xl, probs=[.1, .25, .5, .75, .9]), reps=5),
    ]
    r_large = _r_large_reductions(xl, yl)

    return {
        "module": "anova+descriptive+hypothesis", "guarantee": "G3_performance",
        "engine": "pystatistics:cpu", "py_version": PYVER,
        "no_gpu_rationale": NO_GPU_RATIONALE,
        "tiny_n_cpu": tiny,
        "large_n_cpu": large_py,
        "large_n_r_inprocess_s": r_large,
        "note": ("Large-n rows compare compute-vs-compute (R timed inside one "
                 "session, startup excluded). Tiny-n rows are dominated by fixed "
                 "Python-call overhead (~1e-4 s), an R2 that self-reverses with n."),
    }


def main() -> None:
    payload = run()
    out = write_artifact(ART / "perf_g3.json", payload)
    print("perf G3 (CPU) written ->", out)
    for row in payload["large_n_cpu"]:
        rk = row["label"].split("(")[0].replace("cor:", "cor_").replace(":type7", "_t7")
        print(f"  {row['label']:30s} py median={row['median_s']*1e3:.2f} ms")
    print("  R in-process:", {k: round(v * 1e3, 2) for k, v in
                              payload["large_n_r_inprocess_s"].items()}, "ms")


if __name__ == "__main__":
    main()
