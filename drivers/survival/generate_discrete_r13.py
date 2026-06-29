"""RIGOR R13 — re-prove the fp32 no-silent-wrong gate on discrete-time's REGIME.

``survival.discrete_time`` forwards ``backend=`` to ``regression.fit(family=
'binomial')``. The regression R12 study proved the float32 GPU accept/refuse gate
is a true classifier (never accept a wrong fit, never refuse a correct one) — but
on the regression GLM's well-behaved grid. R13 says that guarantee is NOT
inheritable: discrete-time's person-period regime — heavy low-weight
interval-dummy blocks (low baseline hazard -> tiny IRLS weights), sparse
near-separating intervals — is adversarial in ways the regression validation never
stressed. So the gate must be re-proven HERE, on discrete-time's own designs.

This driver mirrors the regression R12 (``generate_fp32_boundary``) + inference-SE
(``generate_inference_se``) studies, but every design is a person-period binomial
expansion (``_pp_boundary``) plus a REAL ``survival::flchain`` anchor, so the proof
is recognizably discrete-time's, not a generic GLM grid. For each design:

  * float64 reference = the CPU fit (and, on CUDA, ``gpu_fp64`` — exact);
  * the plain float32 GPU fit is attempted:
      - ACCEPTED  -> compare to float64: deviance must match to the fp32 tier
                     (the right optimum), and the covariate SEs must NOT understate
                     instability (gpu_se >= 0.5*cpu_se) — else inference-silent-wrong;
      - REFUSED   -> force it and confirm the forced fit is genuinely wrong / blows
                     up (true positive), not an R9 false negative.

Classifications (boundary):
  safe / refused-true-positive / refused-force-recovers / SILENT-WRONG
Inference verdict (accepted fits):
  inference-warned / INFERENCE-SILENT-WRONG / refused (fail-loud)

A SILENT-WRONG or INFERENCE-SILENT-WRONG verdict is a correctness regression behind
a speed win — the driver prints a loud banner; STOP and surface to the user.

    python -m drivers.survival.generate_discrete_r13 --host powerhouse --device mps
    python -m drivers.survival.generate_discrete_r13 --host forge --device cuda
"""

from __future__ import annotations

import argparse
import csv
import socket
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from drivers.survival import datasets
from drivers.survival._person_period import build_person_period
from drivers.survival._pp_boundary import SWEEPS, eq_cond, make_pp_boundary

_REPO = Path(__file__).resolve().parent.parent.parent

# fp32 tier: a deviance gap above this (relative) means a DIFFERENT (worse) optimum.
_DEV_TOL = 1e-3
# Covariate-SE understatement: gpu_se below this fraction of cpu_se hides instability.
_SE_UNDERSTATE_FRAC = 0.5
# Covariate-SE agreement contract (Tier-2): accepted SEs should track fp64 CPU.
_SE_REL_TOL = 1e-2
# Reference-degeneracy guard (RIGOR R9/G3): a design whose fp64 CPU reference does
# not converge, or carries fewer than this many events, has NO well-defined optimum
# to call the fp32 fit "wrong" against. Such a design is OUTSIDE the validatable
# regime — it is excluded from the no-silent-wrong verdict, never silently passed.
_MIN_EVENTS = 20


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300)))


def _designs() -> list[tuple[str, float, tuple]]:
    """(mechanism, severity, (X_pp, y_pp, n_int, n_cov)) over the PP sweep + anchor."""
    out: list[tuple[str, float, tuple]] = []
    seed = 0
    for mechanism, sweep in SWEEPS.items():
        for sev in sweep:
            seed += 1
            out.append((mechanism, sev, make_pp_boundary(mechanism, sev, seed=seed)))
    # Real flchain anchor: the quarterly person-period design (the granularity that
    # stresses the MPS float32 solve), so the proof includes discrete-time's actual
    # data, not only synthetic designs.
    time_, event, X, names = datasets.load_flchain()
    bounds = datasets.flchain_interval_bounds(time_, event, 91.3)   # quarterly
    X_pp, y_pp, n_int = build_person_period(time_, event, X, bounds)
    out.append(("flchain_quarterly", 0.0, (X_pp, y_pp, n_int, len(names))))
    return out


def generate(host: str, device: str) -> Path:
    from pystatistics.regression import Design, fit
    from pystatistics.core.exceptions import NumericalError, PyStatisticsError

    env = env_manifest(device=device, host=host)
    require_pypi(env)
    has_fp64 = (device == "cuda")

    records: list[dict[str, Any]] = []
    brows: list[dict[str, Any]] = []     # boundary table
    irows: list[dict[str, Any]] = []     # inference-SE table
    silent_wrong: list[dict[str, Any]] = []
    inf_flagged: list[dict[str, Any]] = []

    for mechanism, sev, (X_pp, y_pp, n_int, n_cov) in _designs():
        X_pp = np.asarray(X_pp, float)
        y_pp = np.asarray(y_pp, float).ravel()
        d = Design.from_arrays(X_pp, y_pp)
        npcond = eq_cond(X_pp)
        cov = slice(-n_cov, None)        # covariate tail = the inference target

        cpu = fit(d, family="binomial", backend="cpu")
        ref_dev = float(cpu.deviance)
        ref_cov_coef = np.asarray(cpu.coefficients, float)[cov]
        ref_cov_se = np.asarray(cpu.standard_errors, float)[cov]
        n_events = int(y_pp.sum())
        ref_converged = bool(getattr(cpu, "converged", True))

        # Reference-degeneracy guard (R9/G3): if the fp64 reference itself is
        # undefined (non-convergent / too few events), there is no optimum to call
        # fp32 "wrong" against — record and EXCLUDE, do not judge the gate here.
        if (not ref_converged) or n_events < _MIN_EVENTS:
            brow = {"mechanism": mechanism, "severity": sev, "np_cond": npcond,
                    "n_intervals": n_int, "n_cov": n_cov,
                    "plain_status": f"ref-degenerate(events={n_events},conv={ref_converged})",
                    "accepted_coef_rel_vs_fp64": None, "accepted_dev_rel_vs_fp64": None,
                    "forced_dev_rel_vs_fp64": None, "fp64_dev_rel_vs_cpu": None,
                    "classification": "ref-degenerate (excluded)"}
            irow = {"mechanism": mechanism, "severity": sev, "eq_cond": npcond,
                    "gpu_status": "ref-degenerate", "coef_rel_gpu_vs_cpu": None,
                    "se_rel_gpu_vs_cpu": None,
                    "cpu_se_max": float(np.nanmax(ref_cov_se)), "gpu_se_max": None,
                    "inference_verdict": "excluded (reference undefined)"}
            brows.append(brow)
            irows.append(irow)
            records.append({"engine": f"discrete_r13:{device}", "dataset": mechanism,
                            "n": X_pp.shape[0], "p": X_pp.shape[1], **brow})
            print(f"  {mechanism:16} sev={sev:<8g} cond={npcond:.1e} "
                  f"EXCLUDED (events={n_events}, ref_conv={ref_converged})", flush=True)
            continue

        plain_status = classification = ""
        acc_coef_rel = acc_dev_rel = forced_dev_rel = None
        gpu_status = inf_verdict = ""
        coef_rel = se_rel_cpu = gpu_se_max = None
        try:
            g = fit(d, family="binomial", backend="gpu")
            plain_status = "converged" if bool(getattr(g, "converged", True)) else "not_converged"
            gco = np.asarray(g.coefficients, float)[cov]
            gse = np.asarray(g.standard_errors, float)[cov]
            acc_coef_rel = coef_rel = _rel(gco, ref_cov_coef)
            acc_dev_rel = abs(float(g.deviance) - ref_dev) / max(abs(ref_dev), 1e-300)
            se_rel_cpu = _rel(gse, ref_cov_se)
            gpu_se_max = float(np.nanmax(gse))
            classification = "safe" if acc_dev_rel <= _DEV_TOL else "SILENT-WRONG"
            # Inference: do the accepted covariate SEs flag instability (like R) or
            # understate it (a coefficient wrong AND looking precise)?
            understates = bool(np.any(gse < _SE_UNDERSTATE_FRAC * ref_cov_se))
            gpu_status = "accepted"
            inf_verdict = ("inference-warned"
                           if (se_rel_cpu <= _SE_REL_TOL and not understates)
                           else "INFERENCE-SILENT-WRONG")
        except (NumericalError, PyStatisticsError) as exc:
            plain_status = f"refused:{type(exc).__name__}"
            gpu_status = "refused"
            inf_verdict = "refused (fail-loud)"
            try:
                gf = fit(d, family="binomial", backend="gpu", force=True)
                forced_dev_rel = abs(float(gf.deviance) - ref_dev) / max(abs(ref_dev), 1e-300)
                classification = ("refused-true-positive"
                                  if (not np.isfinite(forced_dev_rel) or forced_dev_rel > _DEV_TOL)
                                  else "refused-force-recovers")
            except Exception:  # noqa: BLE001 - forced fit blew up -> refusal warranted
                classification = "refused-true-positive"
                forced_dev_rel = float("inf")

        fp64_dev_rel = None
        if has_fp64:
            try:
                g64 = fit(d, family="binomial", backend="gpu_fp64")
                fp64_dev_rel = abs(float(g64.deviance) - ref_dev) / max(abs(ref_dev), 1e-300)
            except Exception:  # noqa: BLE001
                fp64_dev_rel = None

        brow = {
            "mechanism": mechanism, "severity": sev, "np_cond": npcond,
            "n_intervals": n_int, "n_cov": n_cov, "plain_status": plain_status,
            "accepted_coef_rel_vs_fp64": acc_coef_rel,
            "accepted_dev_rel_vs_fp64": acc_dev_rel,
            "forced_dev_rel_vs_fp64": forced_dev_rel,
            "fp64_dev_rel_vs_cpu": fp64_dev_rel,
            "classification": classification,
        }
        irow = {
            "mechanism": mechanism, "severity": sev, "eq_cond": npcond,
            "gpu_status": gpu_status, "coef_rel_gpu_vs_cpu": coef_rel,
            "se_rel_gpu_vs_cpu": se_rel_cpu,
            "cpu_se_max": float(np.nanmax(ref_cov_se)), "gpu_se_max": gpu_se_max,
            "inference_verdict": inf_verdict,
        }
        brows.append(brow)
        irows.append(irow)
        records.append({"engine": f"discrete_r13:{device}", "dataset": mechanism,
                        "n": X_pp.shape[0], "p": X_pp.shape[1], **brow,
                        "se_rel_gpu_vs_cpu": se_rel_cpu, "inference_verdict": inf_verdict})
        if classification == "SILENT-WRONG":
            silent_wrong.append(brow)
        if inf_verdict == "INFERENCE-SILENT-WRONG":
            inf_flagged.append(irow)
        print(f"  {mechanism:16} sev={sev:<8g} cond={npcond:.1e} {plain_status:22} "
              f"-> {classification:22}"
              + (f" accDevRel={acc_dev_rel:.1e}" if acc_dev_rel is not None else "")
              + (f" fDevRel={forced_dev_rel:.1e}" if forced_dev_rel is not None else "")
              + f" | {gpu_status}->{inf_verdict}"
              + (f" seRel={se_rel_cpu:.1e}" if se_rel_cpu is not None else ""),
              flush=True)

    config = {"study": "discrete_r13", "device": device, "dev_tol": _DEV_TOL,
              "se_understate_frac": _SE_UNDERSTATE_FRAC, "se_rel_tol": _SE_REL_TOL,
              "sweeps": SWEEPS, "anchor": "survival::flchain quarterly person-period"}
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "survival" / f"v{env['pystatistics_version']}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"discrete_r13_{device}_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    bcols = ["mechanism", "severity", "np_cond", "n_intervals", "n_cov",
             "plain_status", "accepted_coef_rel_vs_fp64", "accepted_dev_rel_vs_fp64",
             "forced_dev_rel_vs_fp64", "fp64_dev_rel_vs_cpu", "classification"]
    icols = ["mechanism", "severity", "eq_cond", "gpu_status", "coef_rel_gpu_vs_cpu",
             "se_rel_gpu_vs_cpu", "cpu_se_max", "gpu_se_max", "inference_verdict"]
    _write_csv(out_dir / f"{stem}_boundary.csv", brows, bcols)
    _write_csv(out_dir / f"{stem}_inference.csv", irows, icols)
    print(f"\nwrote {run_path}")
    from collections import Counter
    print("boundary tally:", dict(Counter(r["classification"] for r in brows)))
    print("inference tally:", dict(Counter(r["inference_verdict"] for r in irows)))
    if silent_wrong or inf_flagged:
        print("\n" + "!" * 72)
        print(f"!!! {len(silent_wrong)} SILENT-WRONG + {len(inf_flagged)} "
              f"INFERENCE-SILENT-WRONG on {device} — STOP, surface to user (A6/R12/R13).")
        print("!" * 72)
    return run_path


def _write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--device", default="mps", choices=["mps", "cuda"])
    args = ap.parse_args()
    generate(args.host, args.device)


if __name__ == "__main__":
    main()
