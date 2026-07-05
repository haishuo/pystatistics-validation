"""Flatten the frozen timeseries run JSONs into summary CSVs + write manifest.json.

One job: a DETERMINISTIC transform of the frozen run artifacts
(artifacts/timeseries/v<ver>/runs/*.json) into the flat per-study summary CSVs the
renderer tables, plus the manifest that ties them together. Numbers are only
read from the frozen JSONs — never authored here (R5).

Run:  python drivers/timeseries/build_manifest.py <version>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_RUNS = Path(__file__).resolve().parents[2] / "artifacts/timeseries/v{ver}/runs"


def _load(runs: Path, name: str) -> dict:
    return json.loads((runs / name).read_text())


def _checks(doc: dict) -> list[dict]:
    out = []
    for rec in doc["records"]:
        out += rec.get("checks", [])
    return out


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _g(d: dict, *keys, default=""):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def deterministic(runs: Path) -> None:
    doc = _load(runs, "deterministic.json")
    det_rows, stat_rows, ndiff_rows = [], [], []
    for c in _checks(doc):
        g, series = c.get("group"), c.get("series")
        if g in ("acf_pacf", "diff", "decompose", "stl"):
            case = c.get("config") or c.get("type") or ""
            for q in ("acf", "pacf", "diff", "seasonal", "trend", "remainder",
                      "random"):
                if isinstance(c.get(q), dict) and "max_abs" in c[q]:
                    det_rows.append([g, series, case, q, c[q]["max_abs"],
                                     c[q]["max_rel"]])
        elif g in ("adf", "kpss"):
            stat_rows.append([
                g, series, c.get("regression", c.get("null", "")),
                _g(c, "stat_vs_tseries", "py") or _g(c, "stat", "py"),
                _g(c, "stat_vs_tseries", "r") or _g(c, "stat", "r"),
                _g(c, "stat_vs_tseries", "abs") or _g(c, "stat", "abs"),
                _g(c, "p_vs_statsmodels", "py") or c.get("p_py", ""),
                _g(c, "p_vs_statsmodels", "r") or c.get("p_r", ""),
                c.get("n_lags", c.get("lag_py", "")), c.get("pass")])
        elif g == "ndiffs":
            ndiff_rows.append([series, c.get("test"), c.get("py"), c.get("r"),
                               c.get("pass"), c.get("note", "")])
    _write_csv(runs / "g1_deterministic_summary.csv",
               ["group", "series", "case", "quantity", "max_abs", "max_rel"],
               det_rows)
    _write_csv(runs / "g1_stationarity_summary.csv",
               ["test", "series", "spec", "stat_py", "stat_r", "stat_abs",
                "p_py", "p_ref", "lag", "pass"], stat_rows)
    _write_csv(runs / "g1_ndiffs_summary.csv",
               ["series", "test", "py", "r", "pass", "note"], ndiff_rows)


def mle(runs: Path) -> None:
    doc = _load(runs, "mle.json")
    rows = []
    for c in _checks(doc):
        g = c.get("group", "")
        spec = (c.get("model") or str(c.get("order", "")) +
                (str(c.get("seasonal")) if c.get("seasonal") else ""))
        rows.append([
            g, c.get("series", ""), spec,
            _g(c, "loglik", "py"), _g(c, "loglik", "r"), _g(c, "loglik", "abs"),
            _g(c, "aic", "abs", default=""),
            _g(c, "coef", "max_abs", default="") or _g(c, "fitted", "max_abs",
                                                       default=""),
            c.get("selected", c.get("order_match", "")), c.get("pass", "")])
    _write_csv(runs / "g1_mle_summary.csv",
               ["group", "series", "spec", "loglik_py", "loglik_r", "loglik_abs",
                "aic_abs", "coef_or_fit_abs", "selected", "pass"], rows)


def fidelity(runs: Path) -> None:
    doc = _load(runs, "fidelity.json")
    rows = [[c.get("key"), c.get("desc"), c.get("expected"), c.get("got"),
             c.get("pass")] for c in _checks(doc)]
    _write_csv(runs / "g2_fidelity_summary.csv",
               ["key", "desc", "expected", "got", "pass"], rows)


def performance(runs: Path) -> None:
    doc = _load(runs, "performance.json")
    rows = []
    for st in doc["records"][0]["studies"]:
        func, ps, rs = st["func"], st["py_slope"], st["r_slope"]
        for r in st["rows"]:
            rows.append([func, r["n"], round(r["py_s"] * 1000, 4),
                         round(r["r_s"] * 1000, 4),
                         round(r["ratio_py_over_r"], 3), ps, rs])
    _write_csv(runs / "g3_performance_summary.csv",
               ["func", "n", "py_ms", "r_ms", "ratio_py_over_r",
                "py_slope", "r_slope"], rows)


def batch_contract(runs: Path) -> None:
    doc = _load(runs, "batch_contract.json")
    rows = []
    for c in _checks(doc):
        detail = ""
        if c.get("key") == "partial_fail":
            detail = f"n_nan={c.get('n_nan')} max_finite_ar={c.get('max_finite_ar'):.3f} warned={c.get('warned')}"
        elif c.get("key") == "all_good":
            detail = f"converged={c.get('converged')}/{c.get('k')} warned={c.get('warned')}"
        elif "raised" in c:
            detail = f"raised={c.get('got')}"
        elif c.get("key") == "batch_vs_single_series":
            detail = f"ar_abs={c['ar']['max_abs']:.1e} ma_abs={c['ma']['max_abs']:.1e}"
        rows.append([c.get("key"), c.get("desc"), c.get("pass"), detail])
    _write_csv(runs / "batch_contract_summary.csv",
               ["key", "desc", "pass", "detail"], rows)


def gpu(runs: Path) -> None:
    doc = _load(runs, "gpu_cuda_forge.json")
    scal = [[r["K"], r["n"], round(r["cpu_s"] * 1000, 1),
             round(r["gpu_s"] * 1000, 1), round(r["gpu_fp64_s"] * 1000, 1),
             round(r["speedup_gpu_vs_cpu"], 2), round(r["speedup_gpu64_vs_cpu"], 2),
             f"{r['fp32_ar_rel']:.2e}", f"{r['fp64_ar_rel']:.2e}"]
            for r in doc["scaling"]]
    _write_csv(runs / "gpu_scaling_summary.csv",
               ["K", "n", "cpu_ms", "gpu_fp32_ms", "gpu_fp64_ms",
                "speedup_fp32_vs_cpu", "speedup_fp64_vs_cpu",
                "fp32_ar_rel", "fp64_ar_rel"], scal)
    fid = [[s["tag"], s["cpu_result"], s["gpu_result"], s.get("contract_parity"),
            s.get("gpu_no_nonstationary_number")] for s in doc["stress_r12_r13"]]
    _write_csv(runs / "gpu_fidelity_summary.csv",
               ["tag", "cpu_result", "gpu_result", "contract_parity",
                "gpu_no_nonstationary_number"], fid)


def build_manifest(runs: Path, ver: str) -> dict:
    env = _load(runs, "deterministic.json")["env"]
    gpu_env = _load(runs, "gpu_cuda_forge.json")["env"]
    return {
        "schema": "validation-artifact-manifest/v1",
        "subsystem": "timeseries",
        "pystatistics_version": ver,
        "install_source": "pypi",
        "evidence_state": "native-harness",
        "frozen_utc": "2026-07-06",
        "provenance": {"note":
            "First-time validation of pystatistics.timeseries (the corpus's "
            "largest module) across its whole public surface. The pass drove two "
            "R16 emergency stops for silent-wrong defects: stl (4.6.1) returned a "
            "trend-leakage decomposition on strongly-trending series (fixed 4.6.3, "
            "now machine-precision vs stats::stl, confirmed against statsmodels); "
            "seasonal ARIMA information criteria (4.6.3) counted the expanded "
            "multiplicative-polynomial coefficients instead of free parameters, "
            "silently inflating AIC/AICc and mis-selecting auto_arima's seasonal "
            "model (fixed 4.6.4). Bundled fixes: ETS 'ZZZ' auto-selection, ndiffs "
            "default test kpss, ADF default regression 'ct' + corrected MacKinnon "
            "p-values, KPSS bandwidth. 4.6.5 aligned the arima_batch GPU/CPU "
            "fail-loud contract. 4.6.6 JIT-compiled the ETS state-space recursion "
            "with numba (@njit cache=True fastmath=False, matching the module's "
            "ARIMA-Kalman/STL kernels), flipping ets from ~10.7x slower than R to "
            "0.43x with BIT-IDENTICAL estimates (max diff 0.0 vs 4.6.5) — an R6 "
            "optimize-and-re-validate cycle closing the module's last gap. "
            "Identical float64 inputs to both engines (each "
            "series dumped from the central HDF5 store; the R reference reads the "
            "same values). References: stats/forecast/tseries; statsmodels as an "
            "independent triangulation for stl and the ADF p-value."},
        "hosts": {
            "arm+r": {"device": f"CPU (numpy {env.get('numpy','?')}) + R "
                      "stats/forecast 9.0.0/tseries 0.10.61"},
            "forge-cuda": {"device": f"{gpu_env.get('device','?')} "
                           f"(torch {gpu_env.get('torch','?')}, CUDA "
                           f"{gpu_env.get('cuda','?')})"},
        },
        "reference": {"name": "R stats/forecast/tseries", "kind": "cran"},
        "studies": [
            {"id": "g1_deterministic", "title": "G1 correctness — TIER 1 (tight): "
             "deterministic quantities vs R (machine precision)",
             "device": ["cpu"], "claim": "agreement", "host": "arm+r",
             "summary": "runs/g1_deterministic_summary.csv",
             "summary_cols": ["group", "series", "case", "quantity", "max_abs",
                              "max_rel"]},
            {"id": "g1_stationarity", "title": "G1 correctness — stationarity "
             "tests vs tseries (statistic tight; p vs statsmodels MacKinnon)",
             "device": ["cpu"], "claim": "agreement", "host": "arm+r",
             "summary": "runs/g1_stationarity_summary.csv",
             "summary_cols": ["test", "series", "spec", "stat_py", "stat_r",
                              "stat_abs", "p_py", "p_ref", "lag", "pass"]},
            {"id": "g1_ndiffs", "title": "G1 correctness — ndiffs vs "
             "forecast::ndiffs (default test=kpss)",
             "device": ["cpu"], "claim": "agreement", "host": "arm+r",
             "summary": "runs/g1_ndiffs_summary.csv",
             "summary_cols": ["series", "test", "py", "r", "pass", "note"]},
            {"id": "g1_mle", "title": "G1 correctness — TIER 2 (two-tier): "
             "arima / ets / auto_arima MLE vs R (matched method)",
             "device": ["cpu"], "claim": "agreement", "host": "arm+r",
             "summary": "runs/g1_mle_summary.csv",
             "summary_cols": ["group", "series", "spec", "loglik_py", "loglik_r",
                              "loglik_abs", "aic_abs", "coef_or_fit_abs",
                              "selected", "pass"]},
            {"id": "g2_fidelity", "title": "G2 fidelity — fail-loud, no silent "
             "substitution", "device": ["cpu"], "claim": "fail-loud",
             "host": "arm+r", "summary": "runs/g2_fidelity_summary.csv",
             "summary_cols": ["key", "desc", "expected", "got", "pass"]},
            {"id": "g3_performance", "title": "G3 performance — CPU vs R across "
             "series length", "device": ["cpu"], "claim": "speed", "host": "arm+r",
             "summary": "runs/g3_performance_summary.csv",
             "summary_cols": ["func", "n", "py_ms", "r_ms", "ratio_py_over_r",
                              "py_slope", "r_slope"]},
            {"id": "batch_contract", "title": "arima_batch failure contract "
             "(CPU) + correctness vs single-series", "device": ["cpu"],
             "claim": "fail-loud", "host": "arm+r",
             "summary": "runs/batch_contract_summary.csv",
             "summary_cols": ["key", "desc", "pass", "detail"]},
            {"id": "gpu_scaling", "title": "arima_batch Whittle GPU (CUDA) — "
             "scaling vs CPU; fp32≡fp64 accuracy (R11)", "device": ["cuda"],
             "claim": "speed", "host": "forge-cuda",
             "summary": "runs/gpu_scaling_summary.csv",
             "summary_cols": ["K", "n", "cpu_ms", "gpu_fp32_ms", "gpu_fp64_ms",
                              "speedup_fp32_vs_cpu", "speedup_fp64_vs_cpu",
                              "fp32_ar_rel", "fp64_ar_rel"],
             "note": "GPU beats CPU beyond K~500 (7.6x fp32 / 6.0x fp64 at "
                     "K=2000). fp32_ar_rel == fp64_ar_rel at every K => the gap "
                     "vs exact ML is the Adam optimizer, not precision (R11)."},
            {"id": "gpu_fidelity", "title": "arima_batch GPU/CPU contract parity "
             "on non-stationary batches (R12/R13/R14)", "device": ["cuda"],
             "claim": "fail-loud", "host": "forge-cuda",
             "summary": "runs/gpu_fidelity_summary.csv",
             "summary_cols": ["tag", "cpu_result", "gpu_result",
                              "contract_parity", "gpu_no_nonstationary_number"],
             "note": "4.6.5: GPU honors the same contract as CPU — no "
                     "non-stationary AR returned as a plain number "
                     "(gpu_no_nonstationary_number=True everywhere)."},
        ],
    }


def main() -> None:
    ver = sys.argv[1] if len(sys.argv) > 1 else "4.6.5"
    runs = Path(str(_RUNS).format(ver=ver))
    deterministic(runs)
    mle(runs)
    fidelity(runs)
    performance(runs)
    batch_contract(runs)
    gpu(runs)
    manifest = build_manifest(runs, ver)
    out = runs.parent / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} + {len(manifest['studies'])} study CSVs")


if __name__ == "__main__":
    main()
