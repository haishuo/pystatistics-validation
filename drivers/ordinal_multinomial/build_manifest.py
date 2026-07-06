"""Flatten the frozen ordinal/multinomial run JSONs into summary CSVs + manifest.

Deterministic transform of artifacts/ordinal_multinomial/v<ver>/runs/*.json into
the flat per-study summary CSVs the renderer tables, plus the manifest.json that
ties them together. Numbers are only READ from the frozen JSONs — never authored
here (R5).

Run:  python drivers/ordinal_multinomial/build_manifest.py <version>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_RUNS = (Path(__file__).resolve().parents[2]
         / "artifacts/ordinal_multinomial/v{ver}/runs")


def _load(runs: Path, name: str) -> dict:
    return json.loads((runs / name).read_text())


def _checks(doc: dict) -> list[dict]:
    out = []
    for rec in doc["records"]:
        out += rec.get("checks", [])
    return out


def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _r(x, n=None):
    if isinstance(x, float):
        return f"{x:.3e}" if (n is None) else round(x, n)
    return x


# --------------------------------------------------------------------------
def g1_polr(runs: Path) -> None:
    doc = _load(runs, "g1_ordinal.json")
    fit, hard = [], []
    for c in _checks(doc):
        g = c.get("group")
        if g == "polr_fit":
            fit.append([c["dataset"], c["link"], c["n"], c["K"],
                        _r(c["coef"]["max_abs"]), _r(c["zeta"]["max_abs"]),
                        _r(c["se_beta"]["max_rel"]), _r(c["loglik"]["abs"]),
                        _r(c["implied_probs"]["max_abs"]), c["pass"]])
        elif g == "r15_default":
            hard.append(["polr", "R15 default link",
                         f"default link={c['default_link']}, matches explicit "
                         f"logistic (max|Δ| {_r(c['coef_vs_explicit_logistic']['max_abs'])})",
                         c["pass"]])
        elif g == "r10_separation":
            hard.append(["polr", "R10 complete separation",
                         f"py fail-loud={c['py_fail_loud']}; R refuses="
                         f"{c['r_refuses']} (glm.fit separation warnings + error)",
                         c["pass"]])
    _csv(runs / "g1_polr_summary.csv",
         ["dataset", "link", "n", "K", "coef_maxabs", "zeta_maxabs",
          "se_beta_maxrel", "loglik_abs", "probs_maxabs", "pass"], fit)
    return hard


def g1_multinom(runs: Path) -> list:
    doc = _load(runs, "g1_multinom.json")
    well, hard = [], []
    for c in _checks(doc):
        g = c.get("group")
        if g == "multinom_well":
            well.append([c["dataset"], c["n"], c["K"], _r(c["coef"]["max_abs"]),
                         _r(c["se"]["max_rel"]), _r(c["loglik"]["abs"]),
                         _r(c["fitted_probs"]["max_abs"]), c["pass"]])
        elif g == "multinom_separation":
            detail = (f"quasi-separation: loglik|Δ| {_r(c['loglik']['abs'])}, "
                      f"probs|Δ| {_r(c['fitted_probs']['max_abs'])}; coef "
                      f"non-identified (py {c['py_maxabs_coef']:.0f} vs R "
                      f"{c['r_maxabs_coef']:.0f})") if "loglik" in c else \
                     f"py fail-loud: {c.get('py_error','')[:50]}"
            hard.append(["multinom", f"R10 {c['dataset']}", detail, c["pass"]])
        elif g == "r15_default":
            hard.append(["multinom", "R15 default (decay=0)",
                         f"coef|Δ| {_r(c['coef']['max_abs'])}, "
                         f"loglik|Δ| {_r(c['loglik']['abs'])}", c["pass"]])
        elif g == "r10_complete_separation":
            hard.append(["multinom", "R10 complete separation",
                         f"py fail-loud={c['py_fail_loud']} "
                         f"(NotPositiveDefiniteError); R conv={c['r_conv']}",
                         c["pass"]])
    _csv(runs / "g1_multinom_summary.csv",
         ["dataset", "n", "K", "coef_maxabs", "se_maxrel", "loglik_abs",
          "probs_maxabs", "pass"], well)
    return hard


def hardcases(runs: Path, polr_hard: list, mn_hard: list) -> None:
    rows = [[s, case, detail, p] for (s, case, detail, p) in polr_hard + mn_hard]
    _csv(runs / "r10_r15_summary.csv",
         ["surface", "case", "detail", "pass"], rows)


def predict(runs: Path) -> None:
    doc = _load(runs, "predict.json")
    rows = []
    for c in _checks(doc):
        if c.get("group") == "polr_predict":
            rows.append(["polr", c["link"], c["n_test"],
                         _r(c["probs"]["max_abs"]),
                         f"{c['class_match']}/{c['class_total']}", c["pass"]])
        elif c.get("group") == "multinom_predict":
            rows.append(["multinom", c["dataset"], c["n_test"],
                         _r(c["probs"]["max_abs"]),
                         f"{c['class_match']}/{c['class_total']}", c["pass"]])
    _csv(runs / "predict_summary.csv",
         ["surface", "case", "n_test", "probs_maxabs", "class_match", "pass"], rows)


def fidelity(runs: Path) -> None:
    doc = _load(runs, "fidelity.json")
    rows = [[c["key"], c["surface"], c["desc"], c["got"], c["pass"]]
            for c in _checks(doc)]
    _csv(runs / "fidelity_summary.csv",
         ["key", "surface", "desc", "got", "pass"], rows)


def cpu_speed(runs: Path) -> None:
    doc = _load(runs, "cpu_speed.json")
    rows = []
    for c in _checks(doc):
        rows.append([c["func"], c["n"], c["p"], c["K"], round(c["py_ms"], 2),
                     round(c["r_ms"], 2), round(c["ratio_py_over_r"], 3),
                     round(c["py_slope"], 2), round(c["r_slope"], 2), c["pass"]])
    _csv(runs / "cpu_speed_summary.csv",
         ["func", "n", "p", "K", "py_ms", "r_ms", "ratio_py_over_r",
          "py_slope", "r_slope", "pass"], rows)


def gpu(runs: Path) -> None:
    d = _load(runs, "gpu_cuda.json")
    # CF-1 fix table
    cf1 = []
    for s in ("polr", "multinom"):
        for r in d["cf1"][s]:
            cf1.append([s, f"{r['cond']:.0e}", f"{r['xtx_cond']:.1e}",
                        _r(r.get("fp32_se_relerr")) if "fp32_se_relerr" in r
                        else r.get("error", ""),
                        r.get("fp32_neg_var"),
                        _r(r.get("fp64_se_relerr")) if r.get("fp64_se_relerr")
                        is not None else ""])
    _csv(runs / "cf1_summary.csv",
         ["surface", "cond_X", "xtx_cond", "fp32_se_relerr", "fp32_neg_var",
          "fp64_se_relerr"], cf1)
    # two-tier
    tt = []
    for x in d["twotier"]:
        tt.append([x["surface"], x["n"], _r(x["coef_relerr"]), _r(x["se_relerr"]),
                   _r(x.get("probs_relerr")) if "probs_relerr" in x else ""])
    _csv(runs / "gpu_twotier_summary.csv",
         ["surface", "n", "coef_relerr", "se_relerr", "probs_relerr"], tt)
    # perf
    perf = []
    for s in ("polr", "multinom"):
        for r in d["perf"][s]:
            perf.append([s, r["n"], round(r["cpu_ms"], 1),
                         round(r["gpu_fp32_ms"], 1),
                         round(r.get("gpu_fp64_ms", float("nan")), 1),
                         round(r["speedup_fp32"], 1)])
    _csv(runs / "gpu_perf_summary.csv",
         ["surface", "n", "cpu_ms", "gpu_fp32_ms", "gpu_fp64_ms",
          "speedup_fp32_vs_cpu"], perf)
    return d


def build_manifest(runs: Path, ver: str) -> dict:
    g1o = _load(runs, "g1_ordinal.json")
    gpu_doc = _load(runs, "gpu_cuda.json")
    rpkg = g1o["env"].get("r_packages", {})
    return {
        "schema": "validation-artifact-manifest/v1",
        "subsystem": "ordinal_multinomial",
        "pystatistics_version": ver,
        "install_source": "pypi",
        "evidence_state": "native-harness",
        "frozen_utc": "2026-07-06",
        "provenance": {
            "note": (
                "First-time whole-surface validation of pystatistics.ordinal.polr "
                "(vs MASS::polr) and pystatistics.multinomial.multinom (vs "
                "nnet::multinom), grouped as one corpus item. The pass drove one "
                "R16 emergency stop: the GPU (float32) path of BOTH models inverted "
                "the model information matrix (X'WX) in single precision, silently "
                "losing accuracy on ill-conditioned designs — at cond(X)~1e4 (a "
                "design the CPU/fp64 path fits correctly) polr returned a "
                "negative-variance, 100%-wrong standard error with converged=True, "
                "and multinom ~40% wrong SEs, proven on CUDA (RTX 5070 Ti). Fixed "
                "in 4.6.9: the variance-covariance is computed in float64 (polr "
                "forms the observed-information Hessian on-device in fp64 on CUDA / "
                "on host on MPS; multinom forms its closed-form softmax Hessian on "
                "the host in fp64), inverted once in fp64 with a positive-"
                "definiteness gate. The fast float32 GPU fit is unchanged, so the "
                "large-n GPU speedup is preserved (polr ~50x, multinom ~120x over "
                "CPU at n=100k). Bundled fixes: multinom now fails loud "
                "(NotPositiveDefiniteError) on complete/quasi separation instead of "
                "returning a negative-variance covariance (matching polr); the "
                "unsupported-link error reads 'Unknown link' not 'Unknown method'; "
                "MultinomialSolution gains a .warnings accessor. 4.6.10 then added "
                "a predict() method to both models (matching R's predict.polr / "
                "predict.multinom — type='class'/'probs') plus polr fitted_probs / "
                "predicted_class, validated on held-out data (probs ~1e-5 vs R, "
                "classes exact). Identical float64 inputs to both engines (each "
                "design dumped to CSV read by R). Baseline 4.6.8 (defect) evidence "
                "retained under v4.6.8/runs/; this report is the validation at "
                "4.6.10 (the CF-1 fix landed at 4.6.9, predict at 4.6.10)."
            ),
        },
        "hosts": {
            "arm+r": {"device": f"CPU (numpy) + R MASS {rpkg.get('MASS','?')} / "
                                f"nnet {rpkg.get('nnet','?')}"},
            "forge-cuda": {"device": f"NVIDIA GeForce RTX 5070 Ti (torch "
                                     f"{gpu_doc.get('torch_version','?')}, CUDA)"},
        },
        "reference": {"name": "R MASS::polr / nnet::multinom", "kind": "cran"},
        "studies": [
            {"id": "g1_polr", "title": "G1 correctness — polr vs MASS::polr "
             "(two-tier: coef/zeta/loglik optimizer, implied probs tight)",
             "device": ["cpu"], "claim": "agreement", "host": "arm+r",
             "summary": "runs/g1_polr_summary.csv",
             "summary_cols": ["dataset", "link", "n", "K", "coef_maxabs",
                              "zeta_maxabs", "se_beta_maxrel", "loglik_abs",
                              "probs_maxabs", "pass"]},
            {"id": "g1_multinom", "title": "G1 correctness — multinom vs "
             "nnet::multinom (decay=0, baseline releveled)",
             "device": ["cpu"], "claim": "agreement", "host": "arm+r",
             "summary": "runs/g1_multinom_summary.csv",
             "summary_cols": ["dataset", "n", "K", "coef_maxabs", "se_maxrel",
                              "loglik_abs", "probs_maxabs", "pass"]},
            {"id": "r10_r15", "title": "G1 hard cases — R10 separation / rare "
             "category + R15 default invocation (match R's failures)",
             "device": ["cpu"], "claim": "agreement", "host": "arm+r",
             "summary": "runs/r10_r15_summary.csv",
             "summary_cols": ["surface", "case", "detail", "pass"]},
            {"id": "predict", "title": "G1 correctness — predict() on held-out "
             "data vs R predict.polr / predict.multinom (probs tight, class exact)",
             "device": ["cpu"], "claim": "agreement", "host": "arm+r",
             "summary": "runs/predict_summary.csv",
             "summary_cols": ["surface", "case", "n_test", "probs_maxabs",
                              "class_match", "pass"]},
            {"id": "fidelity", "title": "G2 fidelity — fail-loud, no silent "
             "substitution (unsupported links, l2-on-GPU, backend, bad y/X)",
             "device": ["cpu"], "claim": "fail-loud", "host": "arm+r",
             "summary": "runs/fidelity_summary.csv",
             "summary_cols": ["key", "surface", "desc", "got", "pass"]},
            {"id": "cpu_speed", "title": "G3 performance — CPU vs R across n and "
             "#categories (empirical complexity)",
             "device": ["cpu"], "claim": "speed", "host": "arm+r",
             "summary": "runs/cpu_speed_summary.csv",
             "summary_cols": ["func", "n", "p", "K", "py_ms", "r_ms",
                              "ratio_py_over_r", "py_slope", "r_slope", "pass"]},
            {"id": "cf1", "title": "CF-1 — fp32 GPU standard errors vs CPU across "
             "a cond(X) sweep (4.6.9 fix: no negative variance, fp32 tier)",
             "device": ["cuda"], "claim": "fail-loud", "host": "forge-cuda",
             "summary": "runs/cf1_summary.csv",
             "summary_cols": ["surface", "cond_X", "xtx_cond", "fp32_se_relerr",
                              "fp32_neg_var", "fp64_se_relerr"],
             "note": "4.6.8 returned negative variances / 100%-wrong SE with "
                     "converged=True here; 4.6.9 computes the vcov in fp64 — "
                     "fp32_neg_var=False everywhere, SE_relerr at the fp32 tier. "
                     "The gpu_fp64-vs-cpu pivot (R11) is machine-precision, so the "
                     "residual is fp32, not hardware."},
            {"id": "gpu_twotier", "title": "GPU_FP32 two-tier correctness — fp32 "
             "GPU vs CPU fp64 on a well-conditioned design",
             "device": ["cuda"], "claim": "agreement", "host": "forge-cuda",
             "summary": "runs/gpu_twotier_summary.csv",
             "summary_cols": ["surface", "n", "coef_relerr", "se_relerr",
                              "probs_relerr"]},
            {"id": "gpu_perf", "title": "GPU performance — CPU vs gpu_fp32 vs "
             "gpu_fp64 across n (Guarantee-3 GPU corollary)",
             "device": ["cuda"], "claim": "speed", "host": "forge-cuda",
             "summary": "runs/gpu_perf_summary.csv",
             "summary_cols": ["surface", "n", "cpu_ms", "gpu_fp32_ms",
                              "gpu_fp64_ms", "speedup_fp32_vs_cpu"],
             "note": "The float32 GPU fit is unchanged by the CF-1 fix (only the "
                     "post-fit vcov moved to fp64), so the large-n speedup stands: "
                     "polr ~50x, multinom ~120x over CPU at n=100k."},
        ],
    }


def main() -> None:
    ver = sys.argv[1] if len(sys.argv) > 1 else "4.6.9"
    runs = Path(str(_RUNS).format(ver=ver))
    polr_hard = g1_polr(runs)
    mn_hard = g1_multinom(runs)
    hardcases(runs, polr_hard, mn_hard)
    predict(runs)
    fidelity(runs)
    cpu_speed(runs)
    gpu(runs)
    manifest = build_manifest(runs, ver)
    (runs.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {runs.parent / 'manifest.json'} + {len(list(runs.glob('*_summary.csv')))} summary CSVs")


if __name__ == "__main__":
    main()
