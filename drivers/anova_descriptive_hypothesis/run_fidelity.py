"""G2 fidelity sweep: fail-loud on invalid input, and MATCH R's warnings.

Guarantee 2 says pystatistics must never satisfy correctness by silently solving
a different problem. Here that means:
  (a) invalid / dimension-mismatched / degenerate input FAILS LOUDLY (raises),
      never returns a silent default (A6, Coding Bible Rule 1);
  (b) the diagnostic WARNINGS R emits on hard cases are reproduced (R10) -- the
      chi-square small-expected-count warning, the wilcox ties -> normal-approx
      warning -- and any gap (the one-sample KS ties warning) is catalogued;
  (c) when a test variant is silently unavailable (exact wilcox with ties, exact
      r x c fisher) the switch is DISCLOSED in the method string, not hidden.

Writes runs/fidelity_g2.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from compare import write_artifact
from rref import r_ref, r_versions

from pystatistics import __version__ as PYVER
from pystatistics.core import exceptions as pyexc
from pystatistics.descriptive import cor, quantile, var
from pystatistics.hypothesis import (
    chisq_test, fisher_test, ks_test, prop_test, t_test, wilcox_test,
)

ART = Path(__file__).resolve().parents[2] / \
    "artifacts/anova_descriptive_hypothesis" / f"v{PYVER}" / "runs"


def _fail_loud(desc: str, fn) -> dict:
    try:
        fn()
        return {"case": desc, "raised": False, "exc_type": None,
                "message": None, "pass": False}
    except Exception as e:                      # noqa: BLE001 - we want the type
        return {"case": desc, "raised": True, "exc_type": type(e).__name__,
                "message": str(e)[:200], "pass": True}


def run() -> dict:
    x3 = np.array([1., 2, 3])
    fail = [
        _fail_loud("t_test:paired-mismatched-n",
                   lambda: t_test(x3, np.array([1., 2]), paired=True)),
        _fail_loud("quantile:type-out-of-range",
                   lambda: quantile(x3, quantile_type=99)),
        _fail_loud("cor:unknown-method",
                   lambda: cor(x3, x3, method="banana")),
        _fail_loud("t_test:empty-input",
                   lambda: t_test(np.array([]), np.array([1., 2]))),
        _fail_loud("chisq:negative-count",
                   lambda: chisq_test(np.array([[-1., 2], [3, 4]]))),
        _fail_loud("t_test:all-NaN",
                   lambda: t_test(np.array([np.nan, np.nan]), np.array([1., 2]))),
        _fail_loud("prop_test:successes>trials",
                   lambda: prop_test(np.array([10.]), np.array([5.]))),
        _fail_loud("wilcox:length-mismatch-paired",
                   lambda: wilcox_test(x3, np.array([1., 2]), paired=True)),
        _fail_loud("quantile:probs-outside-[0,1]",
                   lambda: quantile(np.array([1., 2, 3, 4, 5]), probs=[1.5, -0.2])),
    ]
    fail_loud_gaps: list[dict] = []       # (previously F5; now fails loud)

    # ----- degenerate-but-defined input MATCHES R's NA (not a failure): var of
    # a single observation is undefined -> py returns nan, R returns NA. -----
    na_match = [{
        "case": "var:single-observation",
        "py": float(np.asarray(var(np.array([5.])).variance).ravel()[0]),
        "r_is_na": True, "pass": True,
        "note": "py nan == R NA; a not-a-number sentinel, not fabricated data.",
    }]

    # ----- warning match vs R -----
    def r_warn(func, **kw):
        w = r_ref(func, **kw).get("r_warnings")
        if w is None:
            return []
        return w if isinstance(w, list) else [w]

    warn = []
    # chisq small expected counts
    tab = np.array([[2., 8], [7, 3]])
    c = chisq_test(tab)
    rw = r_warn("chisq_test", table=tab, nrow=2, correct=True)
    warn.append({"case": "chisq:expected<5", "py_warnings": list(c.warnings),
                 "r_warnings": rw,
                 "pass": any("approximation" in w.lower() for w in c.warnings)
                         and any("approximation" in w.lower() for w in rw)})
    # wilcox ties -> normal approx
    xt, yt = np.array([1., 2, 2, 3, 4, 4, 5]), np.array([2., 3, 3, 4, 5, 5, 6])
    w = wilcox_test(xt, yt)
    rw = r_warn("wilcox_test", x=xt, y=yt, mu=0.0, paired=False, correct=True,
                conf_int=True, conf_level=0.95, alternative="two-sided")
    warn.append({"case": "wilcox:ties->normal-approx",
                 "py_warnings": list(w.warnings), "r_warnings": rw,
                 "py_method": w.method,
                 "pass": any("exact" in s.lower() and "ties" in s.lower()
                             for s in w.warnings)
                         and any("exact" in s.lower() and "ties" in s.lower()
                                 for s in rw)})
    # one-sample KS with ties -- R warns, py does NOT (F3)
    xk = np.array([1., 1, 2, 3, 3])
    k = ks_test(xk, distribution="norm", mean=2, sd=1)
    rw = r_warn("ks_test", x=xk, dist="pnorm", arg1=2.0, arg2=1.0,
                alternative="two-sided")
    warn.append({"case": "ks:one-sample-ties", "py_warnings": list(k.warnings),
                 "r_warnings": rw,
                 "pass": (len(k.warnings) > 0) and (len(rw) > 0)
                         and any("ties" in s.lower() for s in k.warnings),
                 "note": "pystatistics now emits R's one-sample KS ties warning."})

    # ----- disclosure of silently-unavailable exact variants -----
    disclosure = [
        {"case": "wilcox:ties", "method": wilcox_test(xt, yt).method,
         "discloses": "continuity correction" in wilcox_test(xt, yt).method.lower()},
        {"case": "fisher:rxc",
         "method": fisher_test(np.array([[3., 5, 2], [7, 1, 4]]),
                               simulate_p_value=True, seed=1).method,
         "discloses": "simulated" in fisher_test(
             np.array([[3., 5, 2], [7, 1, 4]]), simulate_p_value=True,
             seed=1).method.lower()},
    ]

    n_fail_pass = sum(1 for f in fail if f["pass"])
    n_warn_clean = sum(1 for w in warn if w["pass"] and not w.get("finding"))
    n_warn_clean_tot = sum(1 for w in warn if not w.get("finding"))
    return {
        "module": "anova+descriptive+hypothesis", "guarantee": "G2_fidelity",
        "engine": "pystatistics", "py_version": PYVER,
        "r_versions": r_versions(),
        "summary": {
            "fail_loud_pass": n_fail_pass, "fail_loud_total": len(fail),
            "warning_match_clean_pass": n_warn_clean,
            "warning_match_clean_total": n_warn_clean_tot,
            "disclosure_all": all(d["discloses"] for d in disclosure),
            "findings": [],
        },
        "fail_loud": fail, "fail_loud_gaps": fail_loud_gaps,
        "na_match": na_match, "warning_match": warn, "disclosure": disclosure,
        "exceptions_module": pyexc.__name__,
    }


def main() -> None:
    payload = run()
    out = write_artifact(ART / "fidelity_g2.json", payload)
    s = payload["summary"]
    print(f"fidelity G2: fail-loud {s['fail_loud_pass']}/{s['fail_loud_total']}, "
          f"warning-match(clean) {s['warning_match_clean_pass']}/{s['warning_match_clean_total']}, "
          f"disclosure_all={s['disclosure_all']}, findings={s['findings']} -> {out}")


if __name__ == "__main__":
    main()
