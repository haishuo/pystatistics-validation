"""G1 correctness — predict() on held-out data vs R's predict.polr / predict.multinom.

Both models fit on a training split and predict on a held-out split, for
type='probs' (the class-probability matrix) and type='class' (the argmax class).
The contract is TIGHT: given coefficients that already agree at the optimizer
tier, the predicted probabilities must reproduce R's predict() to the tight tier,
and the predicted classes must match exactly.

Emits artifacts/ordinal_multinomial/v<ver>/runs/predict.json.
Run: DATASETS_ROOT=Dev/datasets python drivers/ordinal_multinomial/run_predict.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omdata  # noqa: E402
from omref import (r_predict_polr, r_predict_multinom,  # noqa: E402
                   reorder_fitted, r_package_versions)
from omcompare import arr_cmp, within, to_native  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run  # noqa: E402
# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402
from pystatistics.ordinal import polr  # noqa: E402
from pystatistics.multinomial import multinom  # noqa: E402

_ART = (Path(__file__).resolve().parents[2]
        / "artifacts/ordinal_multinomial/v{ver}/runs/predict.json")


# Shared statistical-link label ("logistic") drives R's predict.polr method= and
# the JSON label; pystatistics 5.0 renamed that link's value to "logit". Translate
# only at the pystatistics call site (see run_g1_ordinal for the same decoupling).
_PYLINK = {"logistic": "logit", "probit": "probit", "cloglog": "cloglog"}


def run_polr() -> list[dict]:
    recs = []
    for link in ("logistic", "probit", "cloglog"):
        d = omdata.synth_ordinal(n=800)
        tr, te = slice(0, 600), slice(600, 800)
        sol = polr(d.y[tr], d.X[tr], link=_PYLINK[link])
        pp = sol.predict(d.X[te], kind="probs")
        pc = sol.predict(d.X[te], kind="class")
        r = r_predict_polr(d.y[tr], d.X[tr], d.X[te], link)
        probs = arr_cmp(pp, np.array(r["probs"]))
        cls_match = int(np.sum(pc == np.array(r["cls"])))
        recs.append({"group": "polr_predict", "link": link, "n_test": len(pc),
                     "probs": probs, "class_match": cls_match,
                     "class_total": len(pc),
                     "pass": bool(within(probs, abs_tol=1e-3)
                                  and cls_match == len(pc))})
    return recs


def run_multinom() -> list[dict]:
    recs = []
    for key, des, split in [
        ("multinom_synth", omdata.load_multinom_synth(), 1100),
        ("synth_K5", omdata.synth_multinom(n=2000, K=5, p=4), 1500),
    ]:
        tr, te = slice(0, split), slice(split, des.y.shape[0])
        sol = multinom(des.y[tr], des.X[tr], max_iter=2000)
        pp = sol.predict(des.X[te], kind="probs")
        pc = sol.predict(des.X[te], kind="class")
        r = r_predict_multinom(des.y[tr], des.X[tr][:, 1:], des.X[te][:, 1:],
                               des.r_levels)
        rprobs = reorder_fitted(r["probs"], r["cols"], des.n_classes)
        probs = arr_cmp(pp, rprobs)
        cls_match = int(np.sum(pc == np.array(r["cls"])))
        recs.append({"group": "multinom_predict", "dataset": key,
                     "n_test": len(pc), "K": des.n_classes, "probs": probs,
                     "class_match": cls_match, "class_total": len(pc),
                     "pass": bool(within(probs, abs_tol=1e-3)
                                  and cls_match == len(pc))})
    return recs


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions()

    polr_r = run_polr()
    mn_r = run_multinom()
    run = build_run(
        env=env,
        config={"suite": "ordinal-multinomial-predict",
                "reference": "R predict.polr / predict.multinom",
                "contract": "predicted probs tight vs R; predicted class exact"},
        records=to_native([{"key": "polr", "checks": polr_r},
                           {"key": "multinom", "checks": mn_r}]),
    )
    out = Path(str(_ART).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    for name, grp in [("polr", polr_r), ("multinom", mn_r)]:
        print(f"  {name:9s} {sum(bool(r['pass']) for r in grp)}/{len(grp)} pass")


if __name__ == "__main__":
    main()
