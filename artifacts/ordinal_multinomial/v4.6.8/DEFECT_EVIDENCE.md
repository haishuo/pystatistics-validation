# 4.6.8 — superseded (CF-1 showstopper), retained as defect evidence only

pystatistics 4.6.8 is **dead on arrival** for the GPU (float32) path of `polr` and
`multinom`: the model information matrix was inverted in single precision, silently
returning wrong / **negative-variance** standard errors with `converged=True` on
ill-conditioned designs (RIGOR R16 showstopper). These `runs/cf1_*.json` files are
the CUDA/MPS proof of the defect; they are **not** a validation of 4.6.8 and there
is no 4.6.8 report. Fixed in **4.6.9** — see `reports/ordinal-multinomial-v4.6.9.md`
and `artifacts/ordinal_multinomial/v4.6.9/runs/cf1_summary.csv`.
