# mvnmle v6.1.5 — canonical evidence (redistributable problem set)

Generated 2026-08-26/27 against `pip install pystatistics==6.1.5` (PyPI).
Problems: simulated survey profiles `simw`/`simg` (seeds 20260826/27,
`drivers/_shared/mvn_sim.py`) + the `nhanes_cardio` extract (NHANES 2017-2018,
public domain). Machines: Apple M2 Max (MPS + CPU) and Forge (AMD Ryzen 5
7600X + NVIDIA RTX 5070 Ti, CUDA 12.8, torch 2.11 nightly; R 4.3.3,
mvnmle 0.1-11.2). Artifacts: `artifacts/mvnmle/v6.1.5/runs/`.

Headlines (details in the paper repo's `results/VENDOR.md` map):

- **CPU vs R** (single machine): relative |dloglik| 1e-11 (p=5) to 6e-6
  (p=25, R at its iteration cap); speedups 122-360x (p<=20), 145x at p=25
  (R: 2,238 s under a 4-h budget; CPU/FP64: 15.5 s). R p=50: degenerate exit
  (loglik -0.0, 0 iter), reproduced on both machines. R p>50: refuses.
- **Cross-backend GPU agreement**: relative <= 3e-7 on every end-to-end cell,
  identical iteration counts per cell; all fits converged. MPS faster in
  every cell (up to 2.5x at simg p=100). CUDA repeats bit-identical; MPS
  repeat spread <= 4e-7 relative loglik, 1-6% wall.
- **FP32 vs FP64** (identical problems): deficits 8-9e-5 relative (p=50) to
  2-8e-4 (p=100); Sigma error 0.6-0.8% Frobenius. Synthetic sweep: flat to
  kappa ~3e4, then 1e-3 (1e5) -> 9e-2 (1e6).
- **Attribution (MPS)**: batching 2.8-4.6x (obj) / 2.6-4.4x (grad); matmul
  inverse 33x on the objective at simw p=50 (7,691.72 -> 232.72 ms);
  closed-form gradient 25x atop it (6,747.09 -> 272.66 ms). CUDA: solve
  preferred (blocked 0.4-0.7x), analytic neutral-favorable.
- **Trace microbenchmark**: MPS solve/blocked 68-144x across v=25-100; naive
  MPS-vs-CUDA solve gap ~840x = ~125x software x ~7x hardware at v=50.
- **Cross-machine determinism**: sim problems regenerate byte-identically
  (n and pattern counts equal across numpy builds on both machines).
