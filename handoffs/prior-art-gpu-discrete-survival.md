# Prior-art trawl — "first discrete-time survival on GPU at scale" (A8)

**Purpose.** Pre-publication due diligence owed under RIGOR R13: a novelty claim
earns extra scrutiny plus a prior-art trawl. This document is the trawl for the
pystatistics claim that its discrete-time survival path is the *first classical
discrete-time survival estimator validated on GPU at scale*.

**Run:** 2026-07-11, four-angle parallel web sweep + adversarial synthesis
(workflow `prior-art-gpu-discrete-survival`, 5 agents).

---

## VERDICT (synthesis)


# Prior-art assessment: GPU classical discrete-time survival novelty claim

## 1. Verdict: NEEDS NARROWING

The claim is **not defensible as literally worded**, but a tightly-scoped version is. The problem is the phrase "person-period / pooled logistic regression ... on GPU at scale." That *exact likelihood* — the person-period discrete-time logistic/cloglog hazard — is already run on GPU by default in pycox (Logistic-Hazard) and Keras Nnet-survival. pycox itself states that Logistic-Hazard = Partial Logistic Regression (Efron 1988) = Nnet-survival, i.e. the estimator is classical statistics, not a new model. So a reviewer who reads only the noun phrase can point at a PyTorch library that has fit that likelihood on CUDA since ~2019 and call the claim anticipated.

What rescues it is the intersection the prior art never occupies simultaneously: **exact MLE (IRLS/Newton to convergence) + full R-equivalent inference (coefficients, SEs, Wald/LR, deviance) + a validated fp32 no-silent-wrong acceptance gate.** That intersection is genuinely unclaimed. But the load-bearing words must be *in the claim*, and "first" must be scoped to that intersection — not left to modify "person-period logistic on GPU," which is preempted.

Two independent bodies of prior art bracket the claim, and the defensible position lives strictly between them:
- **Suchard/OHDSI Cyclops line** — classical, exact, R-parity, GPU, benchmarked at 1e6 (434,866 patients × 9,811 covariates, 35–70× speedup). This defeats any generic "first classical survival estimator on GPU at scale with R-parity." It survives only because Cyclops is **continuous-time Cox/Fine-Gray partial likelihood**, a different likelihood on un-expanded risk-set data — not person-period pooled logistic. "Discrete-time person-period" is therefore load-bearing.
- **Neural discrete-time (Nnet-survival, pycox Logistic-Hazard, DeepHit)** — the *same* person-period likelihood, GPU-native, tested to 1e6 observations. This defeats any generic "first person-period discrete-time survival on GPU." It survives only because those are **SGD-trained, prediction-only, not R-validated, no inference**. "Classical / exact-MLE / R-equivalent inference" is therefore load-bearing.

Drop either qualifier set and the claim is anticipated. Keep both and it holds.

## 2. Closest prior art: Gensheimer & Narasimhan 2019 (Nnet-survival) — with pycox as the sharper GPU threat

**Gensheimer & Narasimhan, "A scalable discrete-time survival model for neural networks," PeerJ 2019** is the closest on *method*. Its underlying estimator is explicitly the textbook person-period discrete-time likelihood (it cites Cox & Oakes 1984, Singer & Willett 1993; reduces to Kaplan-Meier in the null case; the no-hidden-layer "flexible" version is literally per-interval logistic regression, and the "proportional hazards" version is the cloglog discrete-time model). So the crux is exactly "neural vs classical" — and the honest answer is that Nnet-survival *is* the classical likelihood wearing a neural-net implementation.

**Why it does not preempt the narrowed claim:**
1. It fits by **mini-batch SGD**, not exact MLE to convergence. "Scalable" in that paper means out-of-core/mini-batch, explicitly *not* GPU — its own 1,000→1,000,000 benchmark was run **single-core CPU** ("constrained to run on one CPU core").
2. It emits **predictions only**, evaluated by C-index/Brier. No coefficients, SEs, Wald/LR tests, deviance.
3. It is **never validated to parameter-level R-equivalence**, and has **no fp32/fp64 correctness gate** — a neural approximate optimizer has no reason to care about fp32 silent-wrongness.

**The sharper threat is pycox, not Gensheimer.** Gensheimer's benchmark was CPU, so it does not itself establish "on GPU." pycox's Logistic-Hazard is the same likelihood *and* is GPU-executed by default via PyTorch. That is the artifact a reviewer will actually cite to say "this estimator has been run on GPU already." The rebuttal is the same three axes — SGD not exact MLE, prediction-only not inferential, not R-validated, no fp32 gate — but you must name pycox explicitly and pre-empt it, or the reviewer will.

One more honest caveat: **"~1e6 person-period rows" is not an impressive scale number and should not be the novelty lever.** Cyclops fit 434,866 *patients* × ~10⁴ covariates; 1e6 person-period rows may be only ~10⁵ patients × 10 intervals. Lead with the estimator class + validation + fp32 gate, not with scale.

## 3. Revised, defensible claim wording

> "We present the first **GPU-accelerated exact-maximum-likelihood** classical discrete-time survival estimator (person-period / pooled logistic and complementary-log-log hazard), fit by IRLS/Newton to convergence and **validated to parameter-level equivalence with R** (`glm` on person-period data), including full inferential output (coefficients, standard errors, Wald/LR tests, deviance). Unlike neural discrete-time survival models (Nnet-survival, pycox Logistic-Hazard), which parametrize the same hazard likelihood with a network and fit it approximately by mini-batch SGD without R-equivalent inference, and unlike GPU-accelerated continuous-time Cox/Fine-Gray partial-likelihood methods (Cyclops/OHDSI), our estimator reproduces the exact classical person-period fit on GPU and ships a proven fp32 no-silent-wrong acceptance gate that certifies GPU single-precision results against a fp64/R reference before they are returned."

Key moves: (a) "first" now modifies **GPU-accelerated exact-MLE classical person-period estimator with R-equivalent inference**, not "person-period logistic on GPU"; (b) both bracketing literatures are named and distinguished in the sentence itself; (c) the fp32 gate — the single cleanest differentiator, which *no* prior artifact has — is foregrounded; (d) scale is deliberately dropped from the novelty clause. If you want to keep a scale claim, phrase it as a demonstration ("demonstrated at ~10⁶ person-period rows"), not as part of "first."

## 4. Citations a reviewer will expect you to have addressed

1. **Yang, Schuemie, Ji & Suchard (2024), *JCGS* 33(1)** — "Massive Parallelization of Massive Sample-Size Survival Analysis" (Cyclops GPU Cox/Fine-Gray, 35–70× at 1e6). The nearest GPU classical-survival-at-scale prior art; you must distinguish continuous-time partial likelihood from person-period pooled logistic. (Companion: **arXiv 2310.16238**, GPU stratified/time-varying Cox.)
2. **Gensheimer & Narasimhan (2019), *PeerJ* 7:e6257** — Nnet-survival; the discrete-time person-period likelihood in the ML literature. The core "neural vs classical" rebuttal.
3. **Kvamme, Borgan & Scheel — pycox / Logistic-Hazard** — the GPU-executed version of the exact same likelihood, and the artifact that most directly threatens the "on GPU" clause. Cite and pre-empt explicitly.
4. **Efron (1988), *JASA* 83:414** — "Logistic Regression, Survival Analysis, and the Kaplan-Meier Curve" (Partial Logistic Regression). Establishes that the estimator is *classical statistics of long standing*, so your novelty is the GPU/exact-MLE/validation/fp32 layer, not the model. Citing this pre-empts the "you didn't invent this model" objection by conceding it up front.
5. **Suchard/Cyclops software (OHDSI) or Mittal & Madigan (2014), *Biostatistics*** — the CPU CCD-for-Cox foundation; shows you know the massive-sample-survival lineage and are not reinventing it. (Optionally add **survivalGPU** / Feydy et al. as a second, self-described proof-of-concept GPU-Cox artifact.)

Bottom line for the paper: keep the four load-bearing qualifiers — **classical / exact-MLE / R-equivalent-inference / fp32-no-silent-wrong** — attach "first" only to their intersection, name pycox and Cyclops in the claim itself, and do not lean on "~1e6 rows" as the novel part. So scoped, the claim is honest and survives an adversarial review; stated broadly, it is anticipated by pycox Logistic-Hazard (GPU) and Cyclops (classical GPU survival at larger scale).


---

# Appendix — raw findings per search angle


## Angle: discrete-time-gpu

I have enough to synthesize a precise report. Here are my findings.

---

# Prior art: GPU-accelerated discrete-time survival (person-period / pooled logistic)

## Bottom line

I found **no paper or project whose named contribution is GPU-accelerating the person-period / pooled-logistic-regression pipeline for discrete-time survival, with a reported speedup benchmarked at person-period-row scale.** The literature splits cleanly into three adjacent-but-distinct buckets. The closest large-scale GPU survival work (Suchard/Yang, Cyclops) accelerates the *continuous-time Cox partial likelihood* via coordinate descent — a different likelihood on a different data structure, **not** pooled logistic on expanded rows. The discrete-time person-period models that do exist (nnet-survival and the deep-learning family) run on GPU-capable frameworks incidentally, but none present a GPU-acceleration study of the pooled-logistic likelihood itself. This is a genuine gap.

---

## (c) Continuous-time Cox on GPU — the closest large-scale work, but NOT pooled logistic

**1. Yang, Schuemie & Suchard — "Massive Parallelization of Massive Sample-Size Survival Analysis"**
- Venue/year: *Journal of Computational and Graphical Statistics*, 2023 (Taylor & Francis, doi 10.1080/10618600.2023.2213279). URL: https://www.tandfonline.com/doi/full/10.1080/10618600.2023.2213279
- What is GPU-accelerated: the **Cox partial likelihood and its gradient**, plus the **Fine-Gray** competing-risks model, optimized by **cyclic coordinate descent**, using a "single-pass parallel scan algorithm" mapped onto NVIDIA's CUB library. Implemented in the R package **Cyclops**.
- Scale / speedup (quoting the search-surfaced abstract text): GPU "reducing the fitting time for analyses containing **one million patients from nearly one day to just one to two hours**" — "an order of magnitude compared to a similarly optimized CPU implementation."
- Crucial distinction: this fits the **risk-set partial likelihood directly**; it does **not** expand data into person-period rows and does **not** use a logistic/pooled likelihood. So it is category (c), not (a).

**2. Yang, Schuemie & Suchard — "Efficient GPU-accelerated fitting of observational health-scaled stratified and time-varying Cox models"**
- Venue/year: arXiv:2310.16238 (stat.CO), submitted Oct 2023. URL: https://arxiv.org/abs/2310.16238
- What is GPU-accelerated: **stratified Cox, Cox with time-varying covariates, and Cox with time-varying coefficients**, by recasting all three into a stratified-Cox segmented partial likelihood and converting the bottleneck "into an un-segmented operation to leverage the efficient many-core parallel scan algorithm." ~order-of-magnitude speedup.
- Discrete-time relevance (the one real touchpoint): it states "The Cox model with time-varying coefficients can be **transformed into the Cox model with time-varying covariates when using discrete time-to-event data**." This is a *covariate-construction trick within the Cox partial likelihood* — **not** a person-period pooled-logistic reformulation. GPU work is still on the Cox scan.

Both are the same Suchard/OHDSI line; **Cyclops** (https://ohdsi.github.io/Cyclops/) is the software vehicle. Note the Cyclops docs themselves don't advertise GPU — the GPU path is the research add-on for Cox/Fine-Gray.

---

## (a) Discrete-time / person-period survival specifically — exists, but GPU is incidental, no acceleration study

**3. Gensheimer & Narasimhan — "A scalable discrete-time survival model for neural networks" (nnet-survival)**
- Venue/year: *PeerJ*, 2019. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6348952/
- This is the canonical discrete-time person-period model in ML: "Follow-up time is divided into n intervals"; flexible version uses a **sigmoid activation** where "log odds are converted to the conditional probability of surviving this interval" (i.e., a logistic/pooled-logistic discrete hazard); trained "using the maximum likelihood method using **mini-batch stochastic gradient descent (SGD)**."
- Scale: tested on **simulated datasets up to 1,000,000 observations** plus SUPPORT (9,105 patients).
- GPU: **none claimed.** The paper reports running on "Ubuntu Linux server with 3.6 GHz Intel Xeon E5-1650 CPUs"; implemented in Keras/TensorFlow (GPU-capable) but **GPU is neither used nor benchmarked**. So: discrete-time person-period logistic-hazard, "scalable" via mini-batch SGD on CPU — **not** a GPU contribution.

**4. Deep discrete-time survival family (DeepHit, DRSA, DCS, and applications)**
- These parametrize a discrete per-interval hazard with a neural net, negative-log-likelihood loss, mini-batch SGD, in GPU-capable frameworks. Examples: ICU deep-learning discrete-time model (Intensive Care Med Exp, 2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC9474816/); credit-risk age-period-cohort discrete-time NN (Risks, 2024, https://www.mdpi.com/2227-9091/12/2/31).
- GPU status: standard neural-net training on GPU — **incidental**, not a documented speedup study of the person-period logistic likelihood versus CPU. None isolate or benchmark "GPU vs CPU for the pooled-logistic hazard fit."

**5. TorchSurv (Novartis / FDA)** — https://github.com/Novartis/torchsurv
- PyTorch survival library offering discrete-time and Cox losses; runs on GPU like any torch model. It is a **library wrapper**, not a benchmarking study; no reported GPU speedup numbers for pooled-logistic discrete-time fitting.

**6. Federated discrete-time Cox (Andreux et al.)** — arXiv:2006.08997, "Federated Survival Analysis with Discrete-Time Cox Models"
- Uses the **discrete-time / person-period reformulation precisely because it makes the loss separable** ("approximating the Cox loss with a discrete-time model … the loss for this model is separable, it can be easily plugged in existing federated learning algorithms"). But the **contribution is federated learning / privacy, not GPU acceleration**; GPU is not the reported axis.

---

## (b) Generic logistic regression on GPU — well established, not survival-specific

- **Cyclops** (OHDSI) does cyclic coordinate descent for **logistic, Poisson and Cox**; the GPU add-on (Suchard) targets Cox/Fine-Gray, but large-scale regularized logistic regression on GPU is squarely in scope.
- Generic examples exist, e.g. "GPU acceleration of logistic regression with CUDA" (researchgate). These accelerate a plain logistic GLM; **none frame the design matrix as a person-period survival expansion or address the discrete-hazard likelihood structure.**

---

## Interpretation for the task

- The **discrete-time = pooled/person-period logistic** equivalence is textbook and universally cited (Springer BMC 2022 tutorial https://link.springer.com/article/10.1186/s12874-022-01679-6; Framingham comparison https://link.springer.com/article/10.1186/s12874-016-0248-6), but always as a *statistical* device, framed for CPU/standard GLM software.
- **Every** GPU survival result at scale that I can find accelerates the **Cox partial likelihood** (continuous-time, risk-set, coordinate descent), not the pooled-logistic expansion. The two are numerically distinct fits.
- The **only** GPU-adjacent discrete-time person-period work (nnet-survival, DeepHit family, TorchSurv) reaches GPU purely via generic neural-net mini-batch SGD, and **none report a GPU-vs-CPU acceleration benchmark for the person-period logistic likelihood as such.**

Net: a GPU-accelerated **pooled-logistic / person-period discrete-time survival fit with a documented speedup at person-period-row scale** appears to be unclaimed prior art. If pystatistics builds this, the honest positioning is: distinct from the Suchard/Cyclops GPU-Cox line (different likelihood + data structure), and going beyond the incidental-GPU status of the deep-learning discrete-time models by actually benchmarking the GLM-style pooled-logistic hazard fit on GPU.

Sources:
- https://www.tandfonline.com/doi/full/10.1080/10618600.2023.2213279 (JCGS 2023, GPU Cox/Fine-Gray CCD)
- https://arxiv.org/abs/2310.16238 (arXiv 2023, GPU stratified/time-varying Cox)
- https://ohdsi.github.io/Cyclops/ (Cyclops software)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6348952/ (nnet-survival, PeerJ 2019)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9474816/ (ICU discrete-time deep learning, 2022)
- https://www.mdpi.com/2227-9091/12/2/31 (discrete-time NN credit risk, Risks 2024)
- https://github.com/Novartis/torchsurv (TorchSurv)
- https://arxiv.org/abs/2006.08997 (federated discrete-time Cox, 2020)
- https://link.springer.com/article/10.1186/s12874-022-01679-6 (discrete-time modeling tutorial, 2022)


## Angle: cox-gpu

I have enough to compile the report. Here are the findings.

---

# Prior Art: GPU-Accelerated Survival Regression at Scale

## Bottom line

Yes — GPU-accelerated **classical** survival regression at ~1e6 scale has been published, but by essentially **one research group (Suchard/OHDSI, UCLA)** and shipped in **one library (R package `Cyclops`)**. It is Cox (incl. stratified, time-varying, and Fine-Gray competing-risks), fit by cyclic coordinate descent (CCD), with the GPU doing a **segmented/parallel-scan** of the partial-likelihood gradient and Hessian. The much larger literature of "survival on GPU" is **neural-net survival models** (DeepSurv, Nnet-survival, pycox, DeepHit), where the GPU is used trivially the way any deep net uses it (minibatch SGD over an MLP) — that is *not* classical-estimator acceleration and is not R-parity-relevant.

---

## Tier 1 — Classical Cox/survival estimator on GPU at 1e6 scale (the real prior art)

### 1. Yang, Schuemie, Ji, Suchard — "Massive Parallelization of Massive Sample-size Survival Analysis" (2024, JCGS 33(1); arXiv 2204.08183, May 2023)
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11070748/ · arXiv https://arxiv.org/abs/2204.08183 · https://www.tandfonline.com/doi/full/10.1080/10618600.2023.2213279
- **What runs on GPU:** the CCD inner loop's bottleneck — the segmented cumulative sums (risk-set aggregations) needed for the Cox partial-likelihood gradient and Hessian. They recast the segmented scan as an **un-segmented single-pass parallel scan**: "three forward scans and three backward scans" returning the prefix sums `Spre[exp(Xβ)]`, `Spre[exp(Xβ)×Xj]`, etc. A **forward-backward** parallel scan handles the Fine-Gray competing-risks model.
- **Scale:** simulations **up to 1,000,000 samples**; real study = **434,866 hypertension patients × 9,811 covariates** (sparse).
- **Hardware / speedups (quoted):** NVIDIA **Quadro GV100** (5120 CUDA cores, 32 GB) vs 10-core Xeon W-2155 → "**35× faster for this Cox model and 39× faster for this Fine-Gray model**" at 1M samples. NVIDIA **A100** (6912 cores, 80 GB) vs Xeon Silver 4214 → "**up to a 52-fold speedup … Cox … and a 70-fold speedup … Fine-Gray**." Headline range: **35–70× at up to 1M samples.**
- **Real-world timing (quoted):** regularized Cox on the 434,866-patient set dropped from CPU "**almost two days**" to GPU "**3.87 hours**"; Fine-Gray from "**more than three days**" to "**8.57 hours**."
- **Library:** open-source **R package `Cyclops`** (OHDSI).

### 2. (Same group, follow-on) "Efficient GPU-accelerated fitting of observational health-scaled stratified and time-varying Cox models" (arXiv 2310.16238, Oct 2023)
- URL: https://arxiv.org/abs/2310.16238
- **What runs on GPU:** extends the parallel-scan approach to the **stratified Cox model, Cox with time-varying covariates, and Cox with time-varying coefficients**, converting the "segmented partial likelihood and its gradient" into "an un-segmented operation to leverage the efficient many-core parallel scan algorithm."
- **Scale / speedup (quoted from abstract):** "**an order of magnitude speedup**" vs optimized CPU; reduces fitting for analyses with **one million patients "from nearly one day to just one to two hours."** (Full-text hardware table was not retrievable — abstract-level numbers only.)
- **Library:** same `Cyclops` lineage.

### 3. OHDSI symposium abstract — "GPU parallelization of massive sample-size survival analysis" (2021)
- URL: https://www.ohdsi.org/wp-content/uploads/2021/08/72-Abstract-GPU-parallelization-of-massive-sample-size-survival-analysis.pdf
- The conference precursor to #1 (couldn't extract numbers — image-flattened PDF). Same authors/approach.

### 4. Mittal, Madigan, et al. — "High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis" (2014, *Biostatistics*)
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3944969/
- The **CPU** foundation this all builds on: CCD for Cox at "millions of observations and millions of variables," exploiting data sparsity. This is what became `Cyclops`; the 2023–24 papers are the GPU port. Establishes the CCD-for-Cox lineage but is **not itself GPU**.

**Note on RAPIDS/cuML:** despite the obvious search intent, **cuML/RAPIDS has no Cox or survival regression.** cuML's 50+ algorithms mirror scikit-learn (linear/logistic, RF, kNN, clustering, etc.); survival analysis is absent from the current (26.06) docs. So the only classical-estimator GPU survival prior art is the Cyclops line above. (xgbse / XGBoost-survival is GPU-capable but it is a **tree-boosting** survival model, not a Cox/discrete-time MLE estimator.)

---

## Tier 2 — Neural-net survival models (GPU is trivial; NOT classical acceleration)

These use the GPU the way any deep net does — batched tensor ops over an MLP, minibatch SGD. They do **not** accelerate a classical estimator toward R-parity; they *replace* the linear predictor with a network and change the estimand (nonlinear risk).

### 5. Katzman et al. — **DeepSurv** (2018, *BMC Med Res Methodol*)
- URL: https://github.com/jaredleekatzman/DeepSurv
- Cox partial-likelihood loss with the linear term replaced by a network: `h(t|X)=h₀(t)·exp{f_θ(X)}`. Runs on CPU or GPU (nvidia-docker). **Its loss still needs the full risk set per step** — the very scaling limitation Tier 1 solves and Tier 3 avoids.

### 6. Gensheimer & Narasimhan — **Nnet-survival** ("A scalable discrete-time survival model for neural networks," 2019, *PeerJ*)
- URL: https://arxiv.org/abs/1805.00917 · https://pmc.ncbi.nlm.nih.gov/articles/PMC6348952/
- **Discrete-time** hazard model (per-interval logistic/hazard), Keras/TensorFlow. Key scaling property (quoted): the "loss function depends only on the information contained in the current mini-batch, which enables rapid training with mini-batch SGD and application to arbitrary-size datasets." **Scalability tested on simulated datasets "ranging from 1,000 to 1,000,000 patients."** This is the closest neural-side analogue to a discrete-time survival estimator at 1e6 — but it's an SGD-trained network, not a full-likelihood MLE matched to an R routine.

### 7. Kvamme et al. — **pycox** (DeepSurv / **Cox-Time** / **DeepHit**), PyTorch
- URL: https://github.com/havakv/pycox
- The standard PyTorch survival-net toolbox. Cox-Time = non-proportional neural Cox; DeepHit = discrete-time competing risks. GPU via PyTorch. Same caveat: neural models, not classical estimators.

### 8. Others in passing: SODEN (ODE-net survival, arXiv 2008.08637), tdCoxSNN (time-dependent Cox neural net, 2307.05881), GNN-surv, Case-Base neural nets — all neural, GPU-trivial.

---

## The distinction that matters for pystatistics

- **Classical estimator GPU acceleration** (what pystatistics does, R-parity): the *only* published prior art at ~1e6 is the **Suchard/OHDSI `Cyclops`** work — Cox (standard, stratified, time-varying) and Fine-Gray via **CCD + parallel-scan gradient/Hessian on GPU**, **35–70× at up to 1M samples**, real 435k×9.8k study in ~4–8.5 h vs ~2–3 days CPU. It is R-only (`Cyclops`); there is **no Python classical-Cox GPU implementation** and **nothing in cuML/RAPIDS**.
- **Neural survival GPU** (DeepSurv, Nnet-survival, pycox/DeepHit): GPU is incidental (it's just a deep net). They scale to 1e6 via **minibatch SGD**, but they change the model/estimand and are not aiming for exact agreement with `survival::coxph`. Nnet-survival explicitly benchmarks discrete-time survival to 1,000,000 patients — but as an SGD-trained network, not a full-data MLE.

**Gap / positioning:** a Python, GPU-accelerated **classical** Cox / discrete-time survival estimator that reproduces R (`survival`) results at ~1e6 scale is essentially **unoccupied territory** — the exact-MLE + R-parity niche sits between the R-only Cyclops CCD work and the Python neural-net crowd. The scan-based gradient/Hessian trick from Yang et al. (2024) is the canonical technique to cite/borrow for the continuous-time Cox partial likelihood; discrete-time survival maps onto batched logistic-hazard evaluations (the Nnet-survival formulation) which is embarrassingly parallel and the natural GPU target.

Key URLs: Cyclops repo https://github.com/OHDSI/Cyclops · Yang 2024 https://pmc.ncbi.nlm.nih.gov/articles/PMC11070748/ · stratified/time-varying https://arxiv.org/abs/2310.16238 · Mittal 2014 CPU foundation https://pmc.ncbi.nlm.nih.gov/articles/PMC3944969/ · Nnet-survival https://arxiv.org/abs/1805.00917 · pycox https://github.com/havakv/pycox


## Angle: frameworks-libs

I have a complete picture of the landscape. Here is the synthesis.

## Competitive landscape: GPU backends for classical (non-neural) survival estimators

### Bottom line
Among established survival libraries, **none of the mainstream classical toolkits (lifelines, scikit-survival, statsmodels, R `survival`, H2O) offer any GPU acceleration**. GPU survival support exists in only two forms: (1) **deep/neural** packages that run on PyTorch (pycox, torchsurv, torchlife — GPU is for the neural net, not a classical MLE estimator matching R), and (2) a small body of **continuous-time Cox/Fine-Gray partial-likelihood** GPU work (survivalGPU; the Suchard/OHDSI line). **No library or research artifact ships a GPU-accelerated classical discrete-time (person-period logistic / complementary-log-log hazard) estimator.** That specific niche is essentially unoccupied.

### CPU-only classical libraries (no GPU)
- **lifelines** — pure Python, CPU only. TorchSurv's library comparison table (arXiv 2404.10761) marks lifelines: GPU ✗, neural-net ✗, "Computes on CPU." PyPI/docs describe it as a pure-Python implementation. https://github.com/CamDavidsonPilon/lifelines , https://arxiv.org/pdf/2404.10761
- **scikit-survival** — CPU only. Same TorchSurv table: GPU ✗, NN ✗, "Computes on CPU." Built on scikit-learn/NumPy, which has no native GPU path for these estimators. https://github.com/sebp/scikit-survival
- **statsmodels** (`PHReg`, etc.) — no GPU. Tracked in issue #6439 ("GPU support? Or not for this project"), closed with no GPU adoption; statsmodels remains NumPy/SciPy CPU. https://github.com/statsmodels/statsmodels/issues/6439
- **R `survival`** (Therneau) — no GPU. CRAN description lists "Kaplan-Meier and Aalen-Johansen (multi-state) curves, Cox models, and parametric accelerated failure time models" with no GPU support. https://cran.r-project.org/package=survival
- **H2O** (`h2o.coxph`) — Efron/Breslow Cox on H2O's distributed **JVM/CPU** cluster; scale-out is multi-core/multi-node, not GPU. https://docs.h2o.ai (h2o.coxph reference)
- **SurvivalEVAL** — not an estimator; it is an **evaluation-metrics** package (Brier, D-Cal, concordance) wrapping lifelines/pycox/scikit-survival outputs. Irrelevant as a compute competitor. https://ojs.aaai.org/index.php/AAAI-SS/article/download/27713/27486/31764

### Neural / deep libraries (GPU, but via PyTorch for neural models — not classical MLE)
Per the TorchSurv comparison table (arXiv 2404.10761):
- **pycox** — PyTorch, GPU ✓, neural ✓ ("Computes on GPU"). Includes discrete-time neural models (Logistic-Hazard/Nnet-survival, DeepHit) but they are neural nets trained by SGD, not classical person-period MLE. https://github.com/havakv/pycox
- **torchlife** — PyTorch, GPU ✓, neural ✓ ("Computes on GPU").
- **torchsurv** (Novartis) — 100% PyTorch backend, GPU-capable; provides classical `kaplan_meier`/`ipcw` *stats helpers* but positioned for deep survival (custom NN defining model parameters), not a scaled classical estimator. https://opensource.nibr.com/torchsurv/
- **auton-survival** — neural ✓ but table marks GPU ✗, "Computes on CPU."
- **deepsurv** — neural ✓, table marks "Computes on CPU."

### GPU classical estimators — the actual prior art (all continuous-time)
- **survivalGPU** (Feydy, Van Straaten, Jannot; R + Python) — the closest direct competitor. GPU-accelerated **CoxPH and WCE (weighted cumulative exposure)**, motivated by high-throughput bootstrap on French national health data (SNDS) for pharmacovigilance. It is **continuous-time Cox partial likelihood**, not a discrete-time hazard estimator. Explicitly self-described as immature: "this package is still little more than a proof of concept … our solver has not yet been tested thoroughly." https://github.com/jeanfeydy/survivalGPU
- **Suchard / OHDSI line (Cyclops ecosystem)** — the most serious "massive-sample Cox on GPU" research:
  - "Efficient GPU-accelerated fitting of observational health-scaled stratified and time-varying Cox models," arXiv 2310.16238 — GPU parallel-scan for **stratified Cox, time-varying-covariate, and time-varying-coefficient** Cox; claims "an order of magnitude speedup over … CPU-based implementations." Uses discrete time-to-event data only to *reduce* time-varying-coefficient models to time-varying-covariate Cox — still partial-likelihood Cox, not a discrete-hazard MLE. https://arxiv.org/abs/2310.16238
  - "Massive Parallelization of Massive Sample-Size Survival Analysis," JCGS 2023 (Ko, Suchard et al.) — single-pass parallel-scan GPU algorithms for **Cox PH and Fine-Gray** competing-risks; "orders of magnitude" over multi-core CPU. https://www.tandfonline.com/doi/full/10.1080/10618600.2023.2213279

### NVIDIA RAPIDS / cuML
- **cuML has no survival analysis at all.** The stable API reference lists regression/classification, clustering, dimensionality reduction, time series (ARIMA/Holt-Winters only), neighbors, preprocessing, explainability, covariance, ensembles — **no Cox, no Kaplan-Meier, no time-to-event / censored-data estimator**. No "survival" GPU kernels exist in RAPIDS. https://docs.rapids.ai/api/cuml/stable/api/

### Implication for "classical discrete-time survival estimator on GPU at scale"
The niche is open. GPU survival work splits cleanly into **neural** (pycox/torchsurv/torchlife — SGD-trained, not R-equivalent MLE) and **continuous-time partial-likelihood Cox/Fine-Gray** (survivalGPU, Suchard/OHDSI/Cyclops). The one arguably discrete-flavored GPU option, pycox's Logistic-Hazard/Nnet-survival, parametrizes the discrete hazard with a *neural network* rather than fitting the exact person-period logistic/clog-log MLE that R (`glm` on person-period data, `discSurv`) produces. So a GPU classical discrete-time hazard estimator that reproduces R's exact person-period fit at scale has **no established competitor** — survivalGPU (continuous Cox/WCE, proof-of-concept) and the OHDSI Cox-on-GPU papers are the nearest neighbors, and cuML/lifelines/scikit-survival/statsmodels/R-survival offer nothing on GPU.


## Angle: neural-vs-classical

I have all the primary-source material needed. Here is the report.

---

## Findings: neural discrete-time survival on GPU vs. a "classical discrete-time survival estimator on GPU at scale" claim

### 1. What Gensheimer & Narasimhan (2019) / Nnet-survival actually is

**It is a neural-network method — but its likelihood *is* the classical person-period discrete-time survival model.** The paper frames it as a neural net and implements it in Keras/TensorFlow (with conv nets, MNIST demo, etc.), yet is explicit that the estimator underneath is the standard statistical discrete-time hazard likelihood:

- Title/abstract: "we describe a discrete-time survival model that is designed to be used with neural networks, which we refer to as Nnet-survival. The model is trained with the maximum likelihood method using minibatch stochastic gradient descent (SGD)."
- On the loss (pp. 2–3): "A more theoretically justified loss function, which we use in our model, would be the negative of the log likelihood function of a statistical survival model. This likelihood function has been well studied for discrete-time survival models in a non-deep learning context. Adapting eq. 3.4 from (Cox and Oakes, 1984) and eq. 2.17 from (Singer and Willett, 1993)..." — i.e. the textbook person-period likelihood.
- Reduces to Kaplan–Meier in the null case (p. 2): "in the case of a null model with no predictor variables, minimizing the loss ... results in an estimate of the hazard probabilities that equals the Kaplan-Meier maximum likelihood estimate."
- The **"flexible" version with no hidden layers is exactly per-interval logistic regression** (p. 4): "The log odds of surviving each time interval is equal to the dot product of the incoming values and the kernel weights, plus the bias weight. Then, using a sigmoid activation function, log odds are converted to the conditional probability of surviving this interval." Their simplest experiment (p. 7) literally uses "the flexible version of nnet-survival ... with no hidden layers and 39 time intervals."
- The **"proportional hazards" version is a discrete-time complementary-log-log model** (p. 5): "This version is very similar to a traditional proportional hazards discrete-time survival model using a complementary log-log link (see Rodriguez, G., 2016, section 7.5.3: 'Discrete Survival and the C-Log-Log Link')."

Kvamme & Borgan (the pycox paper) corroborate that the estimator is classical, not novel: "To the best of our knowledge, this method was first proposed by Gensheimer and Narasimhan (2019). However, if one considers the special case where ϕⱼ(x)=βᵀx, the approach is well known in the survival literature and seems to have been first addressed by Cox (1972) and Brown (1975)."

pycox's README makes the equivalence explicit: "The Logistic-Hazard method parametrize the discrete hazards and optimize the survival likelihood. **It is also called Partial Logistic Regression** [Efron 1988] **and Nnet-survival** [Gensheimer]." So "Logistic-Hazard" (pycox), "Nnet-survival" (Gensheimer), and "Partial Logistic Regression" (Efron 1988) are the *same classical estimator* under three names.

### 2. Does the paper claim GPU / GPU scale? — **No.**

This is the decisive point for the claim under evaluation.

- **Neither Gensheimer & Narasimhan (2019) nor Kvamme & Borgan mention GPU, CUDA, or hardware acceleration anywhere.**
- Gensheimer's "scalable" refers strictly to **mini-batch SGD / out-of-memory (out-of-core) datasets**, not GPU. Abstract: "The use of SGD enables rapid convergence and application to large datasets that do not fit in memory." Body (p. 3): "The loss function depends only on the information contained in the current minibatch, which enables rapid training with minibatch SGD and application to arbitrary-size datasets."
- The scalability benchmark (1,000 → 1,000,000 patients) was run **entirely on CPU, single-core** (p. 9): "An Ubuntu Linux server with 3.6GHz Intel Xeon E5-1650 CPUs and 32GB of RAM was used. **The models were constrained to run on one CPU core.**"

So the specific phrase "on GPU at scale" is **not** claimed or demonstrated by the foundational discrete-time-survival-NN paper.

### 3. The catch that threatens an unqualified claim

Although the *papers* never advertise GPU, the *implementations are GPU-native by construction*:
- Nnet-survival is Keras/TensorFlow; pycox Logistic-Hazard is PyTorch ("built on ... PyTorch," requires "PyTorch (version >= 1.1)"). Both frameworks move tensors to a GPU automatically when CUDA is present.
- Therefore the classical discrete-time hazard likelihood has been **de facto runnable — and routinely run — on GPUs via these libraries since 2018–2019**, even without a paper saying so. DeepSurv (Cox partial-likelihood NN) and DeepHit (discrete PMF + ranking loss) are likewise GPU-native but are *not* the person-period logistic-hazard estimator; the direct classical analog in the neural family is specifically Nnet-survival / Logistic-Hazard.

**Consequence:** An unqualified claim like "first classical discrete-time survival estimator run on GPU at scale" is **not defensible** — pycox's Logistic-Hazard (and Keras Nnet-survival) are exactly that estimator and are GPU-executed by default. Anyone who trained pycox Logistic-Hazard on a CUDA box already did this.

### 4. What *would* be defensible (the real differentiators)

A narrower claim survives, because the neural-net prior art differs from a classical statistical estimator on three axes none of those papers cross:

1. **Exact MLE vs. stochastic optimization.** Nnet-survival/pycox fit by SGD/Adam/RMSprop (approximate, mini-batch). A classical estimator fits by **exact IRLS/Newton–Raphson to convergence** — the same fit R produces.
2. **Full inferential output.** The neural tools emit predictions evaluated by C-index/Brier only. A classical estimator emits **coefficient estimates, standard errors, Wald/LR tests, deviance** — matching `glm(..., family=binomial("cloglog"))` on person-period data.
3. **Validated to R-equivalence.** None of these papers validate parameter-level equivalence against a reference implementation; they report discrimination/calibration metrics. R-equivalence validation is a different bar.

So a defensible framing is roughly: *"first **exact-MLE** classical discrete-time (person-period logistic / cloglog) survival estimator, with full R-equivalent inference, **GPU-accelerated** at scale"* — explicitly contrasting with the SGD-trained, prediction-only neural implementations (Nnet-survival, pycox Logistic-Hazard). The words "classical," "exact MLE / R-equivalent," and "with inference" are load-bearing; drop them and the claim is anticipated by pycox/Nnet-survival.

### Key sources
- Gensheimer & Narasimhan, "A scalable discrete-time survival model for neural networks," PeerJ 7:e6257 (2019): https://peerj.com/articles/6257/ — arXiv preprint (full text I read): https://arxiv.org/abs/1805.00917 (PDF https://arxiv.org/pdf/1805.00917) — PubMed: https://pubmed.ncbi.nlm.nih.gov/30701130/
- Nnet-survival code (Keras/TensorFlow): https://github.com/MGensheimer/nnet-survival
- Kvamme & Borgan, "Continuous and discrete-time survival prediction with neural networks": https://pmc.ncbi.nlm.nih.gov/articles/PMC8536596/
- pycox (PyTorch; Logistic-Hazard = Partial Logistic Regression = Nnet-survival; also DeepSurv, DeepHit, PC-Hazard, MTLR): https://github.com/havakv/pycox
- Classical person-period logistic / cloglog discrete-time survival in R (`glm`): https://www.rensvandeschoot.com/tutorials/discrete-time-survival/ and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9316420/

Note: I did not modify any repository files — this was read-only research. No local pystatistics source references the above claim; the assessment is based on the external literature only.
