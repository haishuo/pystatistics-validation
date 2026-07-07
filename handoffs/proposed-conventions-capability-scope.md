# PROPOSAL — new `CONVENTIONS.md` section: "Capability scope vs R"

Draft for review. Do NOT merge into `../pystatistics/pystatistics/CONVENTIONS.md` until the wording
is approved. This section exists because the parity bar (math must match R; omissions allowed only
with a GOOD, explained, documented reason) requires every deliberate divergence from R's capability
surface to be justified *in the constitution*, not left as an undocumented gap. Only genuinely
specialist / interface-only / superseded capabilities appear here; everything a working statistician
routinely reaches for is implemented, not scoped out.

---

## N. Capability scope vs R

PyStatistics targets **mathematical parity with R** — every estimator it offers matches the R
reference to the stated tolerance. The *interface* may differ from R, and PyStatistics may
deliberately not implement a capability R offers, **only** where the omission is (a) genuinely
specialist or interface-only, (b) explained here, and (c) fail-loud (never silently accepted). The
following are the accepted, deliberate scope boundaries. Anything not listed here and present in R's
mainstream surface is a gap to close, not a silent omission.

### gam
- **Smooth bases** are `cr` (cubic regression), `tp` (thin-plate), `cc` (cyclic cubic), and `ps`
  (P-splines) — the bases that cover mainstream smoothing, seasonal/periodic terms, and penalized
  regression splines. The specialist mgcv bases `re`, `ds` (Duchon), `gp` (Gaussian-process), and
  `fs` (factor-smooth random effects) are **not** offered; an unrecognized `bs=` fails loud. *Reason:
  these serve niche modeling regimes (random-effect smooths are better expressed through `mixed`;
  Duchon/GP bases are specialist spatial tools); implementing them would add penalty machinery used
  by a small minority of GAM users.*

### regression
- **Fully-general `quasi(link, variance)`** with an arbitrary mean–variance relationship is not
  offered. The mainstream overdispersion families **`quasipoisson` and `quasibinomial` are**
  implemented. *Reason: the arbitrary-variance constructor is a specialist tool; the two standard
  quasi-likelihood cases cover essentially all applied use.*
- **Non-treatment contrast codings** (Helmert / sum / polynomial) are not exposed; PyStatistics uses
  treatment (dummy) coding, which is also R's default. *Reason: this is an interface choice, not a
  mathematical one — the **fit is identical** regardless of contrast coding; only the parameterization
  of the reported coefficients differs, and any contrast is recoverable by re-coding the design.*

### survival
- **Exact partial-likelihood ties** (`ties='exact'`) are not offered; **Efron (default) and Breslow
  are.** An unsupported `ties=` fails loud. *Reason: Efron is the modern default and Breslow the
  classic alternative; the exact method is rarely needed in practice and is computationally costly —
  the two supported methods match `survival::coxph` on their respective conventions.*

### montecarlo
- **Variance-stabilizing transforms** (`h`/`hinv` on `boot.ci`), **antithetic sampling**, and
  **stratified / blocked permutation** are not offered. All five bootstrap-CI types
  (normal/basic/percentile/BCa/studentized) and ordinary + stratified bootstrap are present. *Reason:
  these are advanced variance-reduction refinements; a transform, where wanted, is applied by the
  caller to the statistic, and the CI types PyStatistics computes match `boot.ci` on shared
  replicates.*

### multivariate
- **`prcomp`-style `tol=` rank truncation** is not a separate parameter; `n_components=` provides the
  equivalent control (retain k components / drop the trailing spectrum). *Reason: interface
  consolidation — one truncation control, not two spellings of the same operation.*

---

**Note on what is NOT here:** capabilities a working statistician routinely reaches for — gam tensor
smooths (`te`/`ti`/`s(x,z)`) and `by=` smooths, gam `nb()` with estimated theta, arima `xreg`/drift,
stratified & time-varying Cox, general GLM links/families, glmm offset/weights — are **being
implemented**, not scoped out. They do not belong in this section; they are tracked as CLOSE items in
`handoffs/final-bless-completeness-audit.md`.
