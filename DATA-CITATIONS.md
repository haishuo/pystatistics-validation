# Data citations

The validation reports in this repository compare `pystatistics` against R on
**real survey microdata**. Several of the source programmes require that any
published work using their data cite it. This file discharges that obligation for
every dataset the drivers read.

**No survey microdata is distributed by this repository.** The `artifacts/` and
`reports/` trees contain only derived quantities — log-likelihoods, parameter
estimates, iteration counts and timings. The microdata itself is obtained by each
user directly from the programmes below, under those programmes' own terms.

---

## General Social Survey

Davern, Michael; Bautista, Rene; Freese, Jeremy; Herd, Pamela; and Morgan,
Stephen L. *General Social Survey 1972–2024 Cumulative File* (Release 3, March
2026) [Data set]. Chicago: NORC at the University of Chicago.
<https://gss.norc.org/get-the-data>

File as used: `gss7224_r3.dta`.

## World Values Survey

Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano,
J., Lagos, M., Norris, P., Ponarin, E. & Puranen, B. et al. (eds.). 2022. *World
Values Survey: Round Seven — Country-Pooled Datafile Version 6.0.0*. Madrid,
Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat.
doi:10.14281/18241.24

File as used: `WVS_Cross-National_Wave_7_spss_v6_0.sav`.

## Comparative Study of Electoral Systems

The Comparative Study of Electoral Systems (www.cses.org). *CSES INTEGRATED
MODULE DATASET PHASE 4 RELEASE* [dataset and documentation]. February 27, 2024
version. doi:10.7804/cses.imd.2024-02-27

File as used: `cses_imd.sav`. This is the citation form the CSES Secretariat
specifies, taken verbatim from the codebook shipped with the data — cses.org has
been offline since at least 2026-08-05.

> Phase 4 is an **advance release**: the codebook states it lacks some of what
> the Full Release will carry. Worth noting when comparing against other work.

## Afrobarometer

Afrobarometer Data, Round 9, 2021/2023, available at
<https://www.afrobarometer.org>

File as used: `afrobarometer_r9_39ctry.sav`. Afrobarometer's data usage policy
requires acknowledgement in this bibliographic form.

> Round 9 fieldwork years to be confirmed against the release documentation.

---

## Terms of use

These datasets are obtained from their originating programmes under those
programmes' terms, which vary and in some cases restrict redistribution and
non-research use. Anyone reproducing this validation must obtain the data
themselves and accept those terms directly. Do not commit survey microdata, or
any row-level extract of it, to this repository.
