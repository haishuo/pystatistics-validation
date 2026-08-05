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

World Values Survey Wave 7 (2017–2022), cross-national data file, version 6.0.
Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat.
<https://www.worldvaluessurvey.org/>

File as used: `WVS_Cross-National_Wave_7_spss_v6_0.sav`.

> The exact citation string WVS requires for v6.0 should be confirmed against the
> programme's own "how to cite" page before this file is treated as
> authoritative. The dataset identity above is exact; the formatted reference is
> reconstructed.

## Comparative Study of Electoral Systems

The Comparative Study of Electoral Systems. *CSES Integrated Module Dataset
(IMD)* [Data set]. <https://cses.org/>

File as used: `cses_imd.sav`.

> Formatted reference to be confirmed against the CSES release documentation.

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
