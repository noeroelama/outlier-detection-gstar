# Covariance-aware outlier detection in GSTAR models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21185669.svg)](https://doi.org/10.5281/zenodo.21185669)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation and replication code for the paper

> Nurwahid, A. *Covariance-aware outlier detection in GSTAR models, with an
> application to rubella surveillance in Japan.* (submitted)

The generalized space-time autoregressive (GSTAR) model is widely used for
spatiotemporal count and rate data, but its least-squares fit is sensitive
to outliers. This repository implements an iterative procedure that detects
**additive (AO)** and **innovative (IO)** outliers and folds them into the
model instead of deleting observations. Unlike earlier per-location
procedures, the likelihood-ratio statistics use the **full residual
covariance matrix**, so correlation between locations enters the test; the
detection threshold is **Bonferroni-calibrated** so the family-wise
false-detection rate is controlled at any series length; and each flagged
time point is localized in space by per-location *t*-statistics.

## What is here

```
src/gstar_outlier/
  model.py          GSTAR(1;1) simulation and least-squares fitting; VAR(1) fit
  detection.py      GLS outlier-effect estimators, chi-square statistics,
                    Bonferroni threshold, iterative detect-adjust-refit loop
  benchmark.py      per-location (diagonal-variance) benchmark detector
  identification.py STACF / STPACF, Hosking portmanteau, order diagnostics
experiments/        one script per experiment in the paper (see below)
data/               assembled weekly rubella panel + provenance and rebuild script
results/            precomputed outputs of the experiments
```

## Method in one paragraph

For a GSTAR(1;1) model `Z_t = (Phi0 + Phi1 W) Z_{t-1} + e_t` with
`e_t ~ N(0, Sigma)` and `M = Phi0 + Phi1 W`, an outlier at time `T` leaves a
finite pattern in the residuals `e_t = Z_t - M Z_{t-1}`
(AO: `e_T = eps_T + w`, `e_{T+1} = eps_{T+1} - M w`; IO: `e_T = eps_T + w`).
Generalized least squares gives the effect estimators and their exact
covariances, and the standardized quadratic statistics
`lambda^2_AO = w_hat' A w_hat` and `lambda^2_IO = e_T' Sigma^{-1} e_T`
(with `A = Sigma^{-1} + M' Sigma^{-1} M`) are each chi-square with `N`
degrees of freedom under the null. See the paper for the derivation.

## Reproducing the paper

Requires Python 3.10+ with numpy, scipy, pandas, matplotlib
(`pip install -r requirements.txt`). From the repository root:

```
python experiments/exp01_sanity.py          # sanity check: recover a planted AO and IO
python experiments/exp02_size_power.py       # size and power (Table 1, Fig. 3)
python experiments/exp03_vs_perloc.py        # vs per-location statistics (Table 2, Fig. 4)
python experiments/exp04_masking_joint.py    # multiple outliers / masking (Table 3)
python experiments/exp06_vs_var_tpp.py       # vs unrestricted VAR(1) detection (Sec. 4.4)
python experiments/exp07_rubella_order.py    # rubella order identification (Sec. 5.1)
python experiments/exp08_rubella_japan.py    # rubella application (Sec. 5, Table 4)
python experiments/make_figures.py           # regenerate the four figures -> figures/
```

The Monte Carlo scripts (`exp02`, `exp03`, `exp06`) use multiprocessing and
take a few minutes; random seeds are fixed, so reruns reproduce the shipped
numbers in `results/`. The experiment numbering skips `exp05`: that was an
exploratory analysis on COVID-19 data that is not part of the paper.


## Revision for *Array* (version 2, October 2026)

The revised paper ("Outlier detection in generalized space-time autoregressive
models using the full error covariance, with an application to monthly rainfall
in Indonesia") uses the new module `src/gstar_outlier/general.py`:

- GSTAR(p;1), VAR(p) and independent AR(p) fitting;
- the full-covariance AO/IO statistics for any order p (for p = 1 they equal
  `detection.py`), the diagonal-covariance statistic and the calibrated
  per-location maximum statistic;
- the combined test (`detector="combo"`): full-covariance OR per-location
  maximum statistic, each at level alpha/2 (recommended for routine use);
- the iterative procedure `general.iterative`, which also returns
  `converged` and `hit_cap`. `hit_cap = True` means a sweep stopped at
  `max_outliers`; the count of outliers is then set by the cap, not by the data,
  and the Gaussian reference is probably inadequate (this happened with the
  weekly rubella counts used in version 1, which is why that application was
  dropped).

Scripts (run from `experiments_v2/`, fixed seeds, results in `results_v2/`):

```
python exp10_fwer.py            # family-wise error rate (Table 1)            ~3 min on 2 cores
python exp11_power_baselines.py # power vs baselines (Fig. 4, B.1)       ~15 min
python exp15_gstar_vs_var.py    # GSTAR fit vs unrestricted VAR fit (Table 2)
python exp16_combined.py        # combined test, paired with exp10-12 and exp20 (Table 3, Table 6)
python exp12_adjacent.py        # adjacent outliers (Table 4)
python exp13_classification.py  # AO/IO misclassification (Fig. 5)
python exp14_weights.py         # weight-matrix sensitivity (Table 5)
python exp20_rainfall.py        # rainfall application (Tables 6-7)       ~15 min
python make_tables_v2.py        # LaTeX table rows -> tables_v2/
python make_diagrams_v2.py      # explanatory diagrams (Figs. 1-2)
python make_figures_v2.py       # figures -> figures_v2/ (needs geopandas and
                                #   data/external/provinces_simplified.geojson)
```

Rainfall data: `data/panel_rainfall_long.csv` (CHIRPS v2.0 province means,
38 provinces, 1985-01 to 2025-04; the paper uses 1985-2024) and
`data/a_geo_centroids.csv` (province centroids). The province polygons
(batas-administrasi-indonesia, 13 June 2023) are not redistributed; place them
in `data/external/` to redraw the maps.

## Data (version 1)

The weekly rubella panel in `data/` is assembled from Japan's national
notifiable-disease surveillance (NESID/JIHS); see `data/README.md` for
provenance and `data/download_jihs.py` to rebuild it from source.

## Citing

If you use this code or data, please cite the paper (see `CITATION.cff`).

## License

MIT (see `LICENSE`). The redistributed surveillance data are public and are
included for reproducibility with attribution to their source.
