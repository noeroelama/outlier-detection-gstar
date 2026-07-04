# Covariance-aware outlier detection in GSTAR models

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

## Data

The weekly rubella panel in `data/` is assembled from Japan's national
notifiable-disease surveillance (NESID/JIHS); see `data/README.md` for
provenance and `data/download_jihs.py` to rebuild it from source.

## Citing

If you use this code or data, please cite the paper (see `CITATION.cff`).

## License

MIT (see `LICENSE`). The redistributed surveillance data are public and are
included for reproducibility with attribution to their source.
