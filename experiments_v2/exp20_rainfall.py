"""exp20 - application: monthly rainfall anomalies, 38 Indonesian provinces, 1985-2024.

Data: CHIRPS v2.0 province means (data/panel_rainfall_long.csv). Transform:
square root of monthly rainfall, standardized by calendar month and province
(removes the seasonal cycle and stabilizes the variance).
W: row-standardized inverse great-circle distance between province centroids.

Steps
 1. order identification: BIC for p = 1..4, STPACF, companion spectral radius,
    residual skewness/kurtosis (is the Gaussian reference reasonable?)
 2. detection with the proposed detector and the baselines
 3. weight-matrix sensitivity (inverse distance, 3-nearest neighbours, uniform)
 4. application-size calibration: parametric bootstrap of the first-scan
    maximum from the fitted (cleaned) model, R = 500, gives the false-alarm
    rate at N = 38, T = 480 (dimension not covered by exp10)
 5. semi-synthetic validation: plant AOs/IOs of known size into the real
    (cleaned) anomaly panel and measure recovery, R = 300 per cell
"""
import json
import time
import zlib

import numpy as np
from scipy import stats

from _common import RESULTS
from _rain import load_rain, anomalies, W_rain
import designs as D
from gstar_outlier import general as G
from gstar_outlier.identification import stpacf, stpacf_se

t_start = time.time()
Y, idx, cen = load_rain()
Y, idx = Y.loc[:"2024-12"], idx.loc[:"2024-12"]
A = anomalies(Y)
W = W_rain(cen)
names = cen["name"].values
T, N = A.shape
out = {"T": T, "N": N, "period": [Y.index[0], Y.index[-1]]}
log = []
p_ = log.append

# 1. identification -------------------------------------------------------
bic = {}
for p in range(1, 5):
    f = G.fit_gstar_p(A[4 - p:], W, p)        # common effective sample
    n = len(f.resid)
    S = f.resid.T @ f.resid / n
    bic[p] = float(np.linalg.slogdet(S)[1] + 2 * p * N * np.log(n) / n)
out["bic"] = bic
sp = stpacf(A, W, 4)
out["stpacf"] = {k: [float(a), float(b)] for k, (a, b) in sp.items()}
out["stpacf_bound"] = float(stpacf_se(T, N))
fit1 = G.fit_gstar_p(A, W, 1)
out["spectral_radius"] = G.companion_radius(fit1.Ms)
z = fit1.resid / np.sqrt(np.diag(fit1.Sigma))
out["resid_kurtosis_median_max"] = [float(np.median(stats.kurtosis(z, axis=0))), float(stats.kurtosis(z, axis=0).max())]
out["resid_skew_median"] = float(np.median(stats.skew(z, axis=0)))
Cr = np.corrcoef(fit1.resid.T)
iu = np.triu_indices(N, 1)
out["resid_corr_mean_abs_max"] = [float(np.abs(Cr[iu]).mean()), float(Cr[iu].max())]
p_(f"identification: BIC {bic}; spectral radius {out['spectral_radius']:.3f}")

# 2. detection ---------------------------------------------------------------
dets = {}
for det, fk in [("full", "gstar"), ("perloc", "gstar"), ("perloc", "ar"), ("full", "var"), ("diag", "gstar")]:
    r = G.iterative(A, W, 1, det, fk, max_outliers=60, max_sweeps=15)
    dets[f"{det}/{fk}"] = r
    p_(f"{det}/{fk}: {len(r.detections)} detections, {r.n_sweeps} sweeps, converged={r.converged}, hit_cap={r.hit_cap}, c2 = {r.c2:.1f}")
main = dets["full/gstar"]
rows = []
for d in sorted(main.detections, key=lambda d: d.t):
    order = np.argsort(-np.abs(d.tstats))
    rows.append(dict(month=Y.index[d.t], type=d.kind, lam2_AO=round(d.stat_ao, 1), lam2_IO=round(d.stat_io, 1),
                     nino34=float(idx.nino34.iloc[d.t]), dmi=float(idx.dmi.iloc[d.t]),
                     max_abs_t=float(np.abs(d.tstats).max()),
                     n_abs_t_gt2=int((np.abs(d.tstats) > 2).sum()),
                     top3=[f"{names[k]} ({d.tstats[k]:+.1f})" for k in order[:3]],
                     tstats=[float(v) for v in d.tstats]))
out["detections_full"] = rows
out["n_detections"] = {k: len(v.detections) for k, v in dets.items()}
out["c2"] = {k: v.c2 for k, v in dets.items()}
out["converged_hit_cap"] = {k: [v.converged, v.hit_cap, v.n_sweeps] for k, v in dets.items()}
out["months_var"] = sorted(Y.index[d.t] for d in dets["full/var"].detections)

# whitening check for each detection: largest single |z| of the raw residual
e1 = G._resid_from(A, main.fit.Ms)
out["max_abs_resid_z_at_detections"] = [float(np.max(np.abs(e1[d.t - 1] / np.sqrt(np.diag(main.fit.Sigma))))) for d in main.detections]
out["perloc_threshold_abs_z"] = float(np.sqrt(G.threshold("perloc", N, T - 1)))

# 3. W sensitivity -------------------------------------------------------------
C = cen[["lat", "lon"]].values
from _common import W_invdist
def knn(C, k):
    lat, lon = np.radians(C[:, 0]), np.radians(C[:, 1])
    d = np.arccos(np.clip(np.sin(lat[:, None]) * np.sin(lat) + np.cos(lat[:, None]) * np.cos(lat) * np.cos(lon[:, None] - lon), -1, 1))
    Wk = np.zeros((N, N))
    for i in range(N):
        Wk[i, np.argsort(d[i])[1:k + 1]] = 1
    return Wk / Wk.sum(1, keepdims=True)
wsens = {}
base = {Y.index[d.t] for d in main.detections}
for name, Wx in [("inverse distance", W), ("3 nearest neighbours", knn(C, 3)), ("uniform", D.W_uniform(N))]:
    r = G.iterative(A, Wx, 1, "full", "gstar", max_outliers=60, max_sweeps=15)
    ms = {Y.index[d.t] for d in r.detections}
    wsens[name] = dict(n=len(ms), overlap_with_invdist=len(ms & base), months=sorted(ms))
out["W_sensitivity"] = wsens
p_(f"W sensitivity: {{k: (v['n'], v['overlap_with_invdist']) for k, v in wsens.items()}}")

# 4. parametric bootstrap at application size -----------------------------------
fitc = main.fit
rng = np.random.default_rng(zlib.crc32(b"rain-boot"))
RB = 500
mx = {"full": [], "perloc": []}
for _ in range(RB):
    Zb = G.simulate_gstar_p(fitc.Ms, fitc.Sigma, T, rng=rng)
    for det in mx:
        mx[det].append(G.initial_max(Zb, W, 1, det, "gstar")[0])
out["bootstrap"] = {det: dict(fwer_nominal=float(np.mean(np.array(v) > G.threshold(det, N, T - 1))),
                              c2_size_adjusted=float(np.quantile(v, 0.95)), c2_nominal=G.threshold(det, N, T - 1))
                    for det, v in mx.items()}
p_(f"bootstrap: {out['bootstrap']}")

# 5. semi-synthetic validation -------------------------------------------------
Aclean = G._clean(A, main.detections, fitc.Ms)
sd = np.sqrt(np.diag(fitc.Sigma))
RS = 300
semi = []
for pattern in ("single", "cluster"):
    for kind in ("AO", "IO"):
        for delta in (3, 4, 5):
            rng = np.random.default_rng(zlib.crc32(repr((pattern, kind, delta)).encode()))
            hit = {"full": 0, "perloc": 0}
            false = {"full": 0, "perloc": 0}
            for _ in range(RS):
                t0 = int(rng.integers(12, T - 12))
                j = int(rng.integers(N))
                if pattern == "single":
                    ix = [j]
                else:
                    lat, lon = np.radians(C[:, 0]), np.radians(C[:, 1])
                    dd = np.arccos(np.clip(np.sin(lat[j]) * np.sin(lat) + np.cos(lat[j]) * np.cos(lat) * np.cos(lon[j] - lon), -1, 1))
                    ix = list(np.argsort(dd)[:3])
                om = np.zeros(N)
                om[ix] = delta * sd[ix]
                Psi = G.inject(Aclean, fitc.Ms, t0, om, kind)
                for det in hit:
                    r = G.iterative(Psi, W, 1, det, "gstar", max_outliers=60, max_sweeps=15)
                    near = [d for d in r.detections if abs(d.t - t0) <= 1]
                    hit[det] += bool(near)
                    false[det] += len(r.detections) - len(near)
            semi.append(dict(pattern=pattern, kind=kind, delta=delta,
                             power_full=hit["full"] / RS, power_perloc=hit["perloc"] / RS,
                             false_full=false["full"] / RS, false_perloc=false["perloc"] / RS))
            p_(str(semi[-1]))
out["semi_synthetic"] = semi
out["runtime_s"] = time.time() - t_start
(RESULTS / "exp20_rainfall.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
(RESULTS / "exp20_rainfall.log").write_text("\n".join(log), encoding="utf-8")
np.savez(RESULTS / "exp20_rainfall_series.npz", A=A, W=W, months=np.array(Y.index), names=names,
         lat=C[:, 0], lon=C[:, 1], region_id=np.array(cen.index))
print("\n".join(log))
