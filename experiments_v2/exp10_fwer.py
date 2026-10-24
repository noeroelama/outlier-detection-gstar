"""exp10 - family-wise false-alarm rate of the fitted, iterative procedure (R2-1).

Global null (no outliers). If the FIRST scan (statistics computed with the
least-squares estimates from the full series) does not exceed the threshold,
no series is cleaned, the refit is identical and the procedure stops with no
detection. Hence the exceedance rate of the first-scan maximum, which we record
for every detector, is an upper bound on the FWER of the whole procedure.

Grid: N in {3, 8, 16}; T in {100, 250, 500}; error correlation low/high;
Gaussian or Student-t(5) errors; GSTAR(1;1). Also GSTAR(2;1) at N = 8.
R = 1000 series per configuration. The 95% quantile of the first-scan maximum
is saved as the size-adjusted critical value used by exp11.
"""
import csv
import zlib
import itertools
import time
from multiprocessing import Pool

import numpy as np

from _common import RESULTS
import designs as D
from gstar_outlier import general as G

R = 1000
DETS = [("full", "gstar"), ("diag", "gstar"), ("perloc", "gstar"), ("full", "var"), ("perloc", "ar")]


def run_configuration(cfg):
    N, T, level, errs, p = cfg
    C, W, Sigma, Ms = D.make_design(N, level)
    if p == 2:  # split the dynamics over two lags, same total persistence
        Ms = np.stack([0.6 * Ms[0], 0.3 * Ms[0]])
    rng = np.random.default_rng(zlib.crc32(repr(cfg).encode()))
    df = 5 if errs == "t5" else None
    maxima = {d: [] for d in DETS}
    for _ in range(R):
        Z = G.simulate_model(Ms, Sigma, T, rng=rng, df=df)
        for det, fk in DETS:
            m, Teff = G.first_scan_maximum(Z, W, p, det, fk)
            maxima[(det, fk)].append(m)
    rows = []
    for (det, fk), ms in maxima.items():
        ms = np.array(ms)
        c_nom = G.bonferroni_threshold(det, N, T - p)
        row = dict(N=N, T=T, corr=level, errors=errs, p=p, detector=det, fit=fk,
                   c2_nominal=round(c_nom, 3), fwer=float(np.mean(ms > c_nom)),
                   c2_size_adjusted=float(np.quantile(ms, 0.95)))
        rows.append(row)
        if det == "perloc":  # same statistic, conventional constant |z| > 3
            rows.append(dict(row, detector="fixed3", c2_nominal=9.0, fwer=float(np.mean(ms > 9.0)),
                             c2_size_adjusted=float(np.quantile(ms, 0.95))))
    return rows


if __name__ == "__main__":
    grid = list(itertools.product([3, 8, 16], [100, 250, 500], ["low", "high"], ["gauss", "t5"], [1]))
    grid += [(8, T, lv, "gauss", 2) for T in (100, 250, 500) for lv in ("low", "high")]
    t0 = time.time()
    out = []
    with Pool(2) as pool:
        for rows in pool.imap_unordered(run_configuration, grid):
            out += rows
            print(f"{len(out)//6:3d}/{len(grid)} configs, {time.time()-t0:6.0f}s", flush=True)
    out.sort(key=lambda r: (r["p"], r["N"], r["T"], r["corr"], r["errors"], r["detector"], r["fit"]))
    with open(RESULTS / "exp10_fwer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("saved", RESULTS / "exp10_fwer.csv")
