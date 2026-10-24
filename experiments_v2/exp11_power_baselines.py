"""exp11 - power against baselines and the covariance ablation (R1-2, R2-2).

N = 8 locations, T = 250, GSTAR(1;1), Gaussian errors with low / moderate / high
spatial correlation. One outlier (AO or IO) of magnitude delta (in units of the
error standard deviation) is planted at a random time in [20, T-20] with one of
three spatial patterns:
    single  - one location j (random)
    cluster - location j and its two nearest neighbours
    global  - all eight locations
Detectors (same residual signatures, sweeps and re-estimation; see general.py):
    full/gstar   proposed
    diag/gstar   ablation: quadratic statistic with diag(Sigma)
    perloc/gstar per-location maximum test (calibrated Huda-type / Tsay C-type)
    perloc/ar    N independent AR(1) series, per-location maximum test
    full/var     unrestricted VAR(1) (Tsay et al. 2000 J-type)
Each detector is run with (a) its nominal Bonferroni threshold and (b) a
size-adjusted threshold: the 95% quantile of its first-scan maximum under the
null, from R0 = 1000 outlier-free series of the same design. (b) compares power
at an exactly equal false-alarm rate of 5%.
A planted outlier counts as detected if a detection lies within +-1 time point.
"""
import csv
import itertools
import time
import zlib
from multiprocessing import Pool

import numpy as np

from _common import RESULTS
import designs as D
from gstar_outlier import general as G

N, T, R, R0 = 8, 250, 500, 1000
DETS = [("full", "gstar"), ("diag", "gstar"), ("perloc", "gstar"), ("perloc", "ar"), ("full", "var")]
LEVELS = ["low", "moderate", "high"]


def null_quantiles(level):
    C, W, Sigma, Ms = D.make_design(N, level)
    rng = np.random.default_rng(zlib.crc32(f"null-{level}".encode()))
    mx = {d: [] for d in DETS}
    for _ in range(R0):
        Z = G.simulate_model(Ms, Sigma, T, rng=rng)
        for det, fk in DETS:
            mx[(det, fk)].append(G.first_scan_maximum(Z, W, 1, det, fk)[0])
    return level, {d: float(np.quantile(v, 0.95)) for d, v in mx.items()}, \
        {d: float(np.mean(np.array(v) > G.bonferroni_threshold(d[0], N, T - 1))) for d, v in mx.items()}


def run_configuration(args):
    (level, pattern, kind, delta), cadj = args
    C, W, Sigma, Ms = D.make_design(N, level)
    rng = np.random.default_rng(zlib.crc32(repr((level, pattern, kind, delta)).encode()))
    sd = np.sqrt(np.diag(Sigma))
    acc = {(d, mode): dict(hit=0, type_ok=0, loc_ok=0, false=0) for d in DETS for mode in ("nominal", "adjusted")}
    for _ in range(R):
        Z = G.simulate_model(Ms, Sigma, T, rng=rng)
        t0 = int(rng.integers(20, T - 20))
        j = int(rng.integers(N))
        idx = {"single": [j], "cluster": [j] + D.nearest_neighbours(C, j, 2), "global": list(range(N))}[pattern]
        omega = np.zeros(N)
        omega[idx] = delta * sd[idx]
        Psi = G.add_outlier(Z, Ms, t0, omega, kind)
        for det, fk in DETS:
            for mode in ("nominal", "adjusted"):
                c2 = None if mode == "nominal" else cadj[(det, fk)]
                res = G.iterative_detection(Psi, W, 1, det, fk, c2=c2)
                a = acc[((det, fk), mode)]
                near = [d for d in res.detections if abs(d.t - t0) <= 1]
                a["false"] += len(res.detections) - len(near)
                if near:
                    a["hit"] += 1
                    best = max(near, key=lambda d: max(d.stat_ao, d.stat_io))
                    a["type_ok"] += best.kind == kind
                    a["loc_ok"] += best.loc == j
    rows = []
    for ((det, fk), mode), a in acc.items():
        rows.append(dict(corr=level, pattern=pattern, kind=kind, delta=delta, detector=det, fit=fk,
                         threshold=mode, power=a["hit"] / R,
                         type_acc=a["type_ok"] / a["hit"] if a["hit"] else np.nan,
                         loc_acc=(a["loc_ok"] / a["hit"] if a["hit"] else np.nan) if pattern == "single" else np.nan,
                         false_per_series=a["false"] / R))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as pool:
        nq = pool.map(null_quantiles, LEVELS)
    cadj = {lv: q for lv, q, _ in nq}
    with open(RESULTS / "exp11_null_thresholds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corr", "detector", "fit", "c2_nominal", "fwer_nominal", "c2_size_adjusted"])
        for lv, q, fw in nq:
            for d in DETS:
                w.writerow([lv, d[0], d[1], round(G.bonferroni_threshold(d[0], N, T - 1), 3), fw[d], round(q[d], 3)])
    print(f"null quantiles done {time.time()-t0:.0f}s", flush=True)
    grid = list(itertools.product(LEVELS, ["single", "cluster", "global"], ["AO", "IO"], [3, 4, 5]))
    out = []
    with Pool(2) as pool:
        for k, rows in enumerate(pool.imap_unordered(run_configuration, [(g, cadj[g[0]]) for g in grid])):
            out += rows
            print(f"{k+1}/{len(grid)} {time.time()-t0:.0f}s", flush=True)
    out.sort(key=lambda r: (r["corr"], r["pattern"], r["kind"], r["delta"], r["detector"], r["fit"], r["threshold"]))
    with open(RESULTS / "exp11_power_baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("saved", time.time() - t0)
