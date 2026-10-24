"""exp14 - sensitivity to the spatial weight matrix (R1-1).

Data are generated with the inverse-distance W; the model is fitted with
  inverse-distance (correct), 2-nearest-neighbour contiguity, uniform weights,
  or no spatial lag at all (N separate AR(1) models; the detector still uses
  the full residual covariance).
N = 8, T = 250, moderate and high correlation. Reported: null FWER (R0 = 1000)
and power for one single-location AO or IO of 4 sd (R = 500), nominal threshold.
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

N, T, R, R0, DELTA = 8, 250, 500, 1000, 4.0
FITS = ["inverse-distance (true)", "2-nearest-neighbour", "uniform", "no spatial lag"]


def W_for(C, name):
    if name.startswith("inverse"):
        return D.W_inverse_distance(C), "gstar"
    if name.startswith("2-nearest"):
        return D.W_knn(C, 2), "gstar"
    if name == "uniform":
        return D.W_uniform(len(C)), "gstar"
    return D.W_inverse_distance(C), "ar"


def one(cfg):
    level, fitname = cfg
    C, Wtrue, Sigma, Ms = D.design(N, level)
    Wf, kind_fit = W_for(C, fitname)
    sd = np.sqrt(np.diag(Sigma))
    rng = np.random.default_rng(zlib.crc32(repr(cfg).encode()))
    c2 = G.threshold("full", N, T - 1)
    fw = 0
    for _ in range(R0):
        Z = G.simulate_gstar_p(Ms, Sigma, T, rng=rng)
        fw += G.initial_max(Z, Wf, 1, "full", kind_fit)[0] > c2
    out = dict(corr=level, fitted_W=fitname, fwer=fw / R0)
    for kind in ("AO", "IO"):
        hit = 0
        for _ in range(R):
            Z = G.simulate_gstar_p(Ms, Sigma, T, rng=rng)
            t0 = int(rng.integers(20, T - 20))
            j = int(rng.integers(N))
            om = np.zeros(N)
            om[j] = DELTA * sd[j]
            res = G.iterative(G.inject(Z, Ms, t0, om, kind), Wf, 1, "full", kind_fit)
            hit += any(abs(d.t - t0) <= 1 for d in res.detections)
        out[f"power_{kind}"] = hit / R
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as pool:
        out = pool.map(one, list(itertools.product(["moderate", "high"], FITS)))
    with open(RESULTS / "exp14_weights.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("done", time.time() - t0)
