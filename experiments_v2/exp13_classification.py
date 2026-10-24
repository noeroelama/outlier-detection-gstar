"""exp13 - AO/IO classification error as the outlier grows (R2-3).

Proposition 3 of the revised paper: for a fixed direction, the probability that
the larger-statistic rule picks the wrong type tends to zero as the magnitude
grows, provided the outlier is not annihilated by the dynamics (M omega != 0).
For moderate magnitudes the error is not zero, and this experiment reports it.

N = 8, T = 250, moderate correlation, GSTAR(1;1), proposed detector, nominal
threshold; single-location and cluster patterns; delta in {3,...,12};
R = 500 per cell. Misclassification is computed among detected outliers.
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

N, T, R, LEVEL = 8, 250, 500, "moderate"
DELTAS = [3, 4, 5, 6, 8, 10, 12]


def run_replication(cfg):
    pattern, kind, delta = cfg
    C, W, Sigma, Ms = D.make_design(N, LEVEL)
    sd = np.sqrt(np.diag(Sigma))
    rng = np.random.default_rng(zlib.crc32(repr(cfg).encode()))
    hit = wrong = 0
    for _ in range(R):
        Z = G.simulate_model(Ms, Sigma, T, rng=rng)
        t0 = int(rng.integers(20, T - 20))
        j = int(rng.integers(N))
        idx = [j] if pattern == "single" else [j] + D.nearest_neighbours(C, j, 2)
        om = np.zeros(N)
        om[idx] = delta * sd[idx]
        res = G.iterative_detection(G.add_outlier(Z, Ms, t0, om, kind), W, 1)
        near = [d for d in res.detections if abs(d.t - t0) <= 1]
        if near:
            hit += 1
            best = max(near, key=lambda d: max(d.stat_ao, d.stat_io))
            wrong += best.kind != kind
    return dict(pattern=pattern, kind=kind, delta=delta, power=hit / R,
                misclass=wrong / hit if hit else np.nan, n_detected=hit)


if __name__ == "__main__":
    grid = list(itertools.product(["single", "cluster"], ["AO", "IO"], DELTAS))
    t0 = time.time()
    with Pool(2) as pool:
        out = pool.map(run_replication, grid)
    with open(RESULTS / "exp13_classification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("done", time.time() - t0)
