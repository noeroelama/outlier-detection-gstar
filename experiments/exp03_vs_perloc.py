"""exp03 — multivariate detector vs per-location benchmark (Huda-style).

Both detectors share the identical sweep architecture, signatures, and
Bonferroni FWER control at alpha = 0.05; the ONLY difference is whether the
full residual covariance Sigma (ours) or its diagonal (per-location) enters
the statistics. Any power gap is therefore attributable to the use of
cross-location correlation — the paper's central claim.

Design
------
* T = 250, magnitude 4 sigma, kinds {AO, IO}, R = 1000 per cell
* two error-correlation regimes:
    moderate: corr = (0.5, 0.3, 0.4)   [same as exp01/exp02]
    high    : corr = (0.8, 0.7, 0.75)
* single outlier, uniform-random time in [10, T-10) and location
* clean-series size check per detector per regime (R = 1000)

Expectation from theory: the multivariate noncentrality is
mag^2 * Sigma_jj * (Sigma^{-1})_jj >= mag^2, growing with correlation,
while the per-location statistic's noncentrality stays mag^2. The gap
should widen in the high-correlation regime.
"""

import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gstar_outlier import (  # noqa: E402
    inject_outlier, iterative_detection, make_M, simulate_gstar,
)
from gstar_outlier.benchmark import iterative_detection_perloc  # noqa: E402

W = np.array([
    [0.0,    0.7777, 0.2223],
    [0.7288, 0.0,    0.2712],
    [0.4345, 0.5655, 0.0],
])
PHI0 = np.array([0.40, -0.30, 0.25])
PHI1 = np.array([0.30, 0.20, -0.20])
M = make_M(PHI0, PHI1, W)

SIGMAS = {
    "moderate": np.array([
        [1.00, 0.50, 0.30],
        [0.50, 1.00, 0.40],
        [0.30, 0.40, 1.00],
    ]),
    "high": np.array([
        [1.00, 0.80, 0.70],
        [0.80, 1.00, 0.75],
        [0.70, 0.75, 1.00],
    ]),
}
DETECTORS = {"multivariate": iterative_detection,
             "per-location": iterative_detection_perloc}
ALPHA = 0.05
T = 250
MAG = 4.0
R = 1000


def run_power(args):
    regime, det_name, kind, r = args
    Sigma = SIGMAS[regime]
    rng = np.random.default_rng(hash((regime, det_name, kind, r)) % (2**32))
    Z = simulate_gstar(PHI0, PHI1, W, Sigma, T, rng=rng)
    t0 = int(rng.integers(10, T - 10))
    loc = int(rng.integers(0, 3))
    omega = np.zeros(3)
    omega[loc] = MAG * np.sqrt(Sigma[loc, loc])
    Psi = inject_outlier(Z, M, t0, omega, kind)
    res = DETECTORS[det_name](Psi, W, alpha=ALPHA)
    hit = type_ok = loc_ok = 0
    spurious = 0
    for d in res.detections:
        ts = d.t + 1
        if abs(ts - t0) <= 1 and not hit:
            hit = 1
            type_ok = int(d.kind == kind)
            loc_ok = int(int(np.argmax(np.abs(d.tstats))) == loc)
        else:
            spurious += 1
    return regime, det_name, kind, hit, type_ok, loc_ok, spurious


def run_size(args):
    regime, det_name, r = args
    Sigma = SIGMAS[regime]
    rng = np.random.default_rng(hash((regime, det_name, "clean", r)) % (2**32))
    Z = simulate_gstar(PHI0, PHI1, W, Sigma, T, rng=rng)
    res = DETECTORS[det_name](Z, W, alpha=ALPHA)
    return regime, det_name, len(res.detections)


def main():
    t_start = time.time()
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "exp03_vs_perloc_20260703.csv"

    power_jobs = [(g, d, k, r) for g in SIGMAS for d in DETECTORS
                  for k in ["AO", "IO"] for r in range(R)]
    size_jobs = [(g, d, r) for g in SIGMAS for d in DETECTORS for r in range(R)]
    workers = max(1, mp.cpu_count() - 2)
    print(f"jobs: {len(power_jobs)} power + {len(size_jobs)} size, "
          f"workers={workers}", flush=True)

    with mp.Pool(workers) as pool:
        power_res = pool.map(run_power, power_jobs, chunksize=50)
        size_res = pool.map(run_size, size_jobs, chunksize=50)

    cells = {}
    for g, d, k, hit, tok, lok, spur in power_res:
        c = cells.setdefault((g, d, k), [0, 0, 0, 0, 0])
        c[0] += 1; c[1] += hit; c[2] += tok; c[3] += lok; c[4] += spur
    size = {}
    for g, d, ndet in size_res:
        s = size.setdefault((g, d), [0, 0])
        s[0] += 1; s[1] += ndet > 0

    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["regime", "detector", "kind", "R", "detect_rate",
                       "type_acc_given_hit", "loc_acc_given_hit",
                       "spurious_per_run", "clean_FWER"])
        for (g, d, k), (n, hit, tok, lok, spur) in sorted(cells.items()):
            fw = size[(g, d)][1] / size[(g, d)][0]
            wcsv.writerow([g, d, k, n, hit / n, tok / max(hit, 1),
                           lok / max(hit, 1), spur / n, fw])

    print(f"\n=== exp03: T={T}, magnitude={MAG} sigma, alpha={ALPHA}, R={R} ===")
    print(f"{'regime':<10}{'detector':<14}{'kind':<5}{'detect':>8}"
          f"{'type|hit':>10}{'loc|hit':>9}{'spur':>7}{'FWER':>7}")
    for (g, d, k), (n, hit, tok, lok, spur) in sorted(cells.items()):
        fw = size[(g, d)][1] / size[(g, d)][0]
        print(f"{g:<10}{d:<14}{k:<5}{hit/n:>8.3f}{tok/max(hit,1):>10.3f}"
              f"{lok/max(hit,1):>9.3f}{spur/n:>7.3f}{fw:>7.3f}")
    print(f"\nsaved -> {out_csv}")
    print(f"elapsed: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
