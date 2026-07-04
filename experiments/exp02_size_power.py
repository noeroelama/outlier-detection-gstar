"""exp02 — Monte Carlo size and power of the corrected multivariate procedure.

Design
------
* N = 3 locations, thesis inverse-distance weight matrix (Java), same
  stationary parameters and error correlation as exp01.
* Power cells: kind {AO, IO} x magnitude {3, 4, 5, 6} sigma x T {100, 250, 365},
  R = 1000 replications each. One outlier per series, at a uniform-random
  time in [10, T-10) and a uniform-random location, effect
  omega_loc = mag * sqrt(Sigma_loc,loc).
* Size cells: clean series per T, R = 1000: empirical FWER = P(>= 1 detection)
  at nominal alpha = 0.05 (Bonferroni over 2*T_eff chi2_3 tests).

Metrics per power cell
----------------------
* detect  : share of runs with a detection within +/-1 of the true time
* type_ok : correct AO/IO label given detection
* loc_ok  : argmax |t-stat| equals the true location, given detection
* spurious: mean number of detections away from the true time (masking/noise)

Output: results/exp02_size_power_<date>.csv + printed summary table.
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

W = np.array([
    [0.0,    0.7777, 0.2223],
    [0.7288, 0.0,    0.2712],
    [0.4345, 0.5655, 0.0],
])
PHI0 = np.array([0.40, -0.30, 0.25])
PHI1 = np.array([0.30, 0.20, -0.20])
SIGMA = np.array([
    [1.00, 0.50, 0.30],
    [0.50, 1.00, 0.40],
    [0.30, 0.40, 1.00],
])
M = make_M(PHI0, PHI1, W)
ALPHA = 0.05
R = 1000
KINDS = ["AO", "IO"]
MAGS = [3.0, 4.0, 5.0, 6.0]
TS = [100, 250, 365]


def run_power(args):
    kind, mag, T, r = args
    rng = np.random.default_rng(hash((kind, int(mag * 10), T, r)) % (2**32))
    Z = simulate_gstar(PHI0, PHI1, W, SIGMA, T, rng=rng)
    t0 = int(rng.integers(10, T - 10))
    loc = int(rng.integers(0, 3))
    omega = np.zeros(3)
    omega[loc] = mag * np.sqrt(SIGMA[loc, loc])
    Psi = inject_outlier(Z, M, t0, omega, kind)
    res = iterative_detection(Psi, W, alpha=ALPHA)
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
    return kind, mag, T, hit, type_ok, loc_ok, spurious


def run_size(args):
    T, r = args
    rng = np.random.default_rng(hash(("clean", T, r)) % (2**32))
    Z = simulate_gstar(PHI0, PHI1, W, SIGMA, T, rng=rng)
    res = iterative_detection(Z, W, alpha=ALPHA)
    return T, len(res.detections)


def main():
    t_start = time.time()
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "exp02_size_power_20260703.csv"

    power_jobs = [(k, m, T, r) for k in KINDS for m in MAGS for T in TS
                  for r in range(R)]
    size_jobs = [(T, r) for T in TS for r in range(R)]
    workers = max(1, mp.cpu_count() - 2)
    print(f"jobs: {len(power_jobs)} power + {len(size_jobs)} size, "
          f"workers={workers}", flush=True)

    with mp.Pool(workers) as pool:
        power_res = pool.map(run_power, power_jobs, chunksize=50)
        size_res = pool.map(run_size, size_jobs, chunksize=50)

    # aggregate power
    cells = {}
    for kind, mag, T, hit, tok, lok, spur in power_res:
        c = cells.setdefault((kind, mag, T), [0, 0, 0, 0, 0])
        c[0] += 1; c[1] += hit; c[2] += tok; c[3] += lok; c[4] += spur

    # aggregate size
    size = {}
    for T, ndet in size_res:
        s = size.setdefault(T, [0, 0, 0])
        s[0] += 1; s[1] += ndet > 0; s[2] += ndet

    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["kind", "magnitude", "T", "R", "detect_rate",
                       "type_acc_given_hit", "loc_acc_given_hit",
                       "spurious_per_run"])
        for (kind, mag, T), (n, hit, tok, lok, spur) in sorted(cells.items()):
            wcsv.writerow([kind, mag, T, n, hit / n,
                           tok / max(hit, 1), lok / max(hit, 1), spur / n])
        wcsv.writerow([])
        wcsv.writerow(["clean", "T", "R", "FWER", "mean_detections"])
        for T, (n, fw, nd) in sorted(size.items()):
            wcsv.writerow(["clean", T, n, fw / n, nd / n])

    print(f"\n=== POWER (alpha={ALPHA}, R={R} per cell) ===")
    print(f"{'kind':<5}{'mag':>5}{'T':>6}{'detect':>9}{'type|hit':>10}"
          f"{'loc|hit':>9}{'spur/run':>10}")
    for (kind, mag, T), (n, hit, tok, lok, spur) in sorted(cells.items()):
        print(f"{kind:<5}{mag:>5.1f}{T:>6}{hit/n:>9.3f}"
              f"{tok/max(hit,1):>10.3f}{lok/max(hit,1):>9.3f}{spur/n:>10.3f}")
    print(f"\n=== SIZE (clean series, nominal FWER <= {ALPHA}) ===")
    for T, (n, fw, nd) in sorted(size.items()):
        print(f"  T={T:>4}: empirical FWER={fw/n:.3f}, "
              f"mean detections/run={nd/n:.4f}")
    print(f"\nsaved -> {out_csv}")
    print(f"elapsed: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
