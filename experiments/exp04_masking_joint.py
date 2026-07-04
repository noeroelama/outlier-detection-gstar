"""exp04 — multiple outliers, masking, and joint parameter recovery.

The iterative procedure (inner one-at-a-time detection with residual
adjustment and Sigma re-estimation; outer Chen-Liu-style sweeps that clean
the series and re-fit parameters) is stress-tested against configurations
where naive single-pass detection is known to fail:

  spread3 : 3 outliers (AO, IO, AO), 5 sigma, random times, min separation 20
  cluster2: 2 AOs only 3 steps apart, 5 sigma            (adjacent masking)
  mixmag  : 6-sigma AO + 4-sigma IO, min separation 20   (large masks small)

Metrics (R = 1000 per scenario, T = 250, moderate-correlation Sigma):
  recall    : share of planted outliers matched by a detection within +/-1
  precision : share of detections matching some planted outlier within +/-1
  type_acc  : correct AO/IO label among matched detections
  param MAE : mean |phi_hat - phi_true| over the 6 parameters for
              (naive contaminated fit) vs (our cleaned fit) vs (oracle fit
              on the uncontaminated series) — the paper's argument that the
              procedure repairs estimation bias, not just flags points.
"""

import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gstar_outlier import (  # noqa: E402
    fit_gstar, inject_outlier, iterative_detection, make_M, simulate_gstar,
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
TRUE = np.concatenate([PHI0, PHI1])
ALPHA = 0.05
T = 250
R = 1000
SCENARIOS = ["spread3", "cluster2", "mixmag"]


def plant(scenario: str, rng: np.random.Generator):
    """Return list of (t0, loc, magnitude_sigma, kind)."""
    def sep_times(k, min_sep):
        while True:
            ts = np.sort(rng.integers(15, T - 15, size=k))
            if k == 1 or np.min(np.diff(ts)) >= min_sep:
                return [int(t) for t in ts]

    if scenario == "spread3":
        ts = sep_times(3, 20)
        kinds = ["AO", "IO", "AO"]
        mags = [5.0, 5.0, 5.0]
    elif scenario == "cluster2":
        t1 = int(rng.integers(15, T - 20))
        ts = [t1, t1 + 3]
        kinds = ["AO", "AO"]
        mags = [5.0, 5.0]
    elif scenario == "mixmag":
        ts = sep_times(2, 20)
        kinds = ["AO", "IO"]
        mags = [6.0, 4.0]
    else:
        raise ValueError(scenario)
    locs = [int(rng.integers(0, 3)) for _ in ts]
    return list(zip(ts, locs, mags, kinds))


def run_one(args):
    scenario, r = args
    rng = np.random.default_rng(hash((scenario, r)) % (2**32))
    Z = simulate_gstar(PHI0, PHI1, W, SIGMA, T, rng=rng)
    truths = plant(scenario, rng)
    Psi = Z.copy()
    for t0, loc, mag, kind in truths:
        omega = np.zeros(3)
        omega[loc] = mag * np.sqrt(SIGMA[loc, loc])
        Psi = inject_outlier(Psi, M, t0, omega, kind)

    res = iterative_detection(Psi, W, alpha=ALPHA)

    # greedy match detections to truths within +/-1
    matched_truth = [False] * len(truths)
    matched_type = 0
    n_match = 0
    for d in res.detections:
        ts = d.t + 1
        best, best_dist = -1, 2
        for j, (t0, _, _, kind) in enumerate(truths):
            if not matched_truth[j] and abs(ts - t0) <= 1 and abs(ts - t0) < best_dist:
                best, best_dist = j, abs(ts - t0)
        if best >= 0:
            matched_truth[best] = True
            n_match += 1
            matched_type += d.kind == truths[best][3]

    recall = sum(matched_truth) / len(truths)
    precision = n_match / len(res.detections) if res.detections else 1.0

    naive = fit_gstar(Psi, W)
    oracle = fit_gstar(Z, W)
    mae_naive = float(np.mean(np.abs(np.concatenate([naive.phi0, naive.phi1]) - TRUE)))
    mae_clean = float(np.mean(np.abs(np.concatenate([res.fit.phi0, res.fit.phi1]) - TRUE)))
    mae_oracle = float(np.mean(np.abs(np.concatenate([oracle.phi0, oracle.phi1]) - TRUE)))
    return (scenario, recall, precision, n_match, matched_type,
            mae_naive, mae_clean, mae_oracle)


def main():
    t_start = time.time()
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "exp04_masking_joint_20260703.csv"

    jobs = [(s, r) for s in SCENARIOS for r in range(R)]
    workers = max(1, mp.cpu_count() - 2)
    print(f"jobs: {len(jobs)}, workers={workers}", flush=True)
    with mp.Pool(workers) as pool:
        results = pool.map(run_one, jobs, chunksize=25)

    agg = {}
    for s, rec, prec, nm, mt, mn, mc, mo in results:
        a = agg.setdefault(s, [0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0])
        a[0] += 1; a[1] += rec; a[2] += prec; a[3] += nm; a[4] += mt
        a[5] += mn; a[6] += mc; a[7] += mo

    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["scenario", "R", "recall", "precision",
                       "type_acc_given_match", "param_MAE_naive",
                       "param_MAE_cleaned", "param_MAE_oracle"])
        for s in SCENARIOS:
            n, rec, prec, nm, mt, mn, mc, mo = agg[s]
            wcsv.writerow([s, n, rec / n, prec / n, mt / max(nm, 1),
                           mn / n, mc / n, mo / n])

    print(f"\n=== exp04: T={T}, alpha={ALPHA}, R={R} per scenario ===")
    print(f"{'scenario':<10}{'recall':>8}{'precision':>11}{'type|match':>12}"
          f"{'MAE naive':>11}{'MAE clean':>11}{'MAE oracle':>12}")
    for s in SCENARIOS:
        n, rec, prec, nm, mt, mn, mc, mo = agg[s]
        print(f"{s:<10}{rec/n:>8.3f}{prec/n:>11.3f}{mt/max(nm,1):>12.3f}"
              f"{mn/n:>11.4f}{mc/n:>11.4f}{mo/n:>12.4f}")
    print(f"\nsaved -> {out_csv}")
    print(f"elapsed: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
