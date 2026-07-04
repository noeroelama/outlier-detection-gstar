"""exp06 — GSTAR detector vs unrestricted-VAR(1) detector (Tsay-Pena-Pankratz style).

The referee question this answers: "Why not simply run multivariate outlier
detection (Tsay, Pena & Pankratz 2000, Biometrika) on the unrestricted
VAR(1) instead of imposing the GSTAR structure?"

Both detectors share identical signatures, GLS statistics, Bonferroni FWER
control, and sweep architecture; they differ ONLY in the fitted coefficient
matrix: GSTAR (2N parameters, W known) vs unrestricted VAR(1) (N^2
parameters). With N = 6 and short T, the VAR wastes degrees of freedom on
30 extra parameters, inflating estimation noise in M-hat and Sigma-hat.

Honesty check — run under TWO data-generating processes:
  dgp=gstar : Z_t = (Phi0 + Phi1 W) Z_{t-1} + eps   (GSTAR restriction true)
  dgp=var   : Z_t = Phi_full Z_{t-1} + eps, Phi_full NOT of GSTAR form
              (GSTAR detector misspecified — measures the cost of a wrong
              restriction, not just the benefit of a right one)

Design: N = 6 (real Java inverse-distance W), T in {100, 250}, single
outlier (AO or IO), 4 sigma, random time/location, R = 1000 per cell,
alpha = 0.05; clean-series size runs per (dgp, detector, T).
"""

import csv
import multiprocessing as mp
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gstar_outlier import (  # noqa: E402
    inject_outlier, iterative_detection, make_M, spectral_radius,
)
from gstar_outlier.model import fit_gstar, fit_var1  # noqa: E402

# ---- 6-province Java inverse-distance W (as in exp05) ----------------------
COORDS = [(-6.405817, 106.064018), (-6.211544, 106.845172),
          (-7.090911, 107.668887), (-7.150975, 110.140259),
          (-7.875385, 110.426209), (-7.536064, 112.238402)]
N = 6


def haversine(c1, c2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [*c1, *c2])
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            W[i, j] = 1.0 / haversine(COORDS[i], COORDS[j])
W /= W.sum(axis=1, keepdims=True)

PHI0 = np.array([0.35, -0.30, 0.25, 0.40, -0.25, 0.30])
PHI1 = np.array([0.25, 0.20, -0.20, 0.15, 0.20, -0.15])
M_GSTAR = make_M(PHI0, PHI1, W)

# non-GSTAR VAR coefficient: perturb off-diagonals, rescale to rho = 0.45
_rng = np.random.default_rng(777)
PERT = _rng.uniform(-0.15, 0.15, size=(N, N))
M_VAR = M_GSTAR + PERT - np.diag(np.diag(PERT))
M_VAR *= 0.45 / spectral_radius(M_VAR)

# moderate-correlation Sigma (exchangeable-ish, PD)
SIGMA = 0.6 * np.eye(N) + 0.4 * np.ones((N, N))

ALPHA = 0.05
R = 1000
TS = [100, 250]
DGPS = {"gstar": M_GSTAR, "var": M_VAR}
DETECTORS = {
    "GSTAR": fit_gstar,
    "VAR-TPP": fit_var1,
}


def simulate_var1(Mmat, Sigma, T, rng, burn=200):
    L = np.linalg.cholesky(Sigma)
    Z = np.zeros(len(Mmat))
    out = np.empty((T, len(Mmat)))
    for t in range(burn + T):
        Z = Mmat @ Z + L @ rng.standard_normal(len(Mmat))
        if t >= burn:
            out[t - burn] = Z
    return out


def run_power(args):
    dgp, det_name, kind, T, r = args
    Mmat = DGPS[dgp]
    rng = np.random.default_rng(hash((dgp, det_name, kind, T, r)) % (2**32))
    Z = simulate_var1(Mmat, SIGMA, T, rng)
    t0 = int(rng.integers(10, T - 10))
    loc = int(rng.integers(0, N))
    omega = np.zeros(N)
    omega[loc] = 4.0 * np.sqrt(SIGMA[loc, loc])
    Psi = inject_outlier(Z, Mmat, t0, omega, kind)
    res = iterative_detection(Psi, W, alpha=ALPHA, fit_fn=DETECTORS[det_name])
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
    return dgp, det_name, kind, T, hit, type_ok, loc_ok, spurious


def run_size(args):
    dgp, det_name, T, r = args
    Mmat = DGPS[dgp]
    rng = np.random.default_rng(hash((dgp, det_name, "clean", T, r)) % (2**32))
    Z = simulate_var1(Mmat, SIGMA, T, rng)
    res = iterative_detection(Z, W, alpha=ALPHA, fit_fn=DETECTORS[det_name])
    return dgp, det_name, T, len(res.detections)


def main():
    t_start = time.time()
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "exp06_vs_var_tpp_20260703.csv"

    print(f"spectral radii: M_GSTAR={spectral_radius(M_GSTAR):.3f}, "
          f"M_VAR={spectral_radius(M_VAR):.3f}", flush=True)
    power_jobs = [(g, d, k, T, r) for g in DGPS for d in DETECTORS
                  for k in ["AO", "IO"] for T in TS for r in range(R)]
    size_jobs = [(g, d, T, r) for g in DGPS for d in DETECTORS
                 for T in TS for r in range(R)]
    workers = max(1, mp.cpu_count() - 2)
    print(f"jobs: {len(power_jobs)} power + {len(size_jobs)} size, "
          f"workers={workers}", flush=True)

    with mp.Pool(workers) as pool:
        power_res = pool.map(run_power, power_jobs, chunksize=50)
        size_res = pool.map(run_size, size_jobs, chunksize=50)

    cells = {}
    for g, d, k, T, hit, tok, lok, spur in power_res:
        c = cells.setdefault((g, d, k, T), [0, 0, 0, 0, 0])
        c[0] += 1; c[1] += hit; c[2] += tok; c[3] += lok; c[4] += spur
    size = {}
    for g, d, T, ndet in size_res:
        s = size.setdefault((g, d, T), [0, 0])
        s[0] += 1; s[1] += ndet > 0

    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["dgp", "detector", "kind", "T", "R", "detect_rate",
                       "type_acc_given_hit", "loc_acc_given_hit",
                       "spurious_per_run", "clean_FWER"])
        for (g, d, k, T), (n, hit, tok, lok, spur) in sorted(cells.items()):
            fw = size[(g, d, T)][1] / size[(g, d, T)][0]
            wcsv.writerow([g, d, k, T, n, hit / n, tok / max(hit, 1),
                           lok / max(hit, 1), spur / n, fw])

    print(f"\n=== exp06: N=6, 4 sigma, alpha={ALPHA}, R={R}/cell ===")
    print(f"{'dgp':<7}{'detector':<9}{'kind':<5}{'T':>5}{'detect':>8}"
          f"{'type|hit':>10}{'loc|hit':>9}{'spur':>7}{'FWER':>7}")
    for (g, d, k, T), (n, hit, tok, lok, spur) in sorted(cells.items()):
        fw = size[(g, d, T)][1] / size[(g, d, T)][0]
        print(f"{g:<7}{d:<9}{k:<5}{T:>5}{hit/n:>8.3f}{tok/max(hit,1):>10.3f}"
              f"{lok/max(hit,1):>9.3f}{spur/n:>7.3f}{fw:>7.3f}")
    print(f"\nsaved -> {out_csv}")
    print(f"elapsed: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
