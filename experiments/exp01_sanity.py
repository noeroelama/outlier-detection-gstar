"""exp01 — sanity check of the corrected multivariate AO/IO detection.

Design (single replicate, fixed seed, then a small repetition study):
  * N = 3 locations, thesis weight matrix (COVID Java, inverse-distance)
  * T = 200, stationary diagonal parameters, correlated errors
  * plant one AO at t=60 (location 1) and one IO at t=140 (location 2),
    magnitude 5 * sqrt(Sigma_jj)
  * expect: both flagged, correct type, correct location via t-stats
  * clean-series run: expect no detections (size control)
  * 200-replicate repetition: detection rate, type accuracy, false alarms
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gstar_outlier import (  # noqa: E402
    fit_gstar, inject_outlier, iterative_detection, make_M, simulate_gstar,
    spectral_radius,
)

# ---- design ----------------------------------------------------------------
W = np.array([
    [0.0,    0.7777, 0.2223],
    [0.7288, 0.0,    0.2712],
    [0.4345, 0.5655, 0.0],
])
phi0 = np.array([0.40, -0.30, 0.25])
phi1 = np.array([0.30, 0.20, -0.20])
Sigma = np.array([
    [1.00, 0.50, 0.30],
    [0.50, 1.00, 0.40],
    [0.30, 0.40, 1.00],
])
M = make_M(phi0, phi1, W)
T = 200
AO_T, AO_LOC = 60, 0
IO_T, IO_LOC = 140, 1
MAG = 5.0

print(f"spectral radius of M: {spectral_radius(M):.3f}")

# ---- single replicate, fixed seed ------------------------------------------
rng = np.random.default_rng(20260703)
Z = simulate_gstar(phi0, phi1, W, Sigma, T, rng=rng)

omega_ao = np.zeros(3); omega_ao[AO_LOC] = MAG * np.sqrt(Sigma[AO_LOC, AO_LOC])
omega_io = np.zeros(3); omega_io[IO_LOC] = MAG * np.sqrt(Sigma[IO_LOC, IO_LOC])
Psi = inject_outlier(Z, M, AO_T, omega_ao, "AO")
Psi = inject_outlier(Psi, M, IO_T, omega_io, "IO")

res = iterative_detection(Psi, W, alpha=0.05)
print(f"\n--- contaminated series: {len(res.detections)} detection(s), "
      f"{res.n_sweeps} sweep(s), c2={res.c2:.2f} ---")
for d in res.detections:
    print(f"  {d.kind} at series t={d.t+1} | lam2_AO={d.lam2_ao:.1f} "
          f"lam2_IO={d.lam2_io:.1f} | omega={np.round(d.omega, 2)} "
          f"| t-stats={np.round(d.tstats, 1)}")
print("  truth: AO at t=60 loc 0 (omega_0=5.0), IO at t=140 loc 1 (omega_1=5.0)")

fit_clean_truth = fit_gstar(Z, W)
print(f"\n  phi0 true {phi0} | contaminated-then-cleaned fit {np.round(res.fit.phi0, 3)}"
      f" | clean-data fit {np.round(fit_clean_truth.phi0, 3)}")
print(f"  phi1 true {phi1} | contaminated-then-cleaned fit {np.round(res.fit.phi1, 3)}"
      f" | clean-data fit {np.round(fit_clean_truth.phi1, 3)}")

res0 = iterative_detection(Z, W, alpha=0.05)
print(f"\n--- clean series: {len(res0.detections)} detection(s) (expect 0) ---")

# ---- small repetition study -------------------------------------------------
R = 200
hits_ao = hits_io = type_ok_ao = type_ok_io = 0
false_alarms = 0          # detections on contaminated series away from truth
clean_fa_runs = 0         # clean-series runs with >=1 detection (empirical FWER)
loc_ok_ao = loc_ok_io = 0
for r in range(R):
    rng_r = np.random.default_rng(1000 + r)
    Zr = simulate_gstar(phi0, phi1, W, Sigma, T, rng=rng_r)
    Pr = inject_outlier(Zr, M, AO_T, omega_ao, "AO")
    Pr = inject_outlier(Pr, M, IO_T, omega_io, "IO")
    rr = iterative_detection(Pr, W, alpha=0.05)
    got_ao = got_io = False
    for d in rr.detections:
        ts = d.t + 1
        if abs(ts - AO_T) <= 1 and not got_ao:
            got_ao = True
            hits_ao += 1
            type_ok_ao += d.kind == "AO"
            loc_ok_ao += int(np.argmax(np.abs(d.tstats))) == AO_LOC
        elif abs(ts - IO_T) <= 1 and not got_io:
            got_io = True
            hits_io += 1
            type_ok_io += d.kind == "IO"
            loc_ok_io += int(np.argmax(np.abs(d.tstats))) == IO_LOC
        else:
            false_alarms += 1
    r0 = iterative_detection(Zr, W, alpha=0.05)
    clean_fa_runs += len(r0.detections) > 0

print(f"\n--- repetition study, R={R}, magnitude {MAG} sigma ---")
print(f"  AO detection rate (±1): {hits_ao/R:.3f} | correct type given hit: "
      f"{type_ok_ao/max(hits_ao,1):.3f} | correct location: {loc_ok_ao/max(hits_ao,1):.3f}")
print(f"  IO detection rate (±1): {hits_io/R:.3f} | correct type given hit: "
      f"{type_ok_io/max(hits_io,1):.3f} | correct location: {loc_ok_io/max(hits_io,1):.3f}")
print(f"  spurious detections per contaminated run: {false_alarms/R:.3f}")
print(f"  clean-series empirical FWER (target <= 0.05): {clean_fa_runs/R:.3f}")
