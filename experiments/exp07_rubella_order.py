"""exp07 - order identification for the rubella application (Section 5.1).

Reads the assembled 8-region weekly rubella panel and reports the
space-time ACF/STACF and PACF/STPACF (Pfeifer & Deutsch 1980), the
multivariate Ljung-Box (Hosking) residual whiteness, and a per-location
added-lag-2 F-test. On the log-differenced series the STACF cuts off
after temporal lag 1, supporting the GSTAR(1;1) specification used in the
application.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gstar_outlier import fit_gstar, spectral_radius  # noqa: E402
from gstar_outlier.identification import (  # noqa: E402
    added_lag2_test, hosking_portmanteau, stacf, stpacf, stpacf_se,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "rubella_japan_2012_2022.csv"
REGIONS = {"Hokkaido": [0], "Tohoku": list(range(1, 7)), "Kanto": list(range(7, 14)),
           "Chubu": list(range(14, 23)), "Kinki": list(range(23, 30)),
           "Chugoku": list(range(30, 35)), "Shikoku": list(range(35, 39)),
           "Kyushu-Ok": list(range(39, 47))}
ADJ = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5, 6], 5: [4, 6, 7],
       6: [4, 5, 7], 7: [5, 6]}

rows = list(csv.reader(open(DATA, encoding="utf-8")))
M = np.array([[int(x) for x in r[1:]] for r in rows[1:]], float)
R = np.column_stack([M[:, ix].sum(1) for ix in REGIONS.values()])
N = len(REGIONS)
W = np.zeros((N, N))
for i, ns in ADJ.items():
    for j in ns:
        W[i, j] = 1.0
W /= W.sum(1, keepdims=True)
I = np.eye(N)

X = np.diff(np.log1p(R), axis=0)   # log-differenced growth rates
se = stpacf_se(len(X), N)
ac = stacf(X, [I, W], max_lag=10)
sp = stpacf(X, W, max_lag=5)

print("=== Rubella order identification (8 regions, weekly, log-diff) ===")
print(f"T = {len(X)} growth-rate observations, N = {N} regions\n")
print("STACF spatial-lag 0, rho_0(s), s = 0..10:")
print("  " + "  ".join(f"{v:+.2f}" for v in ac[0]))
print("STACF spatial-lag 1, rho_1(s), s = 0..10:")
print("  " + "  ".join(f"{v:+.2f}" for v in ac[1]))
print(f"\nSTPACF (2-sigma bound +/- {se:.3f}):")
for k, (a, b) in sp.items():
    fa = "*" if abs(a) > se else " "
    fb = "*" if abs(b) > se else " "
    print(f"  lag {k}: own = {a:+.3f}{fa}   spatial = {b:+.3f}{fb}")

fit = fit_gstar(X, W)
print(f"\nfitted spectral radius: {spectral_radius(fit.M):.3f}")
print("\nMultivariate Ljung-Box (Hosking) on GSTAR(1;1) residuals:")
for s, Q, df, pv in hosking_portmanteau(fit.residuals, h=8):
    print(f"  up to lag {s}: Q = {Q:8.1f}, df = {df:3d}, p = {pv:.3f}")
print("\nPer-location added-lag-2 F-test (large p => lag 2 not needed):")
prov = list(REGIONS)
for i, F, pv in added_lag2_test(X, W):
    print(f"  {prov[i]:<11}: F = {F:6.2f}, p = {pv:.3f}")
