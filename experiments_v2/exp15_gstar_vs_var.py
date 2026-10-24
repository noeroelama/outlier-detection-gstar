"""exp15 - does the GSTAR restriction matter? GSTAR fit vs unrestricted VAR fit.

GSTAR(1;1) restricts the VAR(1) coefficient matrix to M = Phi0 + Phi1 W with
Phi0, Phi1 diagonal: 2N coefficients instead of N^2. Both detectors below use
the same full-covariance statistics and Algorithm 1; they differ only in how M
(and therefore the residuals and Sigma-hat) is estimated.

Data: GSTAR(1;1), moderate error correlation (designs.py). N in {8, 16, 32},
T in {60, 120, 250}. For each cell:
  - estimation error of M: mean over series of ||M_hat - M||_F / N
  - FWER at the nominal threshold (R0 = 500 outlier-free series) and the
    size-adjusted threshold (95% quantile of the first-scan maximum)
  - size-adjusted power for one single-location AO or IO of 4 sd (R = 300)
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

R0, R, DELTA, LEVEL = 500, 300, 4.0, "moderate"
FITS = ["gstar", "var"]


def feasible(N, T, f):
    """Residual covariance must be nonsingular: T-1-(regressors per equation) > N."""
    k = 2 if f == "gstar" else N
    return (T - 1 - k) > N


def one(cfg):
    N, T = cfg
    C, W, Sigma, Ms = D.design(N, LEVEL)
    rng = np.random.default_rng(zlib.crc32(repr(cfg).encode()))
    sd = np.sqrt(np.diag(Sigma))
    mx = {f: [] for f in FITS}
    err = {f: [] for f in FITS}
    for _ in range(R0):
        Z = G.simulate_gstar_p(Ms, Sigma, T, rng=rng)
        for f in FITS:
            if not feasible(N, T, f):
                continue
            fit = G.fit_gstar_p(Z, W, 1, f)
            err[f].append(np.linalg.norm(fit.Ms[0] - Ms[0]) / N)
            a, i, *_ = G.scan(fit.resid, fit.Ms, fit.Sigma, "full")
            mx[f].append(max(a.max(), i.max()))
    c_nom = G.threshold("full", N, T - 1)
    cadj = {f: float(np.quantile(mx[f], 0.95)) if mx[f] else np.nan for f in FITS}
    out = []
    power = {(f, k): 0 for f in FITS for k in ("AO", "IO")}
    for kind in ("AO", "IO"):
        for _ in range(R):
            Z = G.simulate_gstar_p(Ms, Sigma, T, rng=rng)
            t0 = int(rng.integers(10, T - 10))
            j = int(rng.integers(N))
            om = np.zeros(N)
            om[j] = DELTA * sd[j]
            Psi = G.inject(Z, Ms, t0, om, kind)
            for f in FITS:
                if not feasible(N, T, f):
                    continue
                res = G.iterative(Psi, W, 1, "full", f, c2=cadj[f])
                power[(f, kind)] += any(abs(d.t - t0) <= 1 for d in res.detections)
    for f in FITS:
        if not feasible(N, T, f):
            out.append(dict(N=N, T=T, fit=f, n_coef=(2 * N if f == "gstar" else N * N), M_error=np.nan,
                            fwer_nominal=np.nan, c2_nominal=c_nom, c2_size_adjusted=np.nan,
                            power_AO=np.nan, power_IO=np.nan))
            continue
        out.append(dict(N=N, T=T, fit=f, n_coef=(2 * N if f == "gstar" else N * N),
                        M_error=float(np.mean(err[f])), fwer_nominal=float(np.mean(np.array(mx[f]) > c_nom)),
                        c2_nominal=c_nom, c2_size_adjusted=cadj[f],
                        power_AO=power[(f, "AO")] / R, power_IO=power[(f, "IO")] / R))
    return out


if __name__ == "__main__":
    t0 = time.time()
    grid = list(itertools.product([8, 16, 32], [60, 120, 250]))
    rows = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(one, grid):
            rows += r
            print(r[0]["N"], r[0]["T"], f"{time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["N"], r["T"], r["fit"]))
    with open(RESULTS / "exp15_gstar_vs_var.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("done", time.time() - t0)
