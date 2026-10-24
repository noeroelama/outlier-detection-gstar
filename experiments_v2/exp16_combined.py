"""exp16 - combined test: full-covariance OR per-location maximum (each at level alpha/2).

The combined detector flags a time point when either the full-covariance statistic
exceeds its Bonferroni threshold at level alpha/2, or the per-location maximum
statistic exceeds its Bonferroni threshold at level alpha/2. By the union bound its
FWER for one scan with known parameters is at most alpha. The AO/IO type is taken
from the full-covariance likelihood-ratio statistics.

The random seeds and the order of random draws are the same as in exp10, exp11,
exp12 and exp20, so the combined detector is evaluated on exactly the same series
as the other detectors (paired comparison).

Parts: A - FWER (exp10 Gaussian grid, p = 1 and p = 2); B - power (exp11 design,
nominal and size-adjusted); C - adjacent outliers (exp12); D - rainfall application
and semi-synthetic power (exp20).
"""
import csv
import itertools
import json
import time
import zlib
from multiprocessing import Pool

import numpy as np

from _common import RESULTS
import designs as D
from gstar_outlier import general as G

DET = "combo"


# ---------------------------------------------------------------- A: FWER
def part_a(cfg):
    N, T, level, errs, p = cfg
    C, W, Sigma, Ms = D.design(N, level)
    if p == 2:
        Ms = np.stack([0.6 * Ms[0], 0.3 * Ms[0]])
    rng = np.random.default_rng(zlib.crc32(repr(cfg).encode()))
    mx = []
    for _ in range(1000):
        Z = G.simulate_gstar_p(Ms, Sigma, T, rng=rng, df=None)
        mx.append(G.initial_max(Z, W, p, DET, "gstar")[0])
    mx = np.array(mx)
    return dict(N=N, T=T, corr=level, errors=errs, p=p, fwer=float(np.mean(mx > 1.0)))


# ---------------------------------------------------------------- B: power
N8, T250 = 8, 250


def null_q(level):
    C, W, Sigma, Ms = D.design(N8, level)
    rng = np.random.default_rng(zlib.crc32(f"null-{level}".encode()))
    mx = []
    for _ in range(1000):
        Z = G.simulate_gstar_p(Ms, Sigma, T250, rng=rng)
        mx.append(G.initial_max(Z, W, 1, DET, "gstar")[0])
    mx = np.array(mx)
    return level, float(np.quantile(mx, 0.95)), float(np.mean(mx > 1.0))


def part_b(args):
    (level, pattern, kind, delta), cadj = args
    C, W, Sigma, Ms = D.design(N8, level)
    rng = np.random.default_rng(zlib.crc32(repr((level, pattern, kind, delta)).encode()))
    sd = np.sqrt(np.diag(Sigma))
    acc = {m: dict(hit=0, type_ok=0, false=0) for m in ("nominal", "adjusted")}
    for _ in range(500):
        Z = G.simulate_gstar_p(Ms, Sigma, T250, rng=rng)
        t0 = int(rng.integers(20, T250 - 20))
        j = int(rng.integers(N8))
        idx = {"single": [j], "cluster": [j] + D.neighbours(C, j, 2), "global": list(range(N8))}[pattern]
        omega = np.zeros(N8)
        omega[idx] = delta * sd[idx]
        Psi = G.inject(Z, Ms, t0, omega, kind)
        for m in acc:
            res = G.iterative(Psi, W, 1, DET, "gstar", c2=None if m == "nominal" else cadj)
            near = [d for d in res.detections if abs(d.t - t0) <= 1]
            acc[m]["false"] += len(res.detections) - len(near)
            if near:
                acc[m]["hit"] += 1
                best = max(near, key=lambda d: max(d.stat_ao, d.stat_io))
                acc[m]["type_ok"] += best.kind == kind
    return [dict(corr=level, pattern=pattern, kind=kind, delta=delta, detector=DET, fit="gstar", threshold=m,
                 power=a["hit"] / 500, type_acc=a["type_ok"] / a["hit"] if a["hit"] else np.nan,
                 false_per_series=a["false"] / 500) for m, a in acc.items()]


# ---------------------------------------------------------------- C: adjacent
def part_c(scen):
    from exp12_adjacent import N, T, R, LEVEL
    name, spec = scen
    C, W, Sigma, Ms = D.design(N, LEVEL)
    sd = np.sqrt(np.diag(Sigma))
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    tp = ndet = type_ok = ntrue = 0
    for _ in range(R):
        Z = G.simulate_gstar_p(Ms, Sigma, T, rng=rng)
        t0 = int(rng.integers(20, T - 45))
        j = int(rng.integers(N))
        nb = D.neighbours(C, j, 1)[0]
        truth, Psi = {}, Z
        for off, where, kind, mag in spec:
            loc = j if where == "j" else nb
            om = np.zeros(N)
            om[loc] = mag * sd[loc]
            Psi = G.inject(Psi, Ms, t0 + off, om, kind)
            truth[t0 + off] = kind
        res = G.iterative(Psi, W, 1, DET, "gstar")
        ntrue += len(truth)
        ndet += len(res.detections)
        for d in res.detections:
            if d.t in truth:
                tp += 1
                type_ok += d.kind == truth[d.t]
    return dict(scenario=name, detector=DET, recall=tp / ntrue, precision=tp / ndet if ndet else np.nan,
                type_acc=type_ok / tp if tp else np.nan)


# ---------------------------------------------------------------- D: rainfall
def part_d(_):
    from _rain import load_rain, anomalies, W_rain
    Y, idx, cen = load_rain()
    Y = Y.loc[:"2024-12"]
    A = anomalies(Y)
    W = W_rain(cen)
    T, N = A.shape
    C = cen[["lat", "lon"]].values
    full = G.iterative(A, W, 1, "full", max_outliers=60, max_sweeps=15)
    comb = G.iterative(A, W, 1, DET, max_outliers=60, max_sweeps=15)
    out = dict(full=sorted(Y.index[d.t] + " " + d.kind for d in full.detections),
               combo=sorted(Y.index[d.t] + " " + d.kind for d in comb.detections),
               combo_converged=comb.converged, combo_hit_cap=comb.hit_cap)
    # bootstrap FWER at application size (same seed as exp20)
    rng = np.random.default_rng(zlib.crc32(b"rain-boot"))
    mx = []
    for _ in range(500):
        Zb = G.simulate_gstar_p(full.fit.Ms, full.fit.Sigma, T, rng=rng)
        G.initial_max(Zb, W, 1, "full", "gstar")              # keep the draw order of exp20
        G.initial_max(Zb, W, 1, "perloc", "gstar")
        mx.append(G.initial_max(Zb, W, 1, DET, "gstar")[0])
    out["bootstrap_fwer_combo"] = float(np.mean(np.array(mx) > 1.0))
    # semi-synthetic (same seeds and draw order as exp20)
    Aclean = G._clean(A, full.detections, full.fit.Ms)
    sd = np.sqrt(np.diag(full.fit.Sigma))
    semi = []
    for pattern in ("single", "cluster"):
        for kind in ("AO", "IO"):
            for delta in (3, 4, 5):
                rng = np.random.default_rng(zlib.crc32(repr((pattern, kind, delta)).encode()))
                hit = false = 0
                for _ in range(300):
                    t0 = int(rng.integers(12, T - 12))
                    j = int(rng.integers(N))
                    if pattern == "single":
                        ix = [j]
                    else:
                        lat, lon = np.radians(C[:, 0]), np.radians(C[:, 1])
                        dd = np.arccos(np.clip(np.sin(lat[j]) * np.sin(lat) + np.cos(lat[j]) * np.cos(lat) * np.cos(lon[j] - lon), -1, 1))
                        ix = list(np.argsort(dd)[:3])
                    om = np.zeros(N)
                    om[ix] = delta * sd[ix]
                    r = G.iterative(G.inject(Aclean, full.fit.Ms, t0, om, kind), W, 1, DET, max_outliers=60, max_sweeps=15)
                    near = [d for d in r.detections if abs(d.t - t0) <= 1]
                    hit += bool(near)
                    false += len(r.detections) - len(near)
                semi.append(dict(pattern=pattern, kind=kind, delta=delta, power_combo=hit / 300, false_combo=false / 300))
    out["semi_synthetic_combo"] = semi
    return out


if __name__ == "__main__":
    t0 = time.time()
    gridA = list(itertools.product([3, 8, 16], [100, 250, 500], ["low", "high"], ["gauss"], [1]))
    gridA += [(8, T, lv, "gauss", 2) for T in (100, 250, 500) for lv in ("low", "high")]
    with Pool(2) as pool:
        resA = pool.map(part_a, gridA)
        print("A done", time.time() - t0, flush=True)
        nq = pool.map(null_q, ["low", "moderate", "high"])
        cadj = {lv: q for lv, q, _ in nq}
        gridB = list(itertools.product(["low", "moderate", "high"], ["single", "cluster", "global"], ["AO", "IO"], [3, 4, 5]))
        resB = sum(pool.map(part_b, [(g, cadj[g[0]]) for g in gridB]), [])
        print("B done", time.time() - t0, flush=True)
        from exp12_adjacent import SCEN
        resC = pool.map(part_c, SCEN)
        print("C done", time.time() - t0, flush=True)
    resD = part_d(None)
    print("D done", time.time() - t0, flush=True)
    for name, rows in [("exp16_fwer.csv", resA), ("exp16_power.csv", resB), ("exp16_adjacent.csv", resC)]:
        with open(RESULTS / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    resD["null_thresholds"] = [dict(corr=lv, c_size_adjusted=q, fwer_nominal=fw) for lv, q, fw in nq]
    (RESULTS / "exp16_rainfall.json").write_text(json.dumps(resD, indent=1, default=str))
    print("saved", time.time() - t0)
