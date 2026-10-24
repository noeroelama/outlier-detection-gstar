"""exp12 - adjacent and overlapping outliers (R2-4).

The original submission separated two additive outliers by three time points;
for GSTAR(1;1) their residual signatures (two points each) then do not overlap.
Here the signatures overlap: consecutive outliers at the same or at
neighbouring locations, a patch of three consecutive AOs, an AO immediately
followed by an IO, and a mixed-magnitude pair.

N = 8, T = 250, moderate error correlation, Gaussian errors, nominal thresholds,
R = 500 series per scenario. A true outlier is recovered when a detection is
made at exactly its time (+-0, because outliers one step apart must be
separated). Precision = share of detections that sit exactly on a true outlier.
"""
import csv
import time
import zlib
from multiprocessing import Pool

import numpy as np

from _common import RESULTS
import designs as D
from gstar_outlier import general as G

N, T, R, LEVEL, DELTA = 8, 250, 500, "moderate", 5.0
DETS = [("full", "gstar"), ("perloc", "gstar")]

# (name, list of (time offset, location: 'j' or 'nb', kind, magnitude))
SCEN = [
    ("AO,AO same location, gap 3 (original design)", [(0, "j", "AO", DELTA), (3, "j", "AO", DELTA)]),
    ("AO,AO same location, gap 2", [(0, "j", "AO", DELTA), (2, "j", "AO", DELTA)]),
    ("AO,AO same location, consecutive", [(0, "j", "AO", DELTA), (1, "j", "AO", DELTA)]),
    ("AO patch, 3 consecutive, same location", [(0, "j", "AO", DELTA), (1, "j", "AO", DELTA), (2, "j", "AO", DELTA)]),
    ("AO,AO neighbouring locations, consecutive", [(0, "j", "AO", DELTA), (1, "nb", "AO", DELTA)]),
    ("AO then IO, same location, consecutive", [(0, "j", "AO", DELTA), (1, "j", "IO", DELTA)]),
    ("IO then AO, same location, consecutive", [(0, "j", "IO", DELTA), (1, "j", "AO", DELTA)]),
    ("6 sd AO and 4 sd IO, 20 apart", [(0, "j", "AO", 6.0), (20, "nb", "IO", 4.0)]),
]


def one(scen):
    name, spec = scen
    C, W, Sigma, Ms = D.design(N, LEVEL)
    sd = np.sqrt(np.diag(Sigma))
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    acc = {d: dict(tp=0, ndet=0, type_ok=0, ntrue=0) for d in DETS}
    for _ in range(R):
        Z = G.simulate_gstar_p(Ms, Sigma, T, rng=rng)
        t0 = int(rng.integers(20, T - 45))
        j = int(rng.integers(N))
        nb = D.neighbours(C, j, 1)[0]
        truth = {}
        Psi = Z
        for off, where, kind, mag in spec:
            loc = j if where == "j" else nb
            om = np.zeros(N)
            om[loc] = mag * sd[loc]
            Psi = G.inject(Psi, Ms, t0 + off, om, kind)
            truth[t0 + off] = kind
        for d in DETS:
            res = G.iterative(Psi, W, 1, d[0], d[1])
            a = acc[d]
            a["ntrue"] += len(truth)
            a["ndet"] += len(res.detections)
            for det in res.detections:
                if det.t in truth:
                    a["tp"] += 1
                    a["type_ok"] += det.kind == truth[det.t]
    rows = []
    for d, a in acc.items():
        rows.append(dict(scenario=name, detector=d[0], fit=d[1], recall=a["tp"] / a["ntrue"],
                         precision=a["tp"] / a["ndet"] if a["ndet"] else np.nan,
                         type_acc=a["type_ok"] / a["tp"] if a["tp"] else np.nan))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    out = []
    with Pool(2) as pool:
        for rows in pool.imap(one, SCEN):
            out += rows
            print(rows[0]["scenario"], f"{time.time()-t0:.0f}s", flush=True)
    with open(RESULTS / "exp12_adjacent.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
