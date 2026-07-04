"""exp08 — application: rubella outbreak detection in Japan (8 regions, weekly).

Data: weekly notifiable rubella cases, 47 prefectures 2012-W37..2022-W52
(JIHS/NIID NESID archive), aggregated to the 8 standard Japanese regions.
Transform: log1p then first difference (STACF cuts off at temporal lag 1;
STPACF tails off -> GSTAR(1;1) on the differenced series, the same
STACF/STPACF order-identification route used in the GSTAR literature).

Rubella has a low endemic baseline (2014-2017, 2020-2022: <230 cases/yr
nationally) punctuated by the 2012-2013 and 2018-2019 epidemics -- a
baseline-plus-outbreak structure in which outbreaks are genuine
space-time outliers (unlike wave-dominated COVID/influenza series).

Localization: a quadratic statistic flags a TIME; we report the affected
region two ways -- by the largest standardized t-statistic (relative
surprise) and by the largest absolute effect |omega_hat| (raw log-growth
magnitude) -- because covariance standardization can otherwise highlight
low-incidence regions where a small absolute change is a large relative one.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gstar_outlier import fit_gstar, iterative_detection, spectral_radius  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "rubella_japan_2012_2022.csv"
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

import csv
rows = list(csv.reader(open(DATA, encoding="utf-8")))
weeks = [r[0] for r in rows[1:]]
M = np.array([[int(x) for x in r[1:]] for r in rows[1:]], float)  # T x 47
REG = {"Hokkaido": [0], "Tohoku": list(range(1, 7)), "Kanto": list(range(7, 14)),
       "Chubu": list(range(14, 23)), "Kinki": list(range(23, 30)),
       "Chugoku": list(range(30, 35)), "Shikoku": list(range(35, 39)),
       "Kyushu-Ok": list(range(39, 47))}
names = list(REG)
Rcount = np.column_stack([M[:, ix].sum(1) for ix in REG.values()])  # T x 8
N = 8

ADJ = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5, 6],
       5: [4, 6, 7], 6: [4, 5, 7], 7: [5, 6]}
W = np.zeros((N, N))
for i, ns in ADJ.items():
    for j in ns:
        W[i, j] = 1
W /= W.sum(1, keepdims=True)

X = np.diff(np.log1p(Rcount), axis=0)         # T-1 x 8 growth rates
dwk = weeks[1:]                                # dates aligned to X

fit = fit_gstar(X, W)
res = iterative_detection(X, W, alpha=0.05)

report = []
p = report.append
p("=== exp08: rubella outbreak detection, Japan 8 regions, weekly ===")
p(f"period {dwk[0]}..{dwk[-1]}  (T={len(X)} growth-rate obs)")
p(f"spectral radius of fitted M: {spectral_radius(fit.M):.3f}")
p(f"Bonferroni c^2 (alpha=0.05): {res.c2:.1f}")
p(f"detections: {len(res.detections)}\n")

OUTBREAK = ("2012", "2013", "2018", "2019")
n_ob = 0
p(f"{'week':<10}{'type':<5}{'std-region':<11}{'abs-region':<11}{'|omega|max':>10}  context (regional cases, det. week)")
for d in sorted(res.detections, key=lambda d: d.t):
    wk = dwk[d.t]
    i_std = int(np.argmax(np.abs(d.tstats)))
    i_abs = int(np.argmax(np.abs(d.omega)))
    yr = wk[:4]
    ob = yr in OUTBREAK
    n_ob += ob
    # raw case context: regional counts at the detected week (t index in Rcount is d.t+1)
    ctx_i = d.t + 1
    top_counts = sorted(zip(names, Rcount[ctx_i].astype(int)), key=lambda kv: -kv[1])[:3]
    ctx = ", ".join(f"{nm}={c}" for nm, c in top_counts if c > 0)
    p(f"{wk:<10}{d.kind:<5}{names[i_std]:<11}{names[i_abs]:<11}{np.abs(d.omega).max():>10.2f}  {ctx}")
p(f"\ndetections in outbreak years {OUTBREAK}: {n_ob}/{len(res.detections)}")

# how often absolute-effect localization points to the true high-incidence regions
HI = {"Kanto", "Kinki", "Chubu", "Kyushu-Ok"}
abs_hi = sum(names[int(np.argmax(np.abs(d.omega)))] in HI for d in res.detections)
std_hi = sum(names[int(np.argmax(np.abs(d.tstats)))] in HI for d in res.detections)
p(f"localized to a high-incidence region (Kanto/Kinki/Chubu/Kyushu): "
  f"absolute-effect {abs_hi}/{len(res.detections)}, standardized {std_hi}/{len(res.detections)}")

text = "\n".join(report)
print(text)
(OUT / "exp08_rubella_japan_20260703.txt").write_text(text, encoding="utf-8")
print("\nsaved ->", OUT / "exp08_rubella_japan_20260703.txt")
