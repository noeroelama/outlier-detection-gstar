"""Regenerate the four figures used in the paper (relative paths).

fig1_rubella          - 8-region weekly rubella series with epidemics shaded
fig2_rubella_detection- detection statistic over time vs Bonferroni threshold
fig3_power            - detection power vs magnitude (from results/exp02 CSV)
fig4_comparison       - multivariate vs per-location power (from results/exp03 CSV)

Figures are written to ../figures/. exp02 and exp03 must have been run
(their result CSVs are shipped in results/).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gstar_outlier import fit_gstar, iterative_detection  # noqa: E402
from gstar_outlier.detection import bonferroni_c2, outlier_statistics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "rubella_japan_2012_2022.csv"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#777777", "#111111"]

# ---- rubella panel ----
rows = list(csv.reader(open(DATA, encoding="utf-8")))
weeks = [r[0] for r in rows[1:]]
M = np.array([[int(x) for x in r[1:]] for r in rows[1:]], float)
REG = {"Hokkaido": [0], "Tohoku": list(range(1, 7)), "Kanto": list(range(7, 14)),
       "Chubu": list(range(14, 23)), "Kinki": list(range(23, 30)),
       "Chugoku": list(range(30, 35)), "Shikoku": list(range(35, 39)),
       "Kyushu-Ok": list(range(39, 47))}
names = list(REG); R = np.column_stack([M[:, ix].sum(1) for ix in REG.values()])
N = 8
ADJ = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5, 6], 5: [4, 6, 7], 6: [4, 5, 7], 7: [5, 6]}
W = np.zeros((N, N))
for i, ns in ADJ.items():
    for j in ns:
        W[i, j] = 1
W /= W.sum(1, keepdims=True)
X = np.diff(np.log1p(R), axis=0)
res = iterative_detection(X, W, alpha=0.05)
yf = lambda w: int(w.split("-W")[0]) + (int(w.split("-W")[1]) - 1) / 52.0
xt = np.array([yf(w) for w in weeks]); EPI = [(2012.7, 2014.0), (2018.0, 2019.6)]

# fig1
det_reg = {i: [] for i in range(N)}
for d in res.detections:
    det_reg[int(np.argmax(np.abs(d.tstats)))].append(d.t + 1)
fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.6), sharex=True); axes = axes.ravel()
for i, nm in enumerate(names):
    ax = axes[i]
    for a, b in EPI:
        ax.axvspan(a, b, color="#f0d0a0", alpha=.35, lw=0)
    ax.plot(xt, R[:, i], lw=.7, color=C[i])
    for ci in det_reg[i]:
        ax.plot(xt[ci], R[ci, i], "v", color="red", ms=5)
    ax.set_title(f"{nm} (total {int(R[:, i].sum())})", fontsize=8.5, loc="left")
    ax.set_ylabel("cases/wk", fontsize=8); ax.tick_params(labelsize=7.5)
fig.tight_layout(); fig.savefig(FIGS / "fig1_rubella.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig1_rubella.png", bbox_inches="tight"); plt.close(fig)

# fig2
fit = fit_gstar(X, W); resid = X[1:] - X[:-1] @ fit.M.T
a2, i2, *_ = outlier_statistics(resid, fit.M, fit.Sigma); lam2 = np.maximum(a2, i2)
rd = xt[2:]; c2 = bonferroni_c2(N, len(resid), 0.05); ymax = lam2.max() * 1.18
fig, ax = plt.subplots(figsize=(7.2, 3.1))
for a, b in EPI:
    ax.axvspan(a, b, color="#f0d0a0", alpha=.45, lw=0)
ax.plot(rd, lam2, lw=.7, color=C[0])
ax.axhline(c2, color=C[1], ls="--", lw=1.1, label=r"Bonferroni threshold ($\alpha=0.05$)")
ax.plot(rd[lam2 > c2], lam2[lam2 > c2], "o", ms=4, mfc="none", mec="red", mew=1.1)
ax.set_ylim(0, ymax)
ax.text(2013.35, ymax * 0.97, "2012-13 epidemic", fontsize=8.5, ha="center", va="top", color="#8a5a00")
ax.text(2018.8, ymax * 0.97, "2018-19 epidemic", fontsize=8.5, ha="center", va="top", color="#8a5a00")
ax.set_ylabel(r"$\max(\hat\lambda^2_{AO,t},\hat\lambda^2_{IO,t})$"); ax.set_xlabel("year")
ax.legend(frameon=False, loc="upper center", fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "fig2_rubella_detection.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig2_rubella_detection.png", bbox_inches="tight"); plt.close(fig)

# fig3 power (from exp02 CSV)
rows = [r for r in csv.reader(open(RESULTS / "exp02_size_power_20260703.csv")) if r and r[0] in ("AO", "IO")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw={"wspace": 0.1})
for k, ax, title in [("AO", axes[0], "(a) Additive outliers"), ("IO", axes[1], "(b) Innovative outliers")]:
    for j, T in enumerate([100, 250, 365]):
        pts = sorted((float(r[1]), float(r[4])) for r in rows if r[0] == k and int(r[2]) == T)
        ax.plot([m for m, _ in pts], [p for _, p in pts], "o-", ms=3.5, lw=1.1, color=C[j], label=f"$T={T}$")
    ax.axhline(0.05, color="grey", lw=0.6, ls=":"); ax.set_xlabel(r"magnitude ($\times\sigma$)")
    ax.set_xticks([3, 4, 5, 6]); ax.set_title(title, loc="left")
axes[0].set_ylabel("detection power"); axes[0].set_ylim(0, 1.02)
axes[0].legend(frameon=False, loc="upper left")
fig.savefig(FIGS / "fig3_power.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig3_power.png", bbox_inches="tight"); plt.close(fig)

# fig4 comparison (from exp03 CSV)
rows = [r for r in csv.reader(open(RESULTS / "exp03_vs_perloc_20260703.csv")) if r and r[0] in ("moderate", "high")]
groups = [("moderate", "AO"), ("moderate", "IO"), ("high", "AO"), ("high", "IO")]
xp = np.arange(len(groups)); width = 0.36
fig, ax = plt.subplots(figsize=(4.6, 2.8))
for k, (det, color, lbl) in enumerate([("multivariate", C[0], "multivariate (proposed)"),
                                       ("per-location", C[4], "per-location benchmark")]):
    vals = [float([r[4] for r in rows if r[0] == g and r[1] == det and r[2] == kind][0]) for g, kind in groups]
    bars = ax.bar(xp + (k - 0.5) * width, vals, width, color=color, label=lbl)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(xp); ax.set_xticklabels(["AO\nmoderate", "IO\nmoderate", "AO\nhigh", "IO\nhigh"])
ax.set_ylabel("detection power"); ax.set_ylim(0, 1.1); ax.legend(frameon=False, loc="upper left")
fig.savefig(FIGS / "fig4_comparison.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig4_comparison.png", bbox_inches="tight"); plt.close(fig)
print("wrote fig1_rubella, fig2_rubella_detection, fig3_power, fig4_comparison to", FIGS)
