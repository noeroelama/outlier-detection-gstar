"""Explanatory diagrams for the revised manuscript (figures_v2/).

figA_patterns - expected effect of one AO and one IO on the observations and on
                the residuals (no noise), for the location hit and a neighbour
figB_ellipse  - two locations with error correlation 0.95: the acceptance region
                of the full-covariance statistic (ellipse) and of the per-location
                maximum statistic (square), with two example outliers
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from scipy import stats

from _common import FIGS
from gstar_outlier import general as G

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#777777"


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure A
W = np.array([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0.]])          # 3 locations on a line
M = np.diag([0.5, 0.5, 0.5]) + np.diag([0.3, 0.3, 0.3]) @ W
Ms = M[None]
T, tau = 14, 4
omega = np.array([0., 3., 0.])                                  # outlier at location 2
t = np.arange(T)
fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.6), sharex=True, sharey="row")
for c, kind in enumerate(["AO", "IO"]):
    eff = G.inject(np.zeros((T, 3)), Ms, tau, omega, kind)      # effect on observations
    res = np.zeros((T, 3))
    res[1:] = eff[1:] - eff[:-1] @ M.T                          # effect on residuals e_t = Psi_t - M Psi_{t-1}
    for r, (Y, lab) in enumerate([(eff, "effect on the observations"), (res, "effect on the residuals")]):
        ax = axes[r, c]
        ax.axhline(0, color=GREY, lw=0.6)
        ax.vlines(t - 0.12, 0, Y[:, 1], color=BLUE, lw=2.2, label="location hit by the outlier")
        ax.vlines(t + 0.12, 0, Y[:, 0], color=ORANGE, lw=2.2, label="neighbouring location")
        ax.set_xticks([tau, tau + 5])
        ax.set_xticklabels([r"$\tau$", r"$\tau+5$"])
        ax.set_xlabel("time" if r == 1 else "")
        if c == 0:
            ax.set_ylabel(lab.replace("effect on the ", "effect on\n"))
        if r == 0:
            ax.set_title("additive outlier (AO)" if kind == "AO" else "innovative outlier (IO)")
axes[1, 0].annotate(r"$+\omega$", (tau - 0.1, 3.0), xytext=(tau + 2.2, 2.6), arrowprops=dict(arrowstyle="->", lw=0.6))
axes[1, 0].annotate(r"$-M\omega$ at $\tau+1$", (tau + 1.2, -1.2), xytext=(tau + 3.0, -1.2), va="center", arrowprops=dict(arrowstyle="->", lw=0.6))
axes[1, 1].annotate(r"$+\omega$ only", (tau - 0.1, 3.0), xytext=(tau + 2.2, 2.6), arrowprops=dict(arrowstyle="->", lw=0.6))
axes[0, 0].legend(frameon=False, loc="upper right", fontsize=7.5)
fig.tight_layout()
save(fig, "figA_patterns")

# ---------------------------------------------------------------- figure B
rho = 0.95
S = np.array([[1, rho], [rho, 1]])
alpha = 0.01
c2 = stats.chi2.ppf(1 - alpha, 2)                 # full-covariance statistic, 2 df
cz = stats.norm.ppf(1 - alpha / 4)                # per-location maximum, Bonferroni over 2
vals, vecs = np.linalg.eigh(S)
ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
rng = np.random.default_rng(3)
X = rng.multivariate_normal([0, 0], S, 600)
fig, ax = plt.subplots(figsize=(3.9, 3.7))
ax.scatter(X[:, 0], X[:, 1], s=3, color=GREY, alpha=0.35, lw=0, label="residuals without outliers")
ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(c2 * vals[1]), 2 * np.sqrt(c2 * vals[0]), angle=ang,
                     fill=False, color=BLUE, lw=1.6, label="full covariance: not flagged inside"))
ax.add_patch(Rectangle((-cz, -cz), 2 * cz, 2 * cz, fill=False, color=ORANGE, lw=1.6, ls="--",
                       label="per-location maximum: not flagged inside"))
A = np.array([1.5, -1.5])
B = np.array([2.95, 2.95])
for P, name, dx in [(A, "A", 0.15), (B, "B", -0.55)]:
    ax.plot(*P, "o", color="black", ms=5)
    ax.text(P[0] + dx, P[1] + 0.2, name, fontsize=10, weight="bold")
ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
ax.set_aspect("equal")
ax.set_xlabel("standardized residual, location 1")
ax.set_ylabel("standardized residual, location 2")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7, ncol=1)
for P, name in [(A, "A"), (B, "B")]:
    q = P @ np.linalg.solve(S, P)
    print(name, "Mahalanobis^2 =", round(q, 2), "(threshold", round(c2, 2), ") max|z| =", np.abs(P).max(), "(threshold", round(cz, 3), ")")
save(fig, "figB_ellipse")
print("written to", FIGS)
