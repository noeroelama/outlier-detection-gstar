"""Figures for the revised manuscript (written to figures_v2/, PDF + PNG).

fig1_study_area      - provinces, inverse-distance W (3 largest weights per province),
                       residual correlation against distance
fig2_power_AO / _IO  - size-adjusted power of the detectors (exp11)
fig3_classification  - AO/IO misclassification against magnitude (exp13)
fig4_rain_timeline   - joint statistic and per-location statistic over 1985-2024
fig5_rain_maps       - per-province t-statistics for four flagged months
fig6_rain_heatmap    - per-province t-statistics for all flagged months
Colours: Okabe-Ito; every detector also has its own marker and line style.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT, RESULTS, FIGS
from gstar_outlier import general as G

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.5})
STY = {"full/gstar": dict(c="#0072B2", m="o", ls="-", lab="full covariance (proposed)"),
       "perloc/gstar": dict(c="#D55E00", m="s", ls="--", lab="per-location maximum"),
       "perloc/ar": dict(c="#E69F00", m="v", ls=":", lab="independent AR series"),
       "diag/gstar": dict(c="#009E73", m="^", ls="-.", lab="diagonal covariance"),
       "full/var": dict(c="#CC79A7", m="D", ls=(0, (1, 1)), lab="unrestricted VAR, full cov."),
       }


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ simulations
d = pd.read_csv(RESULTS / "exp11_power_baselines.csv")
d["det"] = d.detector + "/" + d.fit
for kind in ("AO", "IO"):
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.2), sharex=True, sharey=True)
    for r, pat in enumerate(["single", "cluster", "global"]):
        for c, lv in enumerate(["low", "moderate", "high"]):
            ax = axes[r, c]
            sub = d[(d.kind == kind) & (d.threshold == "adjusted") & (d.pattern == pat) & (d["corr"] == lv)]
            for det, s in STY.items():
                q = sub[sub.det == det].sort_values("delta")
                ax.plot(q.delta, q.power, color=s["c"], marker=s["m"], ls=s["ls"], lw=1.4, ms=4, label=s["lab"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([3, 4, 5])
            if r == 0:
                ax.set_title(f"{lv} error correlation")
            if c == 0:
                ax.set_ylabel(f"{pat} pattern\npower")
            if r == 2:
                ax.set_xlabel("magnitude (error sd)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, f"fig2_power_{kind}")

e = pd.read_csv(RESULTS / "exp13_classification.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
for ax, pat in zip(axes, ["single", "cluster"]):
    for kind, s in [("AO", dict(c="#0072B2", m="o", ls="-")), ("IO", dict(c="#D55E00", m="s", ls="--"))]:
        q = e[(e.pattern == pat) & (e.kind == kind)]
        ax.plot(q.delta, q.misclass, color=s["c"], marker=s["m"], ls=s["ls"], lw=1.4, ms=4, label=f"true {kind}")
    ax.set_title(f"{pat}-location outlier" if pat == "single" else "outlier on a location and its 2 neighbours")
    ax.set_xlabel("magnitude (error sd)")
axes[0].set_ylabel("misclassification rate\n(among detected outliers)")
axes[0].legend(frameon=False)
fig.tight_layout()
save(fig, "fig3_classification")

# ------------------------------------------------------------------ application
js = json.loads((RESULTS / "exp20_rainfall.json").read_text())
z = np.load(RESULTS / "exp20_rainfall_series.npz", allow_pickle=True)
A, W, months, names = z["A"], z["W"], z["months"], z["names"]
lat, lon, rid = z["lat"], z["lon"], z["region_id"]
N = len(names)

import geopandas as gpd
from shapely.geometry import Point
prov = gpd.read_file(ROOT / "data" / "external" / "provinces_simplified.geojson")
pts = gpd.GeoDataFrame({"k": np.arange(N)}, geometry=[Point(x, y) for x, y in zip(lon, lat)], crs=4326)
sj = gpd.sjoin(pts, prov, how="left", predicate="within").drop_duplicates("k").set_index("k")
ISLANDS = {"Bangka Belitung Islands": "Kepulauan Bangka Belitung", "Riau Islands": "Kepulauan Riau",
           "Maluku": "Maluku", "North Maluku": "Maluku Utara", "East Nusa Tenggara": "Nusa Tenggara Timur",
           "North Sulawesi": "Sulawesi Utara", "Central Sulawesi": "Sulawesi Tengah"}
name_of = {k: (sj.PROVINSI[k] if isinstance(sj.PROVINSI[k], str) else ISLANDS[names[k]]) for k in range(N)}
k_of = {v: k for k, v in name_of.items()}
joined = prov.copy()
joined["k"] = joined.PROVINSI.map(k_of)
assert joined.k.nunique() == N, "province-to-centroid matching failed"
world = gpd.read_file(ROOT / "data" / "external" / "naturalearth_lowres" / "naturalearth_lowres.shp")
EXT = (94.5, 141.5, -11.5, 6.5)


def basemap(ax):
    world.plot(ax=ax, color="#f0f0f0", edgecolor="#cfcfcf", lw=0.3)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)


# figure 1
res = js
fit = G.fit_model(A, W, 1)
resid_full = G.compute_residuals(A, fit.Ms)
fig = plt.figure(figsize=(7.2, 2.9))
ax = fig.add_axes([0.0, 0.05, 0.66, 0.9])
basemap(ax)
prov.plot(ax=ax, color="#dbe7f3", edgecolor="white", lw=0.4)
for i in range(N):
    for j in np.argsort(-W[i])[:3]:
        ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="#0072B2", lw=0.4 + 6 * W[i, j], alpha=0.6, zorder=3)
ax.scatter(lon, lat, s=9, color="#0072B2", zorder=4, edgecolor="white", lw=0.5)
ax.set_axis_off()
ax.set_title("(a) 38 provinces and the three largest weights of each row of W", loc="left")
ax.set_xlabel(""); ax.set_ylabel("")
ax2 = fig.add_axes([0.72, 0.18, 0.27, 0.68])
la, lo = np.radians(lat), np.radians(lon)
D = np.arccos(np.clip(np.sin(la[:, None]) * np.sin(la) + np.cos(la[:, None]) * np.cos(la) * np.cos(lo[:, None] - lo), -1, 1)) * 6371
Cr = np.corrcoef(resid_full.T)
iu = np.triu_indices(N, 1)
ax2.scatter(D[iu], Cr[iu], s=3, color="#0072B2", alpha=0.35, lw=0)
ax2.axhline(0, color="#777777", lw=0.6)
ax2.set_xlabel("distance between centroids (km)")
ax2.set_ylabel("residual correlation")
ax2.set_title("(b) GSTAR(1;1) residuals", loc="left")
save(fig, "fig1_study_area")

# figure 4: timeline of the joint and the per-location statistic (final parameters)
final = G.iterative_detection(A, W, 1, max_outliers=60, max_sweeps=15)
e = G.compute_residuals(A, final.fit.Ms)
a, i, *_ = G.scan_statistics(e, final.fit.Ms, final.fit.Sigma, "full")
final_pl = G.iterative_detection(A, W, 1, "perloc", max_outliers=60, max_sweeps=15)   # its own fit
e_pl = G.compute_residuals(A, final_pl.fit.Ms)
pa, pi, *_ = G.scan_statistics(e_pl, final_pl.fit.Ms, final_pl.fit.Sigma, "perloc")
t = pd.to_datetime(pd.Series(months[1:]) + "-15")
c_full = G.bonferroni_threshold("full", N, len(e))
c_pl = G.bonferroni_threshold("perloc", N, len(e))
fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.9), sharex=True)
axes[0].plot(t, np.maximum(a, i), color="#0072B2", lw=0.6)
axes[0].axhline(c_full, color="#111111", ls="--", lw=0.8)
axes[0].text(t.iloc[5], c_full + 3, "Bonferroni threshold (FWER 0.05)", fontsize=7.5)
for dd in final.detections:
    k = dd.t - 1
    axes[0].scatter(t.iloc[k], max(a[k], i[k]), s=26, facecolor="none",
                    edgecolor="#D55E00", marker="o" if dd.kind == "AO" else "^", lw=1.1, zorder=5)
axes[0].set_ylabel(r"$\max(\hat\lambda^2_{AO,t},\hat\lambda^2_{IO,t})$")
axes[0].set_title("(a) full-covariance statistic (circle: AO, triangle: IO)", loc="left")
axes[1].plot(t, np.sqrt(np.maximum(pa, pi)), color="#D55E00", lw=0.6)
axes[1].axhline(np.sqrt(c_pl), color="#111111", ls="--", lw=0.8)
axes[1].set_ylabel(r"$\max_i |z_{i,t}|$")
axes[1].set_title(f"(b) per-location statistic (largest |z| over 38 provinces); {len(final_pl.detections)} detections", loc="left")
fig.tight_layout()
save(fig, "fig4_rain_timeline")

# figure 5: maps of t-statistics for four flagged months
dets = {r["month"]: r for r in js["detections_full"]}
pick = ["1988-02", "1997-02", "2007-12", "2010-04"]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.6))
vmax = 4.5
for ax, m in zip(axes.ravel(), pick):
    basemap(ax)
    ts = np.array(dets[m]["tstats"])
    joined["t"] = ts[joined.k.values]
    joined.plot(ax=ax, column="t", cmap="RdBu", vmin=-vmax, vmax=vmax, edgecolor="white", lw=0.3)
    ax.set_axis_off()
    ax.set_title(f"{m}  ({dets[m]['type']}, " + r"$\hat\lambda^2_{AO}$" + f" = {dets[m]['lam2_AO']:.0f})", loc="left")
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(-vmax, vmax))
cb = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.01)
cb.set_label("per-province t-statistic of the effect\n(blue: wetter, red: drier than the model expects)")
save(fig, "fig5_rain_maps")

# figure 6: heatmap of t-statistics, provinces ordered west to east
order = np.argsort(lon)
mm = [r["month"] for r in js["detections_full"]]
H = np.array([r["tstats"] for r in js["detections_full"]]).T[order]
fig, ax = plt.subplots(figsize=(7.0, 5.6))
im = ax.imshow(H, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_yticks(range(N)); ax.set_yticklabels(np.array(names)[order], fontsize=6.5)
ax.set_xticks(range(len(mm)))
ax.set_xticklabels([f"{m} {r['type']}" for m, r in zip(mm, js["detections_full"])], rotation=60, ha="right", fontsize=7)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
cb.set_label("per-province t-statistic")
ax.set_ylabel("province (west to east)")
fig.tight_layout()
save(fig, "fig6_rain_heatmap")
print("figures written to", FIGS)
