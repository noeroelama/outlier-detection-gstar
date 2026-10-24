"""Rainfall panel helpers: CHIRPS v2.0 province means, 38 Indonesian provinces, 1985-2024."""
import numpy as np
import pandas as pd
from _common import ROOT, W_invdist


def load_rain():
    d = pd.read_csv(ROOT / "data" / "panel_rainfall_long.csv")
    Y = d.pivot(index="year_month", columns="region_id", values="rainfall").sort_index()
    idx = d.drop_duplicates("year_month").set_index("year_month").sort_index()[["nino34", "dmi"]]
    cen = pd.read_csv(ROOT / "data" / "a_geo_centroids.csv").set_index("region_id")
    cen = cen.loc[Y.columns]
    return Y, idx, cen


def anomalies(Y):
    """Square-root rainfall, standardized by calendar month and province."""
    S = np.sqrt(Y.values)
    mon = np.array([int(s[-2:]) for s in Y.index])
    A = np.empty_like(S)
    for m in range(1, 13):
        k = mon == m
        A[k] = (S[k] - S[k].mean(0)) / S[k].std(0, ddof=1)
    return A


def W_rain(cen, kind="invdist"):
    C = cen[["lat", "lon"]].values
    return W_invdist(C)
