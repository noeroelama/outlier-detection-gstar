"""exp21 - check the months flagged in CHIRPS against a second rainfall dataset.

Question: are the 17 months flagged by the full-covariance procedure in CHIRPS
(exp20) also unusual in an independent monthly rainfall panel for the same 38
provinces, or are they specific to CHIRPS?

Input: a long CSV with one row per province and month. The province column may
hold ISO 3166-2 codes (ID-AC, ...), English names or Indonesian names.
Units do not matter: the same transform as exp20 is applied (square root,
standardized by calendar month and province).

For every CHIRPS-flagged month inside the common period we report
  r_anom : correlation over provinces between the CHIRPS and second-dataset anomalies
  r_res  : correlation over provinces between the GSTAR(1;1) residuals of the two panels
  lam2   : first-scan max(lambda2_AO, lambda2_IO) of the second panel at that month
  pct    : percentile of lam2 among all months of the second panel
and whether the iterative procedure applied to the second panel flags it.
Reference values: the same correlations over all months.

Usage:
  python exp21_second_dataset.py --file "D:/MyDrive/Kuliah/Ehime/Doctoral/Data/era5_tp_province_monthly.csv" \
      --prov-col name --value-col value --label ERA5 [--filter-col variable --filter-val tp]
"""
import argparse
import json

import numpy as np
import pandas as pd

from _common import RESULTS
from _rain import load_rain, anomalies, W_rain
from gstar_outlier import general as G

IDN = {'Aceh': 'ID-AC', 'Sumatera Utara': 'ID-SU', 'Sumatera Barat': 'ID-SB', 'Riau': 'ID-RI', 'Jambi': 'ID-JA',
       'Sumatera Selatan': 'ID-SS', 'Bengkulu': 'ID-BE', 'Lampung': 'ID-LA', 'Kepulauan Bangka Belitung': 'ID-BB',
       'Bangka Belitung': 'ID-BB', 'Kepulauan Riau': 'ID-KR', 'DKI Jakarta': 'ID-JK', 'Jawa Barat': 'ID-JB',
       'Jawa Tengah': 'ID-JT', 'Daerah Istimewa Yogyakarta': 'ID-YO', 'DI Yogyakarta': 'ID-YO', 'Jawa Timur': 'ID-JI',
       'Banten': 'ID-BT', 'Bali': 'ID-BA', 'Nusa Tenggara Barat': 'ID-NB', 'Nusa Tenggara Timur': 'ID-NT',
       'Kalimantan Barat': 'ID-KB', 'Kalimantan Tengah': 'ID-KT', 'Kalimantan Selatan': 'ID-KS',
       'Kalimantan Timur': 'ID-KI', 'Kalimantan Utara': 'ID-KU', 'Sulawesi Utara': 'ID-SA',
       'Sulawesi Tengah': 'ID-ST', 'Sulawesi Selatan': 'ID-SN', 'Sulawesi Tenggara': 'ID-SG', 'Gorontalo': 'ID-GO',
       'Sulawesi Barat': 'ID-SR', 'Maluku': 'ID-MA', 'Maluku Utara': 'ID-MU', 'Papua': 'ID-PA',
       'Papua Barat': 'ID-PB', 'Papua Selatan': 'ID-PS', 'Papua Tengah': 'ID-PT', 'Papua Pegunungan': 'ID-PE',
       'Papua Barat Daya': 'ID-PD'}


def to_region(v, eng):
    if isinstance(v, str) and v.startswith("ID-"):
        return v
    return IDN.get(v, eng.get(v))


def corr_rows(X, Y):
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    return (Xc * Yc).sum(1) / np.sqrt((Xc ** 2).sum(1) * (Yc ** 2).sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--prov-col", default="province")
    ap.add_argument("--month-col", default="year_month")
    ap.add_argument("--value-col", default="rainfall")
    ap.add_argument("--filter-col")
    ap.add_argument("--filter-val")
    ap.add_argument("--label", default="second")
    a = ap.parse_args()

    Y, idx, cen = load_rain()
    Y = Y.loc[:"2024-12"]
    eng = dict(zip(cen["name"], cen.index))
    flagged = [r["month"] for r in json.load(open(RESULTS / "exp20_rainfall.json"))["detections_full"]]

    d = pd.read_csv(a.file)
    if a.filter_col:
        d = d[d[a.filter_col].astype(str) == a.filter_val]
    d["rid"] = [to_region(v, eng) for v in d[a.prov_col]]
    miss = sorted(set(d.loc[d.rid.isna(), a.prov_col]))
    if miss:
        raise SystemExit(f"unmatched province names: {miss}")
    d["ym"] = d[a.month_col].astype(str).str[:7]
    Y2 = d.pivot_table(index="ym", columns="rid", values=a.value_col).sort_index()
    Y2 = Y2[Y.columns]                                   # same order as CHIRPS
    common = Y.index.intersection(Y2.index)
    common = common[[pd.Period(c).month for c in common].index(1):]   # start in January
    n_years = len(common) // 12
    common = common[:12 * n_years]
    Yc, Y2c = Y.loc[common], Y2.loc[common].clip(lower=0)
    if Y2c.isna().any().any():
        raise SystemExit("second dataset has missing values in the common period")
    A1, A2 = anomalies(Yc), anomalies(Y2c)
    W = W_rain(cen)
    f1, f2 = G.fit_gstar_p(A1, W, 1), G.fit_gstar_p(A2, W, 1)
    r_anom = corr_rows(A1, A2)
    r_res = corr_rows(f1.resid, f2.resid)
    lam_ao, lam_io, _ = G.stats_quadratic(f2.resid, f2.Ms, f2.Sigma)
    lam = np.maximum(lam_ao, lam_io)
    T2 = len(lam)
    c2 = G.threshold("full", len(W), T2)
    res = G.iterative(A2, W, 1, "full", "gstar", max_outliers=60, max_sweeps=15)
    det2 = sorted(common[dd.t] for dd in res.detections)
    res1 = G.iterative(A1, W, 1, "full", "gstar", max_outliers=60, max_sweeps=15)
    det1 = sorted(common[dd.t] for dd in res1.detections)   # CHIRPS refitted on the common period

    rows = []
    for m in flagged:
        if m not in common:
            continue
        t = list(common).index(m)
        rows.append(dict(month=m, r_anom=round(float(r_anom[t]), 3),
                         r_res=round(float(r_res[t - 1]), 3) if t >= 1 else None,
                         lam2=round(float(lam[t - 1]), 1) if t >= 1 else None,
                         pct=round(float((lam < lam[t - 1]).mean() * 100), 1) if t >= 1 else None,
                         above_c2=bool(lam[t - 1] > c2) if t >= 1 else None,
                         flagged_iterative=m in det2))
    out = dict(label=a.label, period=[common[0], common[-1]], T=len(common),
               median_r_anom_all=round(float(np.median(r_anom)), 3),
               median_r_res_all=round(float(np.median(r_res)), 3),
               c2=round(c2, 1), n_flagged_second=len(det2), flagged_second=det2,
               chirps_common_period_flagged=det1, overlap=sorted(set(det1) & set(det2)),
               chirps_flagged_in_period=len(rows),
               resid_kurt_median=float(np.median(pd.DataFrame(f2.resid).kurt())),
               months=rows)
    fn = RESULTS / f"exp21_second_dataset_{a.label}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "months"}, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
