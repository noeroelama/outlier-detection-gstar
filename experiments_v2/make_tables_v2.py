"""LaTeX tables for the revised manuscript, generated from results_v2/ (written to tables_v2/)."""
import json

import pandas as pd

from _common import ROOT, RESULTS

OUT = ROOT / "tables_v2"
OUT.mkdir(exist_ok=True)
f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"

# Table: false-alarm rate, Gaussian errors -------------------------------------------
d = pd.read_csv(RESULTS / "exp10_fwer.csv")
d = d[(d.p == 1) & (d.errors == "gauss")]
rows = []
for N in (3, 8, 16):
    for lv in ("low", "high"):
        g = d[(d.N == N) & (d["corr"] == lv)]
        val = lambda det, fit, T: g[(g.detector == det) & (g.fit == fit) & (g["T"] == T)].fwer.iloc[0]
        cells = [str(N), lv]
        cells += [f3(val("full", "gstar", T)) for T in (100, 250, 500)]
        cells += [f3(val("perloc", "gstar", T)) for T in (100, 250, 500)]
        cells += [f3(val("diag", "gstar", 250)), f3(val("fixed3", "gstar", 250))]
        rows.append(" & ".join(cells) + r" \\")
(OUT / "tab_fwer.tex").write_text("\n".join(rows) + "\n")

# t5 summary range for the text
t5 = pd.read_csv(RESULTS / "exp10_fwer.csv")
t5 = t5[(t5.errors == "t5") & (t5.detector == "full") & (t5.fit == "gstar")]
p2 = pd.read_csv(RESULTS / "exp10_fwer.csv")
p2 = p2[(p2.p == 2) & (p2.detector == "full") & (p2.fit == "gstar")]
var = pd.read_csv(RESULTS / "exp10_fwer.csv")
var = var[(var.p == 1) & (var.errors == "gauss") & (var.detector == "full") & (var.fit == "var")]
summ = dict(t5_min=t5.fwer.min(), t5_max=t5.fwer.max(), p2_min=p2.fwer.min(), p2_max=p2.fwer.max(),
            var_min=var.fwer.min(), var_max=var.fwer.max(),
            full_gauss_min=d[(d.detector == "full") & (d.fit == "gstar")].fwer.min(),
            full_gauss_max=d[(d.detector == "full") & (d.fit == "gstar")].fwer.max(),
            perloc_gauss_max=d[(d.detector == "perloc") & (d.fit == "gstar")].fwer.max(),
            diag_gauss_min=d[(d.detector == "diag") & (d.fit == "gstar")].fwer.min(),
            diag_gauss_max=d[(d.detector == "diag") & (d.fit == "gstar")].fwer.max(),
            fixed3_min=d[(d.detector == "fixed3")].fwer.min())
# Table: adjacent outliers ----------------------------------------------------------
a = pd.read_csv(RESULTS / "exp12_adjacent.csv")
rows = []
for sc in a.scenario.unique():
    f = a[(a.scenario == sc) & (a.detector == "full")].iloc[0]
    p = a[(a.scenario == sc) & (a.detector == "perloc")].iloc[0]
    rows.append(f"{sc} & {f2(f.recall)} & {f2(f.precision)} & {f2(f.type_acc)} & {f2(p.recall)} & {f2(p.precision)} & {f2(p.type_acc)} \\\\")
(OUT / "tab_adjacent.tex").write_text("\n".join(rows) + "\n")

# Table: weight matrix ----------------------------------------------------------------
w = pd.read_csv(RESULTS / "exp14_weights.csv")
rows = [f"{r['corr']} & {r.fitted_W} & {f3(r.fwer)} & {f2(r.power_AO)} & {f2(r.power_IO)} \\\\" for _, r in w.iterrows()]
(OUT / "tab_weights.tex").write_text("\n".join(rows) + "\n")

# Application ------------------------------------------------------------------------
js = json.loads((RESULTS / "exp20_rainfall.json").read_text())
rows = []
for r in js["detections_full"]:
    top = "; ".join(r["top3"]).replace("(-", "($-$")
    rows.append(f"{r['month']} & {r['type']} & {r['lam2_AO']:.1f} & {r['lam2_IO']:.1f} & {r['max_abs_t']:.1f} & {r['n_abs_t_gt2']} & {top} \\\\")
(OUT / "tab_rain_detections.tex").write_text("\n".join(rows) + "\n")
rows = []
for r in js["semi_synthetic"]:
    rows.append(f"{r['pattern']} & {r['kind']} & {r['delta']} & {f2(r['power_full'])} & {f2(r['power_perloc'])} & {f2(r['false_full'])} & {f2(r['false_perloc'])} \\\\")
(OUT / "tab_rain_semi.tex").write_text("\n".join(rows) + "\n")
summ.update(app_bootstrap=js["bootstrap"], app_n=js["n_detections"], W_sens={k: (v["n"], v["overlap_with_invdist"]) for k, v in js["W_sensitivity"].items()})
(OUT / "summary_numbers.json").write_text(json.dumps(summ, indent=1, default=float))
print(json.dumps(summ, indent=1, default=float))

# Table: GSTAR restriction vs unrestricted VAR (exp15) ---------------------------------
v = pd.read_csv(RESULTS / "exp15_gstar_vs_var.csv")
rows = []
for (N, T), g in v.groupby(["N", "T"]):
    gs = g[g.fit == "gstar"].iloc[0]
    va = g[g.fit == "var"].iloc[0]
    fmt = lambda x, f: "--" if pd.isna(x) else f(x)
    rows.append(f"{N} & {T} & {f3(gs.M_error)} & {fmt(va.M_error, f3)} & {f3(gs.fwer_nominal)} & {fmt(va.fwer_nominal, f3)} & "
                f"{f2(gs.power_AO)} & {fmt(va.power_AO, f2)} & {f2(gs.power_IO)} & {fmt(va.power_IO, f2)} \\\\")
(OUT / "tab_gstar_var.tex").write_text("\n".join(rows) + "\n")

# Table: combined test vs the two components (exp11 + exp16, size-adjusted, 4 sd) -------
b = pd.read_csv(RESULTS / "exp16_power.csv")
e = pd.read_csv(RESULTS / "exp11_power_baselines.csv")
e = e[(e.fit == "gstar") & (e.detector.isin(["full", "perloc"]))]
cols = ["corr", "pattern", "kind", "delta", "detector", "threshold", "power"]
pw = pd.concat([e[cols], b[cols]])
pw = pw[(pw.threshold == "adjusted") & (pw.delta == 4)].pivot_table(index=["pattern", "corr"], columns=["kind", "detector"], values="power")
rows = []
for pat in ("single", "cluster", "global"):
    for lv in ("low", "moderate", "high"):
        r = pw.loc[(pat, lv)]
        cells = [pat, lv] + [f2(r[(k, d)]) for k in ("AO", "IO") for d in ("full", "perloc", "combo")]
        rows.append(" & ".join(cells) + r" \\")
(OUT / "tab_combo.tex").write_text("\n".join(rows) + "\n")

# Table: semi-synthetic rainfall with the combined test ------------------------------
jc = json.loads((RESULTS / "exp16_rainfall.json").read_text())
rows = []
for r, c in zip(js["semi_synthetic"], jc["semi_synthetic_combo"]):
    assert (r["pattern"], r["kind"], r["delta"]) == (c["pattern"], c["kind"], c["delta"])
    rows.append(f"{r['pattern']} & {r['kind']} & {r['delta']} & {f2(r['power_full'])} & {f2(r['power_perloc'])} & {f2(c['power_combo'])} & "
                f"{f2(r['false_full'])} & {f2(r['false_perloc'])} & {f2(c['false_combo'])} \\\\")
(OUT / "tab_rain_semi.tex").write_text("\n".join(rows) + "\n")
