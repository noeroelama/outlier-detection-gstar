"""Rebuild the rubella and measles panels from the Japanese national
surveillance archive (JIHS/NIID NESID weekly notifiable-disease reports).

The shipped CSVs (rubella_japan_2012_2022.csv, measles_japan_2012_2022.csv)
were produced by this script. Rerunning it downloads the weekly ``zensu''
files, locates the rubella (fuu-shin) and measles (mashin) columns by
header name, and assembles a week x 47-prefecture panel.

Source: https://id-info.jihs.go.jp/niid/images/idwr/sokuho/
        idwr-{YEAR}/{YEAR}{WW}/{YEAR}-{WW}-zensu.csv   (Shift-JIS / cp932)
Notifiable rubella reporting has been complete in Japan since 2008.
"""
import csv
import urllib.request

BASE = "https://id-info.jihs.go.jp/niid/images/idwr/sokuho"
PREF_EN = ["Hokkaido", "Aomori", "Iwate", "Miyagi", "Akita", "Yamagata", "Fukushima",
           "Ibaraki", "Tochigi", "Gunma", "Saitama", "Chiba", "Tokyo", "Kanagawa",
           "Niigata", "Toyama", "Ishikawa", "Fukui", "Yamanashi", "Nagano", "Gifu",
           "Shizuoka", "Aichi", "Mie", "Shiga", "Kyoto", "Osaka", "Hyogo", "Nara",
           "Wakayama", "Tottori", "Shimane", "Okayama", "Hiroshima", "Yamaguchi",
           "Tokushima", "Kagawa", "Ehime", "Kochi", "Fukuoka", "Saga", "Nagasaki",
           "Kumamoto", "Oita", "Miyazaki", "Kagoshima", "Okinawa"]
DISEASES = {"rubella": "風しん", "measles": "麻しん"}  # 風しん / 麻しん


def url(year, week):
    return f"{BASE}/idwr-{year}/{year}{week:02d}/{year}-{week:02d}-zensu.csv"


def build(year_from=2012, year_to=2022):
    panels = {d: {} for d in DISEASES}
    weeks = []
    for year in range(year_from, year_to + 1):
        for week in range(1, 54):
            try:
                raw = urllib.request.urlopen(url(year, week), timeout=30).read()
            except Exception:
                continue
            rows = list(csv.reader(raw.decode("cp932", errors="replace").splitlines()))
            if len(rows) < 6:
                continue
            header = rows[2]
            cols = {d: [j for j, c in enumerate(header) if c.strip() == jp]
                    for d, jp in DISEASES.items()}
            if any(not v for v in cols.values()):
                continue
            prow = [r for r in rows[5:] if r and r[0].strip() and r[0].strip() != "総数"]
            if len(prow) < 47:
                continue
            wk = f"{year}-W{week:02d}"
            weeks.append(wk)
            for d, js in cols.items():
                j = js[0]
                panels[d][wk] = [int(prow[i][j]) if prow[i][j].strip().lstrip("-").isdigit() else 0
                                 for i in range(47)]
    return weeks, panels


def save(weeks, panel, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["week"] + PREF_EN)
        for wk in weeks:
            w.writerow([wk] + panel[wk])


if __name__ == "__main__":
    weeks, panels = build()
    print(f"downloaded {len(weeks)} weeks: {weeks[0]} .. {weeks[-1]}")
    save(weeks, panels["rubella"], "rubella_japan_2012_2022.csv")
    save(weeks, panels["measles"], "measles_japan_2012_2022.csv")
    print("wrote rubella_japan_2012_2022.csv and measles_japan_2012_2022.csv")
