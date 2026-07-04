# Data

## Files

- `rubella_japan_2012_2022.csv` — weekly notified rubella cases, 47 Japanese
  prefectures, 2012 week 37 to 2022 week 52 (538 weeks). One `week` column
  (`YYYY-Www`) plus 47 prefecture columns in JIS order.
- `measles_japan_2012_2022.csv` — the same, for measles (provided for
  completeness; not used in the paper).
- `download_jihs.py` — rebuilds both panels from the source archive.

## Source and provenance

Weekly notifiable-disease counts from the National Epidemiological
Surveillance of Infectious Diseases (NESID), published by the Japan
Institute for Health Security / National Institute of Infectious Diseases
in the Infectious Disease Weekly Report (IDWR):

  https://id-info.jihs.go.jp/

Rubella has been fully notifiable in Japan since 2008, so these are
complete case counts, not sentinel samples. The panels were assembled by
`download_jihs.py`, which downloads each weekly `zensu` file (Shift-JIS),
locates the rubella (風しん) and measles (麻しん) columns by header name,
and stacks the 47 prefecture rows. The upstream archive may revise
historical counts; the CSVs here are the frozen snapshot used in the paper
(accessed June 2026).

## Notes

- Rubella is a rare disease outside epidemics: about 89% of the
  prefecture-week counts are zero. The analysis therefore aggregates the 47
  prefectures to the 8 standard geographic regions of Japan.
- Two national epidemics dominate: 2012-2013 and 2018-2019, both centred on
  the Kanto and Kinki urban regions.
