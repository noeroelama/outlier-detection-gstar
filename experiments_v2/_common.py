"""Shared helpers for the revision experiments (Oct 2026)."""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results_v2"
RESULTS.mkdir(exist_ok=True)
FIGS = ROOT / "figures_v2"
FIGS.mkdir(exist_ok=True)

REGIONS = {"Hokkaido": [0], "Tohoku": list(range(1, 7)), "Kanto": list(range(7, 14)),
           "Chubu": list(range(14, 23)), "Kinki": list(range(23, 30)),
           "Chugoku": list(range(30, 35)), "Shikoku": list(range(35, 39)),
           "Kyushu-Okinawa": list(range(39, 47))}
NAMES = list(REGIONS)
# land/strait contiguity between the 8 standard regions (as in the original paper)
ADJ = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5, 6], 5: [4, 6, 7],
       6: [4, 5, 7], 7: [5, 6]}


def load_rubella():
    rows = list(csv.reader(open(ROOT / "data" / "rubella_japan_2012_2022.csv", encoding="utf-8")))
    weeks = [r[0] for r in rows[1:]]
    P = np.array([[int(x) for x in r[1:]] for r in rows[1:]], float)
    R = np.column_stack([P[:, ix].sum(1) for ix in REGIONS.values()])
    return weeks, P, R


def W_contiguity():
    N = len(NAMES)
    W = np.zeros((N, N))
    for i, ns in ADJ.items():
        W[i, ns] = 1.0
    return W / W.sum(1, keepdims=True)


def W_uniform(N):
    W = np.ones((N, N)) - np.eye(N)
    return W / W.sum(1, keepdims=True)


def W_invdist(coords):
    """Row-standardized inverse great-circle distance weights from (lat, lon)."""
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    d = np.arccos(np.clip(np.sin(lat[:, None]) * np.sin(lat) +
                          np.cos(lat[:, None]) * np.cos(lat) * np.cos(lon[:, None] - lon), -1, 1)) * 6371
    W = np.where(d > 0, 1 / np.where(d > 0, d, 1), 0)
    return W / W.sum(1, keepdims=True)
