"""Re-examination of the version-1 rubella application (why it was dropped).

Shows (1) the count of detections follows the cap max_outliers; (2) the
residuals are heavy-tailed; (3) the STACF of the log-differenced series has
lag-1 value about -0.48 and then cuts off (over-differencing pattern).
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from _common import load_rubella, W_contiguity, RESULTS
from gstar_outlier import iterative_detection, fit_gstar
from gstar_outlier.identification import stacf

weeks, P, R = load_rubella()
W = W_contiguity()
X = np.diff(np.log1p(R), axis=0)
out = []
for cap in (20, 40, 100, 200):
    r = iterative_detection(X, W, max_outliers=cap)
    yrs = [weeks[d.t + 1][:4] for d in r.detections]
    inep = sum(y in ("2012", "2013", "2018", "2019") for y in yrs)
    out.append(f"cap {cap:3d}: {len(r.detections)} detections, {inep} in epidemic years, sweeps {r.n_sweeps} (max 10)")
f = fit_gstar(X, W)
z = f.residuals / np.sqrt(np.diag(f.Sigma))
out.append("residual excess kurtosis by region: " + " ".join(f"{k:.1f}" for k in stats.kurtosis(z, axis=0)))
ac = stacf(X, [np.eye(8), W], 5)
out.append("STACF spatial lag 0, s=0..5: " + " ".join(f"{v:+.2f}" for v in ac[0][:6]))
txt = "\n".join(out)
print(txt)
(RESULTS / "rubella_v1_check.txt").write_text(txt, encoding="utf-8")
