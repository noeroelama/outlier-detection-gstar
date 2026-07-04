"""Order identification and adequacy diagnostics for GSTAR(1;1).

Space-time ACF/PACF follow Pfeifer & Deutsch (1980, Technometrics 22(3):397).
The space-time autocovariance between spatial lags l and k at time lag s is

    gamma_{lk}(s) = (1/N) tr[ W^(l)' W^(k) Gamma(s) ],   Gamma(s) = E[z_t z_{t+s}'],

an AGGREGATE over all N locations via the weight matrices and the trace/N
normalization -- it is NOT a per-location quantity. STACF is
gamma_{l0}(s) normalized; STPACF is the last-lag coefficient of a
space-time AR fit of increasing temporal order (operational Yule-Walker
definition), pooling locations under the STARMA homogeneity assumption
used for order selection.

Because that lifting from STARMA (homogeneous) to GSTAR (heterogeneous)
is an approximation, every function here is validated on simulated data
of KNOWN order in exp07 before being applied to real data; and residual
whiteness (multivariate Ljung-Box / Hosking 1980) plus a nested
higher-order significance test corroborate the STACF/STPACF reading.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- STACF ----
def _gamma(Z: np.ndarray, s: int) -> np.ndarray:
    """Sample Gamma(s) = (1/n) sum_t z_t z_{t+s}' on a centered series."""
    n = len(Z)
    Zc = Z - Z.mean(axis=0)
    return Zc[: n - s].T @ Zc[s:] / n


def st_autocov(Z, Wl, Wk, s) -> float:
    G = _gamma(Z, s)
    N = Z.shape[1]
    return float(np.trace(Wl.T @ Wk @ G) / N)


def stacf(Z: np.ndarray, Ws: list[np.ndarray], max_lag: int) -> dict[int, list[float]]:
    """STACF rho_l(s) = gamma_{l0}(s)/sqrt(gamma_{ll}(0) gamma_{00}(0)).

    Ws[0] must be the identity (spatial lag 0); Ws[l] the l-th spatial weight.
    """
    g00 = st_autocov(Z, Ws[0], Ws[0], 0)
    out = {}
    for l in range(len(Ws)):
        gll = st_autocov(Z, Ws[l], Ws[l], 0)
        denom = np.sqrt(gll * g00)
        out[l] = [st_autocov(Z, Ws[l], Ws[0], s) / denom for s in range(max_lag + 1)]
    return out


# --------------------------------------------------------------- STPACF ----
def stpacf(Z: np.ndarray, W: np.ndarray, max_lag: int) -> dict[int, tuple[float, float]]:
    """Operational space-time PACF at temporal lag k for spatial lags 0 and 1.

    Fit, for each k = 1..max_lag, the pooled space-time AR(k) regression
        z_i(t) = sum_{j=1}^k [ a_j z_i(t-j) + b_j (W z(t-j))_i ] + e_i(t)
    (homogeneous coefficients across locations, STARMA order-selection
    convention). The STPACF at lag k is the last-lag pair (a_k, b_k); a
    cut-off after the true temporal order p means (a_k, b_k) ~ 0 for k > p.
    Returns {k: (a_k, b_k)}.
    """
    n, N = Z.shape
    Zc = Z - Z.mean(axis=0)
    WZ = Zc @ W.T
    out = {}
    for k in range(1, max_lag + 1):
        rows_y, rows_X = [], []
        for t in range(k, n):
            for i in range(N):
                feats = []
                for j in range(1, k + 1):
                    feats += [Zc[t - j, i], WZ[t - j, i]]
                rows_X.append(feats)
                rows_y.append(Zc[t, i])
        X = np.asarray(rows_X)
        y = np.asarray(rows_y)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[k] = (float(beta[-2]), float(beta[-1]))  # (a_k, b_k)
    return out


def stpacf_se(n_obs: int, N: int) -> float:
    """Approximate 2-sigma bound for STPACF coefficients ~ 2/sqrt(N*n)."""
    return 2.0 / np.sqrt(N * n_obs)


# ------------------------------------------------ residual whiteness -------
def hosking_portmanteau(resid: np.ndarray, h: int, n_params_per_eq: int = 2):
    """Multivariate Ljung-Box (Hosking 1980) up to lag h.

    Q_h = n^2 sum_{s=1}^h (n-s)^{-1} tr( C(s)' C0^{-1} C(s) C0^{-1} ),
    C(s) residual autocovariance. Under adequacy Q_h ~ chi^2 with
    df = N^2 h - (fitted AR coefficients). For GSTAR(1;1) the fitted
    coefficients number 2N (2 per equation), so we subtract N*n_params_per_eq.
    Returns list of (h, Q, df, pvalue) for lags 1..h.
    """
    n, N = resid.shape
    R = resid - resid.mean(axis=0)
    def C(s):
        return R[: n - s].T @ R[s:] / n
    C0inv = np.linalg.inv(C(0))
    out = []
    Q = 0.0
    for s in range(1, h + 1):
        Cs = C(s)
        Q += (n * n) / (n - s) * np.trace(Cs.T @ C0inv @ Cs @ C0inv)
        df = N * N * s - N * n_params_per_eq
        if df > 0:
            p = 1.0 - stats.chi2.cdf(Q, df)
            out.append((s, float(Q), int(df), float(p)))
    return out


# --------------------------------------- nested higher-order order test ----
def added_lag2_test(Z: np.ndarray, W: np.ndarray):
    """Per-location F-test: does adding temporal lag 2 (own + spatial) help?

    Restricted:   z_i(t) = a1 z_i(t-1) + b1 (Wz(t-1))_i + e
    Unrestricted: + a2 z_i(t-2) + b2 (Wz(t-2))_i
    Returns list of (location_index, F, p_value); large p => lag 2 not needed.
    """
    n, N = Z.shape
    Zc = Z - Z.mean(axis=0)
    WZ = Zc @ W.T
    out = []
    for i in range(N):
        y = Zc[2:, i]
        Xr = np.column_stack([Zc[1:-1, i], WZ[1:-1, i]])
        Xu = np.column_stack([Zc[1:-1, i], WZ[1:-1, i], Zc[:-2, i], WZ[:-2, i]])
        def rss(X):
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ b
            return r @ r
        rss_r, rss_u = rss(Xr), rss(Xu)
        q = Xu.shape[1] - Xr.shape[1]
        dof_u = len(y) - Xu.shape[1]
        F = ((rss_r - rss_u) / q) / (rss_u / dof_u)
        p = 1.0 - stats.f.cdf(F, q, dof_u)
        out.append((i, float(F), float(p)))
    return out
