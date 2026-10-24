"""Order-p GSTAR outlier detection and the benchmark detectors (revision, Oct 2026).

Model (GSTAR(p;1), zero-mean series):

    Z_t = sum_{k=1}^p (Phi_k0 + Phi_k1 W) Z_{t-k} + eps_t,   eps_t ~ N(0, Sigma)

with M_k = Phi_k0 + Phi_k1 W.  Write Pi_0 = I and Pi_k = -M_k (k = 1..p), so the
residual filter is e_t = sum_k Pi_k Psi_{t-k}.

Residual signatures of one outlier with effect vector omega at time T:

    AO:  e_{T+j} = eps_{T+j} + Pi_j omega,   j = 0..p   (p+1 points)
    IO:  e_T     = eps_T + omega                         (one point)

GLS (Gaussian ML for known M_k, Sigma), with the window truncated at the sample end:

    A_T      = sum_j Pi_j' Sigma^{-1} Pi_j
    g_T      = sum_j Pi_j' Sigma^{-1} e_{T+j}
    omega_AO = A_T^{-1} g_T,           lam2_AO = g_T' A_T^{-1} g_T   ~ chi2_N under H0
    omega_IO = e_T,                    lam2_IO = e_T' Sigma^{-1} e_T  ~ chi2_N under H0

For p = 1 this is exactly the statistic of detection.py (original submission).

Detectors (all share the residual signatures, the sweep architecture and the
re-estimation steps; they differ only in how the residual covariance is used):

  "full"    : the proposed statistics above (full Sigma), chi2_N reference.
  "diag"    : the same quadratic statistics with Sigma replaced by diag(Sigma)
              (working independence).  Under cross-location correlation the
              chi2_N reference is NOT valid for it; this is the ablation
              "covariance switched off".
  "perloc"  : per-location maximum test.  For every location i and time t the
              per-location effect estimate under working independence is
              standardized by its exact standard deviation under the true
              Sigma (sandwich), so each z is N(0,1) under H0; the maximum |z|
              over locations, times and the two outlier types is compared with
              a Bonferroni threshold over 2*N*T_eff tests.  This is the
              calibrated version of the per-location rule of Huda et al.
              (2022) and the C-type (component-wise maximum) statistic of
              Tsay et al. (2000).
  "fixed3"  : the "perloc" statistics compared with the conventional constant
              |z| > 3 used in the GSTAR literature (no multiplicity control);
              reported only for its false-alarm rate.

Fitting options: "gstar" (restricted), "var" (unrestricted VAR(p), N^2 p
coefficients, Tsay et al. 2000), "ar" (N independent AR(p) series, no spatial
lag: the 'independent time series' baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
@dataclass
class FitP:
    Ms: np.ndarray            # (p, N, N) coefficient matrices M_1..M_p
    Sigma: np.ndarray         # (N, N) residual covariance
    resid: np.ndarray         # (T-p, N); resid[k] = e_{k+p}
    p: int
    coef: np.ndarray          # (N, 2p) per-location [phi_k0, phi_k1]_{k=1..p} (gstar) or empty
    kind: str


def _lags(Z: np.ndarray, p: int) -> list[np.ndarray]:
    T = len(Z)
    return [Z[p - k:T - k] for k in range(1, p + 1)]   # lag k aligned with Z[p:]


def fit_gstar_p(Z: np.ndarray, W: np.ndarray, p: int = 1, kind: str = "gstar") -> FitP:
    """Least squares for GSTAR(p;1) ('gstar'), N independent AR(p) ('ar') or VAR(p) ('var')."""
    T, N = Z.shape
    Y = Z[p:]
    L = _lags(Z, p)
    Ms = np.zeros((p, N, N))
    resid = np.empty_like(Y)
    coef = np.zeros((N, 2 * p))
    if kind == "var":
        X = np.hstack(L)                                  # (T-p, N p)
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)         # (N p, N)
        for k in range(p):
            Ms[k] = B[k * N:(k + 1) * N].T
        resid = Y - X @ B
    else:
        V = [lk @ W.T for lk in L]
        for i in range(N):
            cols = []
            for k in range(p):
                cols.append(L[k][:, i])
                if kind == "gstar":
                    cols.append(V[k][:, i])
            Xi = np.column_stack(cols)
            b, *_ = np.linalg.lstsq(Xi, Y[:, i], rcond=None)
            resid[:, i] = Y[:, i] - Xi @ b
            step = 2 if kind == "gstar" else 1
            for k in range(p):
                a = b[k * step]
                Ms[k, i, i] += a
                coef[i, 2 * k] = a
                if kind == "gstar":
                    c = b[k * step + 1]
                    Ms[k, i, :] += c * W[i, :]
                    coef[i, 2 * k + 1] = c
    Sigma = np.cov(resid, rowvar=False, ddof=1)
    return FitP(Ms, Sigma, resid, p, coef, kind)


def companion_radius(Ms: np.ndarray) -> float:
    """Spectral radius of the companion matrix (stationarity iff < 1)."""
    p, N, _ = Ms.shape
    C = np.zeros((N * p, N * p))
    C[:N, :] = np.hstack(list(Ms))
    if p > 1:
        C[N:, :-N] = np.eye(N * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def psi_weights(Ms: np.ndarray, H: int) -> np.ndarray:
    """MA(infinity) weights Psi_0..Psi_{H-1} of phi(B)^{-1}: IO propagation."""
    p, N, _ = Ms.shape
    Psi = np.zeros((H, N, N))
    Psi[0] = np.eye(N)
    for h in range(1, H):
        for k in range(1, min(h, p) + 1):
            Psi[h] += Ms[k - 1] @ Psi[h - k]
    return Psi


def simulate_gstar_p(Ms, Sigma, T, burn=200, rng=None, df=None):
    """Simulate a GSTAR(p)/VAR(p) path. df: Student-t innovations (unit covariance scaled)."""
    rng = rng if rng is not None else np.random.default_rng()
    p, N, _ = Ms.shape
    if companion_radius(Ms) >= 1:
        raise ValueError("non-stationary")
    Lc = np.linalg.cholesky(Sigma)
    hist = np.zeros((p, N))
    out = np.empty((T, N))
    for t in range(burn + T):
        u = rng.standard_normal(N)
        if df is not None:
            u = u / np.sqrt(rng.chisquare(df) / df) * np.sqrt((df - 2) / df)
        z = Lc @ u
        for k in range(p):
            z = z + Ms[k] @ hist[k]
        hist = np.vstack([z, hist[:-1]])
        if t >= burn:
            out[t - burn] = z
    return out


def inject(Z, Ms, t0, omega, kind):
    """Plant one AO (single observation) or IO (propagated through Psi weights)."""
    Psi = Z.copy()
    if kind == "AO":
        Psi[t0] += omega
    else:
        P = psi_weights(Ms, len(Z) - t0)
        Psi[t0:] += np.einsum("hij,j->hi", P, omega)
    return Psi


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def _pis(Ms):
    p, N, _ = Ms.shape
    return np.concatenate([np.eye(N)[None], -Ms], axis=0)      # (p+1, N, N)


def _gls_pieces(resid, Ms, Sinv):
    """g_t = sum_j Pi_j' Sinv e_{t+j} and A_t (window truncated at the end)."""
    Tr, N = resid.shape
    Pis = _pis(Ms)
    P1 = len(Pis)
    F = resid @ Sinv                                           # rows: (Sinv e_t)'
    g = np.zeros((Tr, N))
    for j in range(P1):
        g[:Tr - j] += F[j:] @ Pis[j]                           # (Pi_j' Sinv e_{t+j})'
    Afull = sum(Pis[j].T @ Sinv @ Pis[j] for j in range(P1))
    A_list = {}
    for t in range(max(0, Tr - P1 + 1), Tr):                    # truncated windows
        J = Tr - 1 - t
        A_list[t] = sum(Pis[j].T @ Sinv @ Pis[j] for j in range(J + 1))
    return g, Afull, A_list, Pis


def stats_quadratic(resid, Ms, S):
    """lam2_AO, lam2_IO, omega_AO for the quadratic detector with covariance S
    (S = Sigma for 'full', S = diag(Sigma) for 'diag')."""
    Sinv = np.linalg.inv(S)
    g, Afull, A_list, _ = _gls_pieces(resid, Ms, Sinv)
    Ainv = np.linalg.inv(Afull)
    omega = g @ Ainv                                           # Ainv symmetric
    lam_ao = np.einsum("ti,ti->t", omega, g)
    for t, At in A_list.items():
        w = np.linalg.solve(At, g[t])
        omega[t] = w
        lam_ao[t] = w @ g[t]
    lam_io = np.einsum("ti,ij,tj->t", resid, Sinv, resid)
    return lam_ao, lam_io, omega


def stats_perloc(resid, Ms, Sigma):
    """Per-location z statistics (working independence estimate, exact sandwich sd)."""
    D = np.diag(np.diag(Sigma))
    Dinv = np.linalg.inv(D)
    g, Afull, A_list, Pis = _gls_pieces(resid, Ms, Dinv)

    def sandwich(At, J):
        Ainv = np.linalg.inv(At)
        B = sum(Pis[j].T @ Dinv @ Sigma @ Dinv @ Pis[j] for j in range(J + 1))
        return Ainv, np.sqrt(np.diag(Ainv @ B @ Ainv))

    Ainv, sd = sandwich(Afull, len(Pis) - 1)
    omega = g @ Ainv
    z_ao = omega / sd
    for t, At in A_list.items():
        Ai, sdt = sandwich(At, len(resid) - 1 - t)
        omega[t] = Ai @ g[t]
        z_ao[t] = omega[t] / sdt
    z_io = resid / np.sqrt(np.diag(Sigma))
    return z_ao, z_io, omega


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------
def threshold(detector: str, N: int, T_eff: int, alpha: float = 0.05) -> float:
    """Squared critical value on the scale of the detector's statistic."""
    if detector in ("full", "diag"):
        return float(stats.chi2.ppf(1 - alpha / (2 * T_eff), N))
    if detector == "perloc":
        return float(stats.norm.ppf(1 - alpha / (4 * N * T_eff)) ** 2)
    if detector == "fixed3":
        return 9.0
    if detector == "combo":
        return 1.0          # statistics are already divided by their own critical values
    raise ValueError(detector)


def scan(resid, Ms, Sigma, detector, alpha=0.05):
    """Per-time maximal statistic for each type: returns (s_ao, s_io, omega_ao, loc_ao, loc_io)."""
    if detector == "full":
        a, i, w = stats_quadratic(resid, Ms, Sigma)
        return a, i, w, None, None
    if detector == "combo":
        # union of the full-covariance test and the per-location maximum test, each at
        # level alpha/2 (Bonferroni over the two families): flag iff either exceeds its
        # own threshold, i.e. iff max(lam2 / c_full, z2max / c_perloc) > 1
        Tr, N = resid.shape
        c_f = threshold("full", N, Tr, alpha / 2)
        c_p = threshold("perloc", N, Tr, alpha / 2)
        a, i, w = stats_quadratic(resid, Ms, Sigma)
        za, zi, _ = stats_perloc(resid, Ms, Sigma)
        sa = np.maximum(a / c_f, np.max(za ** 2, 1) / c_p)
        si = np.maximum(i / c_f, np.max(zi ** 2, 1) / c_p)
        return sa, si, w, None, None
    if detector == "diag":
        a, i, w = stats_quadratic(resid, Ms, np.diag(np.diag(Sigma)))
        return a, i, w, None, None
    if detector in ("perloc", "fixed3"):
        za, zi, w = stats_perloc(resid, Ms, Sigma)
        return (np.max(za ** 2, 1), np.max(zi ** 2, 1), w,
                np.argmax(za ** 2, 1), np.argmax(zi ** 2, 1))
    raise ValueError(detector)


def initial_max(Z, W, p=1, detector="full", fit_kind="gstar"):
    """Maximum statistic of the first scan. Under the global null the iterative
    procedure can report an outlier only if this exceeds the threshold (Prop. 2)."""
    fit = fit_gstar_p(Z, W, p, fit_kind)
    a, i, *_ = scan(fit.resid, fit.Ms, fit.Sigma, detector)
    return float(max(a.max(), i.max())), len(fit.resid)


# --------------------------------------------------------------------------
# iterative procedure
# --------------------------------------------------------------------------
@dataclass
class Det:
    t: int            # series time index of the outlier
    kind: str
    stat_ao: float
    stat_io: float
    omega: np.ndarray
    tstats: np.ndarray
    loc: int


@dataclass
class Result:
    detections: list
    fit: FitP
    c2: float
    n_sweeps: int
    history: list = field(default_factory=list)
    converged: bool = True      # False: max_sweeps reached without a stable detection set
    hit_cap: bool = False       # True: some sweep stopped at max_outliers (the Gaussian
                                # reference is then probably inadequate; do not trust the count)


def _adjust(resid, Ms, d_t_res, kind, omega):
    out = resid.copy()
    if kind == "IO":
        out[d_t_res] -= omega
    else:
        Pis = _pis(Ms)
        for j in range(len(Pis)):
            if d_t_res + j < len(out):
                out[d_t_res + j] -= Pis[j] @ omega
    return out


def _clean(Z, dets, Ms):
    Zc = Z.copy()
    for d in dets:
        if d.kind == "AO":
            Zc[d.t] -= d.omega
        else:
            P = psi_weights(Ms, len(Z) - d.t)
            Zc[d.t:] -= np.einsum("hij,j->hi", P, d.omega)
    return Zc


def _resid_from(Z, Ms):
    p = Ms.shape[0]
    Y = Z[p:].copy()
    for k in range(1, p + 1):
        Y -= Z[p - k:len(Z) - k] @ Ms[k - 1].T
    return Y


def iterative(Z, W, p=1, detector="full", fit_kind="gstar", alpha=0.05, c2=None,
              max_outliers=30, max_sweeps=10, rel_tol=1e-3):
    """Detect-adjust-refit procedure (Chang et al. 1988; Chen and Liu 1993)."""
    fit = fit_gstar_p(Z, W, p, fit_kind)
    T_eff = len(fit.resid)
    N = Z.shape[1]
    if c2 is None:
        c2 = threshold(detector, N, T_eff, alpha)
    dets, prev_keys, prev = [], set(), fit.Ms.ravel()
    history = []
    sweep = 0
    converged, hit_cap = False, False
    for sweep in range(1, max_sweeps + 1):
        Zc = _clean(Z, dets, fit.Ms) if dets else Z
        fit = fit_gstar_p(Zc, W, p, fit_kind)
        resid = _resid_from(Z, fit.Ms)
        Sigma = fit.Sigma.copy()
        new = []
        for _ in range(max_outliers):
            a, i, w, la, li = scan(resid, fit.Ms, Sigma, detector, alpha)
            m = np.maximum(a, i)
            t = int(np.argmax(m))
            if m[t] <= c2:
                break
            if detector == "combo":   # type from the likelihood-ratio statistics
                lam_a, lam_i, _ = stats_quadratic(resid, fit.Ms, Sigma)
                is_ao = lam_a[t] > lam_i[t]
            else:
                is_ao = a[t] > i[t]
            if is_ao:
                kind, omega = "AO", w[t].copy()
            else:
                kind, omega = "IO", resid[t].copy()
            if detector in ("full", "combo"):
                if kind == "AO":
                    Sinv = np.linalg.inv(Sigma)
                    J = min(p, len(resid) - 1 - t)
                    Pis = _pis(fit.Ms)
                    At = sum(Pis[j].T @ Sinv @ Pis[j] for j in range(J + 1))
                    se = np.sqrt(np.diag(np.linalg.inv(At)))
                else:
                    se = np.sqrt(np.diag(Sigma))
                ts = omega / se
                loc = int(np.argmax(np.abs(ts)))
            elif detector == "diag":
                ts = omega / np.sqrt(np.diag(Sigma))
                loc = int(np.argmax(np.abs(ts)))
            else:
                loc = int(la[t] if kind == "AO" else li[t])
                ts = omega / np.sqrt(np.diag(Sigma))
            new.append(Det(t + p, kind, float(a[t]), float(i[t]), omega, ts, loc))
            resid = _adjust(resid, fit.Ms, t, kind, omega)
            Sigma = np.cov(resid, rowvar=False, ddof=1)
            history.append(f"sweep {sweep}: {kind} at t={t + p}, stats AO={a[t]:.2f} IO={i[t]:.2f}")
        if len(new) >= max_outliers:          # cap reached: is there still an exceedance?
            a_, i_, *_ = scan(resid, fit.Ms, Sigma, detector, alpha)
            hit_cap = hit_cap or float(np.max(np.maximum(a_, i_))) > c2
        keys = {(d.t, d.kind) for d in new}
        cur = fit.Ms.ravel()
        change = np.max(np.abs(cur - prev)) if sweep > 1 else np.inf
        dets = new
        if keys == prev_keys and change < rel_tol:
            converged = True
            break
        prev_keys, prev = keys, cur
    Zc = _clean(Z, dets, fit.Ms) if dets else Z
    fit = fit_gstar_p(Zc, W, p, fit_kind)
    return Result(dets, fit, c2, sweep, history, converged, hit_cap)
