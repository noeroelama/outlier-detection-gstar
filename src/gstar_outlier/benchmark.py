"""Per-location (scalar-variance) benchmark detector, Huda et al. (2022) style.

Huda, Mukhaiyar & Imro'ah (2022, BAREKENG 16(3):975-984) detect AO/IO in
GSTAR(1;1) with location-wise statistics standardized by a scalar residual
standard deviation — cross-location covariance never enters the test.

To isolate exactly that difference (and nothing else), this benchmark uses
THE SAME residual signatures, filtering matrix M, and iterative sweeps as
our multivariate detector, but standardizes per location with diag(Sigma)
only:

    IO:  lam_IO(i,t) = e_{i,t} / sigma_i                     ~ N(0,1) under H0
    AO:  omega_A(i,t) via the same GLS numerator but with Sigma replaced by
         diag(sigma_1^2..sigma_N^2);  lam_AO(i,t) = omega_A_i / se_i

Detection: max over locations AND times; Bonferroni threshold over
2 * N * T_eff one-dimensional tests at the same nominal alpha, so both
detectors control the same family-wise error rate (verified empirically in
exp03). Location is available directly (argmax over i)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .detection import Detection, IterativeResult, adjust_residuals, clean_series
from .model import fit_gstar


def perloc_statistics(resid: np.ndarray, M: np.ndarray, Sigma: np.ndarray):
    """Per-location AO/IO z-statistics using diagonal covariance only."""
    Tr, N = resid.shape
    D = np.diag(np.diag(Sigma))
    Dinv = np.linalg.inv(D)
    A = Dinv + M.T @ Dinv @ M            # diagonal-Sigma Gram matrix
    Ainv = np.linalg.inv(A)
    se_ao = np.sqrt(np.diag(Ainv))
    se_io = np.sqrt(np.diag(D))
    z_ao = np.empty((Tr, N))
    z_io = np.empty((Tr, N))
    omega_ao = np.empty((Tr, N))
    for t in range(Tr):
        e_t = resid[t]
        z_io[t] = e_t / se_io
        if t < Tr - 1:
            w = Ainv @ (Dinv @ e_t - M.T @ Dinv @ resid[t + 1])
        else:
            w = e_t
        omega_ao[t] = w
        z_ao[t] = w / se_ao
    return z_ao, z_io, omega_ao, Ainv, D


def bonferroni_z(N: int, T_eff: int, alpha: float = 0.05) -> float:
    """|z| threshold controlling FWER over 2 * N * T_eff univariate tests."""
    return float(stats.norm.ppf(1.0 - alpha / (2.0 * 2.0 * N * T_eff)))


def detect_once_perloc(resid, M, Sigma, zcrit) -> Detection | None:
    z_ao, z_io, omega_ao, Ainv, D = perloc_statistics(resid, M, Sigma)
    a_max = np.max(np.abs(z_ao)); i_a = np.unravel_index(np.argmax(np.abs(z_ao)), z_ao.shape)
    i_max = np.max(np.abs(z_io)); i_i = np.unravel_index(np.argmax(np.abs(z_io)), z_io.shape)
    if max(a_max, i_max) <= zcrit:
        return None
    if a_max >= i_max:
        t_star = int(i_a[0])
        kind, omega, var = "AO", omega_ao[t_star], Ainv
        za, zi = a_max, np.max(np.abs(z_io[t_star]))
    else:
        t_star = int(i_i[0])
        kind, omega, var = "IO", resid[t_star].copy(), D
        za, zi = np.max(np.abs(z_ao[t_star])), i_max
    se = np.sqrt(np.diag(var))
    return Detection(t=t_star, kind=kind, lam2_ao=float(za**2), lam2_io=float(zi**2),
                     omega=omega, omega_se=se, tstats=omega / se)


def iterative_detection_perloc(
    Z: np.ndarray,
    W: np.ndarray,
    alpha: float = 0.05,
    max_outliers: int = 20,
    max_sweeps: int = 10,
    rel_tol: float = 1e-3,
) -> IterativeResult:
    """Same sweep architecture as `iterative_detection`, per-location statistics."""
    history: list[str] = []
    fit = fit_gstar(Z, W)
    T_eff = len(fit.residuals)
    N = Z.shape[1]
    zc = bonferroni_z(N, T_eff, alpha)

    detections: list[Detection] = []
    prev_keys: set[tuple[int, str]] = set()
    prev_params = np.concatenate([fit.phi0, fit.phi1])

    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        Zc = clean_series(Z, detections, fit.M) if detections else Z
        fit = fit_gstar(Zc, W)
        resid = Z[1:] - Z[:-1] @ fit.M.T
        Sigma = fit.Sigma.copy()

        sweep_dets: list[Detection] = []
        for _ in range(max_outliers):
            det = detect_once_perloc(resid, fit.M, Sigma, zc)
            if det is None:
                break
            sweep_dets.append(det)
            resid = adjust_residuals(resid, det, fit.M)
            Sigma = np.cov(resid, rowvar=False, ddof=1)
            history.append(f"sweep {sweep}: {det.kind} at residual t={det.t}")

        keys = {(d.t, d.kind) for d in sweep_dets}
        params = np.concatenate([fit.phi0, fit.phi1])
        param_change = np.max(np.abs(params - prev_params)) if sweep > 1 else np.inf
        detections = sweep_dets
        if keys == prev_keys and param_change < rel_tol:
            break
        prev_keys, prev_params = keys, params

    Zc = clean_series(Z, detections, fit.M) if detections else Z
    fit = fit_gstar(Zc, W)
    return IterativeResult(detections, fit, zc**2, sweep, history)
