"""Multivariate AO/IO detection statistics for GSTAR(1;1) — corrected GLS form.

Residual signatures of an outlier with effect vector omega at time T,
where e_t = phi(B) Psi_t and M = Phi0 + Phi1 W:

    AO:  e_T = eps_T + omega,      e_{T+1} = eps_{T+1} - M omega
    IO:  e_T = eps_T + omega       (single point)

GLS estimators (stacking y = [e_T; e_{T+1}], X = [I; -M], V = diag(Sigma, Sigma)):

    omega_IO = e_T                          Var(omega_IO) = Sigma
    omega_AO = A^{-1} (Sigma^{-1} e_T - M' Sigma^{-1} e_{T+1})
    A        = Sigma^{-1} + M' Sigma^{-1} M Var(omega_AO) = A^{-1}

NOTE — this replaces the thesis formulas: the Gram matrix is I + M'M in
standardized form (transpose-square), NOT I + M^2; and the AO numerator uses
the forward-filtered residual, not e_T alone.

Test statistics, each chi^2_N under H0 (Gaussian errors, parameters known):

    lam2_IO(t) = e_t' Sigma^{-1} e_t
    lam2_AO(t) = omega_AO' A omega_AO

Critical value: Bonferroni over the 2 * T_eff tests in one sweep,
c2 = chi2.ppf(1 - alpha / (2 * T_eff), N). With estimated parameters the
chi^2 null is approximate — empirical size is quantified in exp02.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .model import GstarFit, fit_gstar


@dataclass
class Detection:
    t: int                    # residual index (residual k corresponds to series time k+1)
    kind: str                 # "AO" or "IO"
    lam2_ao: float
    lam2_io: float
    omega: np.ndarray         # GLS effect estimate for the chosen kind
    omega_se: np.ndarray      # per-location std. errors of omega
    tstats: np.ndarray        # per-location omega_j / se_j  (spatial localization)


@dataclass
class IterativeResult:
    detections: list[Detection]
    fit: GstarFit             # final fit after series cleaning
    c2: float                 # squared critical value used
    n_sweeps: int
    history: list[str] = field(default_factory=list)


def outlier_statistics(resid: np.ndarray, M: np.ndarray, Sigma: np.ndarray):
    """lam2_AO, lam2_IO arrays over all residual time points, plus reusable pieces.

    At the last point the AO signature truncates to the IO one (J=0).
    """
    Tr, N = resid.shape
    Sinv = np.linalg.inv(Sigma)
    A = Sinv + M.T @ Sinv @ M
    Ainv = np.linalg.inv(A)
    lam2_ao = np.empty(Tr)
    lam2_io = np.empty(Tr)
    omega_ao = np.empty((Tr, N))
    for t in range(Tr):
        e_t = resid[t]
        lam2_io[t] = e_t @ Sinv @ e_t
        if t < Tr - 1:
            w = Ainv @ (Sinv @ e_t - M.T @ Sinv @ resid[t + 1])
            omega_ao[t] = w
            lam2_ao[t] = w @ A @ w
        else:  # end-of-sample truncation: AO estimator == IO estimator
            omega_ao[t] = e_t
            lam2_ao[t] = lam2_io[t]
    return lam2_ao, lam2_io, omega_ao, A, Ainv, Sinv


def bonferroni_c2(N: int, T_eff: int, alpha: float = 0.05) -> float:
    """Squared critical value controlling FWER at alpha over 2*T_eff tests."""
    return float(stats.chi2.ppf(1.0 - alpha / (2.0 * T_eff), df=N))


def detect_once(resid, M, Sigma, c2) -> Detection | None:
    """One detection step: the single most extreme exceedance, or None."""
    lam2_ao, lam2_io, omega_ao, A, Ainv, Sinv = outlier_statistics(resid, M, Sigma)
    lam2_max = np.maximum(lam2_ao, lam2_io)
    t_star = int(np.argmax(lam2_max))
    if lam2_max[t_star] <= c2:
        return None
    if lam2_ao[t_star] > lam2_io[t_star]:
        kind, omega, var = "AO", omega_ao[t_star], Ainv
    else:
        kind, omega, var = "IO", resid[t_star].copy(), Sigma
    se = np.sqrt(np.diag(var))
    return Detection(
        t=t_star, kind=kind,
        lam2_ao=float(lam2_ao[t_star]), lam2_io=float(lam2_io[t_star]),
        omega=omega, omega_se=se, tstats=omega / se,
    )


def adjust_residuals(resid: np.ndarray, det: Detection, M: np.ndarray) -> np.ndarray:
    """Remove the estimated outlier signature from the residual series."""
    out = resid.copy()
    out[det.t] -= det.omega
    if det.kind == "AO" and det.t + 1 < len(out):
        out[det.t + 1] += M @ det.omega
    return out


def clean_series(Z: np.ndarray, detections: list[Detection], M: np.ndarray) -> np.ndarray:
    """Subtract estimated outlier effects from the OBSERVED series.

    Residual index k corresponds to series time k+1 (fit_gstar loses t=0).
    AO: remove omega at the single observation.
    IO: remove the propagated effect M^j omega for all later times.
    """
    Zc = Z.copy()
    for d in detections:
        t0 = d.t + 1
        if d.kind == "AO":
            Zc[t0] -= d.omega
        else:
            effect = d.omega.copy()
            for t in range(t0, len(Z)):
                Zc[t] -= effect
                effect = M @ effect
    return Zc


def iterative_detection(
    Z: np.ndarray,
    W: np.ndarray,
    alpha: float = 0.05,
    max_outliers: int = 20,
    max_sweeps: int = 10,
    rel_tol: float = 1e-3,
    fit_fn=fit_gstar,
) -> IterativeResult:
    """Full iterative procedure (Chen-Liu-style alternation).

    Inner loop: detect one outlier at a time, adjust the residual signature,
    re-estimate Sigma from adjusted residuals, repeat.
    Outer loop (sweep): clean the observed series of all estimated effects,
    re-fit (Phi0, Phi1, Sigma) on the CLEANED series, then re-run detection on
    the ORIGINAL series' residuals computed with the refined parameters (the
    original series still contains the outliers — the cleaned series is only
    used to get uncontaminated parameter/covariance estimates). Stop when a
    sweep reproduces the same set of (t, kind) and parameters move < rel_tol.
    """
    history: list[str] = []
    fit = fit_fn(Z, W)
    T_eff = len(fit.residuals)
    N = Z.shape[1]
    c2 = bonferroni_c2(N, T_eff, alpha)

    detections: list[Detection] = []
    prev_keys: set[tuple[int, str]] = set()
    prev_params = fit.M.flatten()

    for sweep in range(1, max_sweeps + 1):
        # parameters and Sigma from the series cleaned of current effects...
        Zc = clean_series(Z, detections, fit.M) if detections else Z
        fit = fit_fn(Zc, W)
        # ...but detection runs on the ORIGINAL series' residuals
        resid = Z[1:] - Z[:-1] @ fit.M.T
        Sigma = fit.Sigma.copy()

        sweep_dets: list[Detection] = []
        for _ in range(max_outliers):
            det = detect_once(resid, fit.M, Sigma, c2)
            if det is None:
                break
            sweep_dets.append(det)
            resid = adjust_residuals(resid, det, fit.M)
            Sigma = np.cov(resid, rowvar=False, ddof=1)  # re-estimate after adjustment
            history.append(
                f"sweep {sweep}: {det.kind} at residual t={det.t} "
                f"(series t={det.t+1}), lam2_AO={det.lam2_ao:.2f}, "
                f"lam2_IO={det.lam2_io:.2f}, c2={c2:.2f}"
            )

        keys = {(d.t, d.kind) for d in sweep_dets}
        params = fit.M.flatten()
        param_change = np.max(np.abs(params - prev_params)) if sweep > 1 else np.inf
        detections = sweep_dets
        if keys == prev_keys and param_change < rel_tol:
            break
        prev_keys, prev_params = keys, params

    # final re-fit on fully cleaned series
    Zc = clean_series(Z, detections, fit.M) if detections else Z
    fit = fit_fn(Zc, W)
    return IterativeResult(detections, fit, c2, sweep, history)
