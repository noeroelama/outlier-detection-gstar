"""GSTAR(1;1) simulation and least-squares estimation.

Model:
    Z_t = Phi0 Z_{t-1} + Phi1 W Z_{t-1} + eps_t,   eps_t ~ N(0, Sigma)

with Phi0, Phi1 diagonal (N x N), W row-standardized with zero diagonal.
Writing M = Phi0 + Phi1 W, the process is a restricted VAR(1):
    Z_t = M Z_{t-1} + eps_t
and is stationary iff the spectral radius of M is < 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def make_M(phi0: np.ndarray, phi1: np.ndarray, W: np.ndarray) -> np.ndarray:
    """M = Phi0 + Phi1 W from the diagonal parameter vectors."""
    return np.diag(phi0) + np.diag(phi1) @ W


def simulate_gstar(
    phi0: np.ndarray,
    phi1: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    T: int,
    burn: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a clean GSTAR(1;1) path of length T (rows = time)."""
    rng = rng if rng is not None else np.random.default_rng()
    N = len(phi0)
    M = make_M(phi0, phi1, W)
    rho = spectral_radius(M)
    if rho >= 1:
        raise ValueError(f"Non-stationary parameters: spectral radius {rho:.3f} >= 1")
    L = np.linalg.cholesky(Sigma)
    Z = np.zeros(N)
    out = np.empty((T, N))
    for t in range(burn + T):
        Z = M @ Z + L @ rng.standard_normal(N)
        if t >= burn:
            out[t - burn] = Z
    return out


def inject_outlier(
    Z: np.ndarray,
    M: np.ndarray,
    t0: int,
    omega: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Contaminate a clean series with one outlier of effect vector omega at t0.

    AO:  Psi_t = Z_t + I(t=t0) * omega            (observation-level blip)
    IO:  Psi_t = Z_t + M^{t-t0} omega  for t>=t0  (shock enters the dynamics:
         phi(B)^{-1} applied to an innovation impulse, Pi_k = M^k)
    """
    Psi = Z.copy()
    if kind == "AO":
        Psi[t0] += omega
    elif kind == "IO":
        effect = omega.copy()
        for t in range(t0, len(Z)):
            Psi[t] += effect
            effect = M @ effect
    else:
        raise ValueError(f"kind must be 'AO' or 'IO', got {kind!r}")
    return Psi


@dataclass
class GstarFit:
    phi0: np.ndarray          # (N,) diagonal of Phi0
    phi1: np.ndarray          # (N,) diagonal of Phi1
    M: np.ndarray             # (N, N) Phi0 + Phi1 W
    Sigma: np.ndarray         # (N, N) residual covariance (ML, divisor T-1)
    residuals: np.ndarray     # (T-1, N); residuals[k] = e_{k+1} (no residual at t=0)
    se_phi0: np.ndarray       # (N,) OLS standard errors
    se_phi1: np.ndarray


def fit_gstar(Z: np.ndarray, W: np.ndarray) -> GstarFit:
    """Location-by-location OLS for GSTAR(1;1) on a zero-mean series.

    For each location i:
        Z_{i,t} = phi0_i * Z_{i,t-1} + phi1_i * V_{i,t-1} + e_{i,t},
    with V = Z W' (spatially weighted neighbors). No intercept: the model is
    intended for centered/differenced data. Center the series beforehand if
    needed (the caller's responsibility, kept explicit to avoid the thesis'
    intercept inconsistency between estimation and prediction).
    """
    T, N = Z.shape
    Y = Z[1:]                  # t = 1..T-1
    X1 = Z[:-1]                # own lag
    X2 = Z[:-1] @ W.T          # weighted-neighbor lag
    phi0 = np.empty(N)
    phi1 = np.empty(N)
    se0 = np.empty(N)
    se1 = np.empty(N)
    resid = np.empty_like(Y)
    for i in range(N):
        Xi = np.column_stack([X1[:, i], X2[:, i]])
        beta, _, _, _ = np.linalg.lstsq(Xi, Y[:, i], rcond=None)
        phi0[i], phi1[i] = beta
        r = Y[:, i] - Xi @ beta
        resid[:, i] = r
        dof = len(r) - 2
        s2 = r @ r / dof
        XtX_inv = np.linalg.inv(Xi.T @ Xi)
        se0[i], se1[i] = np.sqrt(s2 * np.diag(XtX_inv))
    M = make_M(phi0, phi1, W)
    Sigma = np.cov(resid, rowvar=False, ddof=1)
    return GstarFit(phi0, phi1, M, Sigma, resid, se0, se1)


def fit_var1(Z: np.ndarray, W: np.ndarray | None = None) -> GstarFit:
    """Unrestricted VAR(1) OLS fit, returned in GstarFit form.

    Basis for the Tsay-Pena-Pankratz (2000)-style benchmark detector: the
    same outlier signatures and GLS statistics, but with the full N x N
    coefficient matrix estimated (N^2 parameters vs 2N for GSTAR). W is
    ignored (kept in the signature so fitters are interchangeable).
    phi0 is reported as diag(Phi) and phi1 as zeros; use .M for the full
    coefficient matrix.
    """
    Y = Z[1:]
    A = Z[:-1]
    Phi_T, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)   # Y ~ A @ Phi'
    M = Phi_T.T
    resid = Y - A @ Phi_T
    Sigma = np.cov(resid, rowvar=False, ddof=1)
    N = Z.shape[1]
    zeros = np.zeros(N)
    return GstarFit(np.diag(M).copy(), zeros, M, Sigma, resid, zeros, zeros)
