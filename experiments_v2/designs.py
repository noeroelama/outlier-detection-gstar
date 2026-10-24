"""Simulation designs shared by exp10-exp14 (revision, Oct 2026).

Locations: N points drawn once (fixed seed) uniformly on the unit square.
Weights:   row-standardized inverse-distance W (the fitted model uses the same W
           unless a misspecification experiment says otherwise).
Errors:    exponential spatial correlation Corr(eps_i, eps_j) = exp(-d_ij / phi),
           unit variances; phi is solved so that the mean off-diagonal correlation
           equals the target level (low 0.1, moderate 0.35, high 0.65).
Parameters: location-specific phi_i0 ~ U(0.2, 0.5), phi_i1 ~ U(0.1, 0.3), fixed seed;
           stationarity checked (spectral radius of M < 1).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

CORR_LEVELS = {"none": 0.0, "low": 0.10, "moderate": 0.35, "high": 0.65}


def random_locations(N: int, seed: int = 2026) -> np.ndarray:
    return np.random.default_rng(seed + N).uniform(0, 1, size=(N, 2))


def distance_matrix(C: np.ndarray) -> np.ndarray:
    return np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))


def W_inverse_distance(C: np.ndarray) -> np.ndarray:
    D = distance_matrix(C)
    W = np.where(D > 0, 1.0 / np.where(D > 0, D, 1.0), 0.0)
    return W / W.sum(1, keepdims=True)


def W_knn(C: np.ndarray, k: int = 2) -> np.ndarray:
    D = distance_matrix(C)
    N = len(C)
    W = np.zeros((N, N))
    for i in range(N):
        nn = np.argsort(D[i])[1:k + 1]
        W[i, nn] = 1.0
    return W / W.sum(1, keepdims=True)


def W_uniform(N: int) -> np.ndarray:
    W = np.ones((N, N)) - np.eye(N)
    return W / W.sum(1, keepdims=True)


def sigma_exponential(C: np.ndarray, level: str) -> np.ndarray:
    target = CORR_LEVELS[level]
    D = distance_matrix(C)
    N = len(C)
    if target == 0:
        return np.eye(N)
    iu = np.triu_indices(N, 1)
    f = lambda phi: np.exp(-D[iu] / phi).mean() - target
    phi = brentq(f, 1e-4, 1e4)
    return np.exp(-D / phi)


def draw_coefficients(N: int, W: np.ndarray, seed: int = 7):
    rng = np.random.default_rng(seed + N)
    for _ in range(1000):
        phi0 = rng.uniform(0.2, 0.5, N)
        phi1 = rng.uniform(0.1, 0.3, N)
        M = np.diag(phi0) + np.diag(phi1) @ W
        if np.max(np.abs(np.linalg.eigvals(M))) < 0.95:
            return phi0, phi1, M
    raise RuntimeError("no stationary draw")


def make_design(N: int, level: str):
    C = random_locations(N)
    W = W_inverse_distance(C)
    Sigma = sigma_exponential(C, level)
    _, _, M = draw_coefficients(N, W)
    return C, W, Sigma, M[None]          # Ms with p = 1


def nearest_neighbours(C: np.ndarray, j: int, k: int = 2) -> list[int]:
    return list(np.argsort(distance_matrix(C)[j])[1:k + 1])
