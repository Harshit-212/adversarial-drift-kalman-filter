"""
Estimator implementations.

Four estimators are provided, matched to the accompanying paper:

1. `baseline`                    -- direct-measurement, no filtering.
2. `kalman_filter_step`          -- standard (misspecified) Kalman filter,
                                     unaware of the persistent bias `delta`.
3. `bias_aware_kalman_filter_step` -- proposed fix: augments the state with
                                     a latent bias term so the filter can
                                     jointly estimate and subtract it off.
4. `particle_filter_step`        -- sequential Monte Carlo baseline, included
                                     for comparison; it is equally unaware of
                                     the bias unless the state is augmented.

All covariance updates use the numerically stable Joseph-form update to
guard against covariance collapse / indefiniteness over long horizons.
"""

from typing import Tuple

import numpy as np


def joseph_update(P: np.ndarray, K: np.ndarray, C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Numerically stable (Joseph form) covariance update.

    P_new = (I - K C) P (I - K C)^T + K R K^T

    This form guarantees P_new remains symmetric positive semi-definite
    even under floating point error, unlike the short-form update
    `(I - K C) P`.
    """
    I = np.eye(P.shape[0])
    IKC = I - K @ C
    return IKC @ P @ IKC.T + K @ R @ K.T


def baseline(y: np.ndarray) -> np.ndarray:
    """Naive estimator: treat the raw (biased) measurement as the position
    estimate and assume zero velocity. No filtering, no bias correction."""
    y = np.atleast_1d(y)
    return np.array([float(y[0]), 0.0])


def kalman_filter_step(
    x: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """One predict-update cycle of a standard linear Kalman filter.

    This filter is *misspecified* with respect to the true generative
    process whenever a persistent additive bias is present on `y`: it has
    no state dimension to absorb `delta`, so the bias leaks directly into
    the position estimate every step.
    """
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ np.linalg.inv(S)

    innovation = np.atleast_1d(y) - C @ x_pred
    x_new = x_pred + (K @ innovation).ravel()
    P_new = joseph_update(P_pred, K, C, R)

    return x_new, P_new


def bias_aware_kalman_filter_step(
    x: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    A_aug: np.ndarray,
    C_aug: np.ndarray,
    Q_aug: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """One predict-update cycle of the proposed bias-aware (augmented)
    Kalman filter.

    The state is augmented with a latent, (near) constant bias term so
    that the persistent measurement offset is explicitly modeled and
    estimated rather than absorbed as spurious innovation into the
    kinematic states. Returns the full augmented state; the caller should
    slice `x_new[:2]` to recover the (position, velocity) estimate.
    """
    x_pred = A_aug @ x
    P_pred = A_aug @ P @ A_aug.T + Q_aug

    S = C_aug @ P_pred @ C_aug.T + R
    K = P_pred @ C_aug.T @ np.linalg.inv(S)

    innovation = np.atleast_1d(y) - C_aug @ x_pred
    x_new = x_pred + (K @ innovation).ravel()
    P_new = joseph_update(P_pred, K, C_aug, R)

    return x_new, P_new


def particle_filter_step(
    particles: np.ndarray,
    weights: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One predict-update-resample cycle of a bootstrap particle filter.

    Included as a sequential-Monte-Carlo comparison point. Like the
    standard Kalman filter, this filter has no mechanism to represent a
    persistent additive bias unless its state space is likewise augmented.
    """
    process_noise = rng.multivariate_normal(np.zeros(A.shape[0]), Q, size=n_particles)
    particles = particles @ A.T + process_noise

    y = np.atleast_1d(y)
    R_inv = np.linalg.inv(R)
    diffs = (C @ particles.T).T - y  # (n_particles, obs_dim)
    log_likelihoods = -0.5 * np.einsum("ij,jk,ik->i", diffs, R_inv, diffs)
    log_likelihoods -= log_likelihoods.max()  # numerical stability
    weights = weights * np.exp(log_likelihoods)

    weights += 1e-300
    weights /= weights.sum()

    x_hat = np.average(particles, axis=0, weights=weights)

    indices = rng.choice(n_particles, size=n_particles, p=weights)
    particles = particles[indices]
    weights = np.ones(n_particles) / n_particles

    return particles, weights, x_hat
