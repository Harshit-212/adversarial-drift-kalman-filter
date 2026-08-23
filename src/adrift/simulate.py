"""
Trajectory generation and experiment runners.

Each run uses an independent `numpy.random.RandomState`/`Generator` seeded
by the run index, so results are exactly reproducible given a seed list.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .config import SystemConfig
from .filters import baseline, bias_aware_kalman_filter_step, joseph_update, kalman_filter_step, particle_filter_step


@dataclass
class RunResult:
    """Per-run outputs: true trajectory plus per-method error traces."""

    x_true: np.ndarray
    err_kf: np.ndarray
    err_bias_aware: np.ndarray
    err_baseline: np.ndarray
    err_particle: Optional[np.ndarray] = None
    kf_cov_trace: Optional[np.ndarray] = None


def generate_trajectory(cfg: SystemConfig, rng: np.random.RandomState):
    """Simulate the true (2D) state trajectory and biased measurements.

    Returns
    -------
    x : (T, 2) array of true states.
    y : (T, 1) array of measurements, each corrupted by zero-mean noise
        plus the persistent additive bias `cfg.delta`.
    """
    T, n, m = cfg.T, cfg.state_dim, cfg.obs_dim
    x = np.zeros((T, n))
    y = np.zeros((T, m))

    for t in range(T):
        w = rng.multivariate_normal(np.zeros(n), cfg.Q)
        v = rng.multivariate_normal(np.zeros(m), cfg.R)
        y[t] = (cfg.C @ x[t]).reshape(m) + v + cfg.delta
        if t < T - 1:
            x[t + 1] = cfg.A @ x[t] + w

    return x, y


def run_single_seed(seed: int, cfg: SystemConfig, include_particle_filter: bool = False) -> RunResult:
    """Run every estimator once over a full trajectory for a single seed."""
    rng = np.random.RandomState(seed)
    x, y = generate_trajectory(cfg, rng)

    T = cfg.T
    Q_aug = cfg.Q_aug

    x_kf = np.zeros(cfg.state_dim)
    P_kf = np.eye(cfg.state_dim)

    x_aug = np.zeros(cfg.aug_state_dim)
    P_aug = np.eye(cfg.aug_state_dim)

    err_kf = np.zeros(T)
    err_bias_aware = np.zeros(T)
    err_baseline = np.zeros(T)
    kf_cov_trace = np.zeros(T)

    err_particle = np.zeros(T) if include_particle_filter else None
    if include_particle_filter:
        pf_rng = np.random.default_rng(seed)
        particles = np.zeros((cfg.n_particles, cfg.state_dim))
        weights = np.ones(cfg.n_particles) / cfg.n_particles

    for t in range(T):
        x_kf, P_kf = kalman_filter_step(x_kf, P_kf, y[t], cfg.A, cfg.C, cfg.Q, cfg.R)
        x_aug, P_aug = bias_aware_kalman_filter_step(x_aug, P_aug, y[t], cfg.A_aug, cfg.C_aug, Q_aug, cfg.R)
        x_base = baseline(y[t])

        err_kf[t] = np.linalg.norm(x[t] - x_kf)
        err_bias_aware[t] = np.linalg.norm(x[t] - x_aug[:2])
        err_baseline[t] = np.linalg.norm(x[t] - x_base)
        kf_cov_trace[t] = np.trace(P_kf)

        if include_particle_filter:
            particles, weights, x_pf = particle_filter_step(
                particles, weights, y[t], cfg.A, cfg.C, cfg.Q, cfg.R, cfg.n_particles, pf_rng
            )
            err_particle[t] = np.linalg.norm(x[t] - x_pf)

    return RunResult(
        x_true=x,
        err_kf=err_kf,
        err_bias_aware=err_bias_aware,
        err_baseline=err_baseline,
        err_particle=err_particle,
        kf_cov_trace=kf_cov_trace,
    )


def run_multi_seed(seeds, cfg: SystemConfig) -> Dict[str, np.ndarray]:
    """Run the cumulative-error experiment across many seeds.

    Returns a dict of arrays, each of shape (n_seeds,), holding the total
    (summed-over-time) error per run for each estimator.
    """
    cum_kf, cum_bias_aware, cum_baseline = [], [], []

    for seed in seeds:
        result = run_single_seed(seed, cfg, include_particle_filter=False)
        cum_kf.append(result.err_kf.sum())
        cum_bias_aware.append(result.err_bias_aware.sum())
        cum_baseline.append(result.err_baseline.sum())

    return {
        "kalman_filter": np.array(cum_kf),
        "bias_aware_kalman_filter": np.array(cum_bias_aware),
        "baseline": np.array(cum_baseline),
    }
