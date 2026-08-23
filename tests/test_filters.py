"""
Basic correctness and reproducibility tests.

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adrift.config import SystemConfig  # noqa: E402
from adrift.filters import baseline, bias_aware_kalman_filter_step, joseph_update, kalman_filter_step  # noqa: E402
from adrift.metrics import percent_reduction, summarize  # noqa: E402
from adrift.simulate import generate_trajectory, run_multi_seed, run_single_seed  # noqa: E402


def test_joseph_update_symmetric_psd():
    rng = np.random.RandomState(0)
    P = np.eye(2)
    C = np.array([[1.0, 0.0]])
    R = np.array([[0.1]])
    K = np.array([[0.5], [0.1]])

    P_new = joseph_update(P, K, C, R)

    assert np.allclose(P_new, P_new.T)
    eigvals = np.linalg.eigvalsh(P_new)
    assert np.all(eigvals >= -1e-10)


def test_baseline_ignores_velocity():
    y = np.array([3.2])
    x_base = baseline(y)
    assert np.allclose(x_base, [3.2, 0.0])


def test_kalman_filter_step_shapes():
    cfg = SystemConfig()
    x = np.zeros(cfg.state_dim)
    P = np.eye(cfg.state_dim)
    y = np.array([1.0])

    x_new, P_new = kalman_filter_step(x, P, y, cfg.A, cfg.C, cfg.Q, cfg.R)

    assert x_new.shape == (cfg.state_dim,)
    assert P_new.shape == (cfg.state_dim, cfg.state_dim)


def test_bias_aware_filter_explains_persistent_offset():
    """Under a constant offset and near-zero process/measurement noise, the
    augmented filter's position + bias states should jointly account for
    the full measurement offset `delta`. (With a static, zero-velocity
    true state, position and bias are only identifiable through their sum,
    C_aug = [1, 0, 1]; this test checks that joint observability, not a
    unique split between the two.)"""
    cfg = SystemConfig(T=200, delta=0.7, Q=1e-6 * np.eye(2), R=np.array([[1e-6]]))

    x = np.zeros(cfg.aug_state_dim)
    P = np.eye(cfg.aug_state_dim)

    for _ in range(cfg.T):
        y = np.array([cfg.delta])  # noiseless biased measurement of zero true state
        x, P = bias_aware_kalman_filter_step(x, P, y, cfg.A_aug, cfg.C_aug, cfg.Q_aug, cfg.R)

    assert abs((x[0] + x[2]) - cfg.delta) < 1e-2


def test_generate_trajectory_reproducible():
    cfg = SystemConfig(T=50)
    x1, y1 = generate_trajectory(cfg, np.random.RandomState(42))
    x2, y2 = generate_trajectory(cfg, np.random.RandomState(42))

    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)


def test_run_single_seed_reproducible():
    cfg = SystemConfig(T=50)
    r1 = run_single_seed(7, cfg)
    r2 = run_single_seed(7, cfg)

    assert np.allclose(r1.err_kf, r2.err_kf)
    assert np.allclose(r1.err_bias_aware, r2.err_bias_aware)


def test_bias_aware_filter_beats_standard_kf_on_average():
    """Core empirical claim: under persistent bias, the proposed bias-aware
    filter achieves lower mean cumulative error than the standard KF."""
    cfg = SystemConfig(T=200, delta=0.5)
    results = run_multi_seed(range(20), cfg)

    mean_kf, _ = summarize(results["kalman_filter"])
    mean_aware, _ = summarize(results["bias_aware_kalman_filter"])

    assert mean_aware < mean_kf
    assert percent_reduction(results["kalman_filter"], results["bias_aware_kalman_filter"]) > 0
