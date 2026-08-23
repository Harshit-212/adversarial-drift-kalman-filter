"""
Default system definition and experiment configuration.

The true system is a discrete-time constant-velocity model:

    x_{t+1} = A x_t + w_t,      w_t ~ N(0, Q)
    y_t     = C x_t + v_t + delta,   v_t ~ N(0, R)

where `delta` is a persistent (adversarial) additive measurement bias
that is NOT modeled by a standard Kalman filter. The augmented model
below adds a third state dimension representing this bias so that a
bias-aware Kalman filter can jointly estimate it.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    """Container for system matrices and experiment-wide constants."""

    # -- Horizon / adversarial bias --------------------------------------
    T: int = 200
    delta: float = 0.5          # persistent additive measurement bias
    n_particles: int = 100      # particle filter population size

    # -- True (2D: position, velocity) system -----------------------------
    A: np.ndarray = field(default_factory=lambda: np.array([[1.0, 1.0],
                                                              [0.0, 1.0]]))
    C: np.ndarray = field(default_factory=lambda: np.array([[1.0, 0.0]]))
    Q: np.ndarray = field(default_factory=lambda: 0.01 * np.eye(2))
    R: np.ndarray = field(default_factory=lambda: np.array([[0.1]]))

    # -- Augmented (3D: position, velocity, bias) system -------------------
    # The bias is modeled as a (near) constant latent state, matching the
    # true generative process where `delta` does not change over time.
    A_aug: np.ndarray = field(default_factory=lambda: np.array([[1.0, 1.0, 0.0],
                                                                  [0.0, 1.0, 0.0],
                                                                  [0.0, 0.0, 1.0]]))
    C_aug: np.ndarray = field(default_factory=lambda: np.array([[1.0, 0.0, 1.0]]))

    @property
    def Q_aug(self) -> np.ndarray:
        """Process noise for the augmented system: no injected noise on the
        bias term itself (the bias is treated as constant, matching the
        true data-generating process)."""
        Q_aug = np.zeros((3, 3))
        Q_aug[:2, :2] = self.Q
        return Q_aug

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def aug_state_dim(self) -> int:
        return self.A_aug.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.C.shape[0]


DEFAULT_CONFIG = SystemConfig()

# Default multi-seed experiment settings
DEFAULT_N_RUNS = 50
DEFAULT_SEEDS = range(DEFAULT_N_RUNS)
