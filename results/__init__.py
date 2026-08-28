"""
adrift: Adversarial Drift in Sequential Inference Systems
-----------------------------------------------------------
A small research library for studying how persistent (adversarial)
measurement bias silently degrades Bayesian sequential estimators
(Kalman filters, particle filters) and how an augmented, bias-aware
Kalman filter can recover consistent estimates under such drift.

Modules
-------
config     : System matrices and default experiment configuration.
filters    : Estimator implementations (baseline, KF, bias-aware KF, PF).
simulate   : Trajectory generation and single/multi-seed experiment runners.
metrics    : Summary statistics for error trajectories.
plotting   : Plot helpers for error, cumulative error, and covariance traces.
"""

from . import config, filters, metrics, plotting, simulate

__all__ = ["config", "filters", "metrics", "plotting", "simulate"]
__version__ = "1.0.0"
