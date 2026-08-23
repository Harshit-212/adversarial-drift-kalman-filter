"""Plot helpers for error curves and covariance traces.

All functions save a PNG to `output_dir` and return the saved path. No
`plt.show()` calls are made so these are safe to call from non-interactive
(headless) contexts such as CI or a batch script.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_instant_error(err_kf, err_bias_aware, output_dir=".", filename="comparison_error.png"):
    plt.figure()
    plt.plot(err_kf, label="Kalman Filter")
    plt.plot(err_bias_aware, label="Proposed (Bias-Aware)")
    plt.title("Estimation Error under Persistent Bias")
    plt.xlabel("Time Step")
    plt.ylabel("Error (L2 Norm)")
    plt.legend()
    plt.grid(True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    return path


def plot_cumulative_error(err_kf, err_bias_aware, output_dir=".", filename="comparison_cumulative.png"):
    plt.figure()
    plt.plot(np.cumsum(err_kf), label="Kalman Filter")
    plt.plot(np.cumsum(err_bias_aware), label="Proposed (Bias-Aware)")
    plt.title("Cumulative Error under Persistent Bias")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Error")
    plt.legend()
    plt.grid(True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    return path


def plot_covariance_trace(kf_cov_trace, output_dir=".", filename="covariance_plot.png"):
    plt.figure()
    plt.plot(kf_cov_trace)
    plt.title("Kalman Filter Covariance Trace")
    plt.xlabel("Time Step")
    plt.ylabel("Trace(P)")
    plt.grid(True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    return path
