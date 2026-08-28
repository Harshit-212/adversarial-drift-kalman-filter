#!/usr/bin/env python3
"""
Reproduce the main multi-seed experiment and figures.

Usage
-----
    python scripts/run_experiment.py
    python scripts/run_experiment.py --n-runs 50 --T 200 --delta 0.5 --viz-seed 0

Outputs
-------
Prints a results table (mean/std cumulative error per estimator, and the
percent error reduction of the proposed bias-aware filter vs. the standard
Kalman filter) and writes three figures to `--output-dir` (default:
`results/`):

    comparison_error.png       instantaneous error vs. time (single run)
    comparison_cumulative.png  cumulative error vs. time (single run)
    covariance_plot.png        Kalman filter covariance trace (single run)
"""

import argparse
import os
import sys

# Allow running directly from a source checkout without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adrift.config import SystemConfig  # noqa: E402
from adrift.metrics import format_report  # noqa: E402
from adrift.plotting import plot_covariance_trace, plot_cumulative_error, plot_instant_error  # noqa: E402
from adrift.simulate import run_multi_seed, run_single_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-runs", type=int, default=50, help="Number of seeds for the multi-seed experiment.")
    parser.add_argument("--T", type=int, default=200, help="Trajectory length (time steps).")
    parser.add_argument("--delta", type=float, default=0.5, help="Persistent additive measurement bias.")
    parser.add_argument("--viz-seed", type=int, default=0, help="Seed used for the single-run visualization.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to write figures to.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = SystemConfig(T=args.T, delta=args.delta)

    # -- Multi-seed cumulative error experiment --------------------------
    results = run_multi_seed(range(args.n_runs), cfg)
    print(format_report(results))

    # -- Single-run visualization ------------------------------------------
    single = run_single_seed(args.viz_seed, cfg)
    err_path = plot_instant_error(single.err_kf, single.err_bias_aware, args.output_dir)
    cum_path = plot_cumulative_error(single.err_kf, single.err_bias_aware, args.output_dir)
    cov_path = plot_covariance_trace(single.kf_cov_trace, args.output_dir)

    print("\nFigures written:")
    for path in (err_path, cum_path, cov_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
