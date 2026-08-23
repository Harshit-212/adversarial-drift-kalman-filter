"""Summary statistics for experiment results."""

from typing import Dict, Tuple

import numpy as np


def summarize(data: np.ndarray) -> Tuple[float, float]:
    """Return (mean, std) of a 1D array of per-run cumulative errors."""
    return float(np.mean(data)), float(np.std(data))


def percent_reduction(reference: np.ndarray, proposed: np.ndarray) -> float:
    """Percent reduction in mean cumulative error of `proposed` relative to
    `reference` (e.g. proposed bias-aware filter vs. standard Kalman filter).
    """
    ref_mean = np.mean(reference)
    return float((ref_mean - np.mean(proposed)) / ref_mean * 100.0)


def format_report(results: Dict[str, np.ndarray]) -> str:
    """Build a human-readable results table matching the paper's reporting
    format, including the percent reduction of the proposed method over
    the standard Kalman filter baseline."""
    lines = ["===== FINAL RESULTS =====", ""]

    labels = {
        "kalman_filter": "Kalman Filter",
        "bias_aware_kalman_filter": "Proposed (Bias-Aware)",
        "baseline": "Baseline",
    }

    for key, label in labels.items():
        if key not in results:
            continue
        mean, std = summarize(results[key])
        lines.append(f"{label}:")
        lines.append(f"  Cumulative Error: ({mean:.2f}, {std:.2f})")
        lines.append("")

    if "kalman_filter" in results and "bias_aware_kalman_filter" in results:
        reduction = percent_reduction(results["kalman_filter"], results["bias_aware_kalman_filter"])
        lines.append(f"Reduction vs. Kalman Filter: {reduction:.1f}%")

    return "\n".join(lines)
