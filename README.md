# Adversarial Drift in Sequential Inference Systems: A Hidden Failure Mode in Bayesian Estimators

Reproducible code accompanying the paper of the same name. This repository
demonstrates that a persistent (adversarial) additive measurement bias
causes a standard Kalman filter to fail *silently* — its reported error
covariance stays bounded even as its true tracking error grows — and shows
that augmenting the state with a latent bias term recovers substantially
lower error.

Submitted for anonymous peer review; all author-identifying metadata has
been removed from this repository.

## Key result

| Estimator                     | Cumulative Error (mean ± std, 50 seeds, T=200, δ=0.5) |
|-------------------------------|--------------------------------------------------------|
| Naive baseline (unfiltered)   | 197.0 ± 69.1                                            |
| Standard Kalman filter        | 107.6 ± 4.9                                             |
| **Proposed (bias-aware KF)**  | **75.5 ± 17.6**                                         |

The proposed bias-aware filter reduces cumulative error by **~30%** relative
to the standard Kalman filter under a fixed bias of δ = 0.5, with the
advantage growing as the bias magnitude increases (see
`notebooks/adversarial_drift_analysis.ipynb`, Section 4).

## Problem setup

The true generative process is a discrete-time constant-velocity model with
a persistent additive bias on the measurement channel:

```
x_{t+1} = A x_t + w_t,          w_t ~ N(0, Q)
y_t     = C x_t + v_t + delta,  v_t ~ N(0, R)
```

A standard Kalman filter assumes `delta = 0` and has no state dimension to
absorb it, so the bias leaks directly into the position estimate at every
time step — while the filter's own covariance estimate, which only tracks
*noise*, not *unmodeled bias*, remains bounded and gives no warning that
anything is wrong. This is the "hidden failure mode" the paper studies.

The proposed fix augments the state with a third, (near-)constant latent
bias dimension:

```
state = [position, velocity, bias]
```

so the filter can jointly estimate and subtract off the persistent offset.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── src/
│   └── adrift/
│       ├── __init__.py
│       ├── config.py       # system matrices & experiment configuration
│       ├── filters.py      # baseline, Kalman filter, bias-aware KF, particle filter
│       ├── simulate.py     # trajectory generation, single/multi-seed runners
│       ├── metrics.py      # summary statistics / reporting
│       └── plotting.py     # figure generation
├── scripts/
│   └── run_experiment.py   # CLI: reproduce the main experiment + figures
├── notebooks/
│   └── adversarial_drift_analysis.ipynb   # end-to-end reproducible analysis
├── tests/
│   └── test_filters.py     # correctness & reproducibility tests
└── results/                # figures are written here
```

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate adrift
```

## Reproducing the results

**Command line** (prints the results table and writes figures to `results/`):

```bash
python scripts/run_experiment.py
```

Options:

```bash
python scripts/run_experiment.py --n-runs 50 --T 200 --delta 0.5 --viz-seed 0 --output-dir results
```

**Notebook** (same experiment plus a bias-magnitude sensitivity sweep):

```bash
jupyter notebook notebooks/adversarial_drift_analysis.ipynb
```

**Tests**:

```bash
pip install pytest
pytest tests/
```

## Method summary

Four estimators are compared (see `src/adrift/filters.py`):

1. **Baseline** — raw, unfiltered measurement (assumes zero velocity).
2. **Standard Kalman filter** — correctly specified for the kinematic
   dynamics, but *misspecified* with respect to the bias term.
3. **Bias-aware (augmented) Kalman filter** — the proposed method; adds a
   latent, near-constant bias state so the persistent offset is explicitly
   modeled and estimated rather than absorbed as spurious innovation.
4. **Particle filter** — a sequential Monte Carlo comparison point, included
   in `filters.py` and `simulate.run_single_seed(..., include_particle_filter=True)`
   for completeness; like the standard KF, it has no mechanism to represent
   a persistent bias unless its state space is likewise augmented.

All covariance updates use the numerically stable Joseph-form update
(`joseph_update` in `filters.py`) to avoid covariance collapse over long
horizons. All simulations use per-run seeded random number generators
(`numpy.random.RandomState(seed)`), so every reported number is exactly
reproducible.

## Citation

This work is under anonymous peer review. A citation entry will be added
upon publication.

## License

Released under the MIT License — see `LICENSE`.
