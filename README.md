# MESS: Multiproposal Elliptical Slice Sampling

Python implementation of Elliptical Slice Sampling (ESS) and its generalization MESS for Bayesian inference with Gaussian priors and non-Gaussian likelihoods.

## Quick Start

See the `notebooks/` directory for examples:

- **01_gp_regression_sanity.ipynb** - Basic GP regression example
- **02_gp_regression_ess_vs_mess.ipynb** - ESS vs MESS comparison on GP regression
- **03_gp_regression_distances.ipynb** - Distance metrics in MESS
- **04_gp_regression_diagnostics.ipynb** - Convergence diagnostics
- **05_linear_programming_for_transition_matrix.ipynb** - LP-based transition matrix
- **06_mess_uniform_varying_m.ipynb** - MESS performance with varying M
- **07_lambda_sensitivity.ipynb** - Sensitivity analysis
- **08_logistic_regression_ess_mess.ipynb** - Bayesian logistic regression example

## Installation

```bash
pip install -e .
```

## Features

- ESS and MESS algorithms with multiple distance metrics (uniform, angular, euclidean)
- Built-in problems: GP regression, logistic regression, semi-blind deconvolution
- Efficient sampling from Gaussian priors via Cholesky decomposition
- ESS/MSJD diagnostics for sampler efficiency

See `README_context.md` for detailed design documentation.