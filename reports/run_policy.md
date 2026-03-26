# Run Output Policy

Goal: outputs are stored under run-specific folders that uniquely identify dataset generation and algorithm configuration. This enables cache reuse and prevents accidental reuse when hyperparameters change.

## Required practice
- Always build a run-specific directory using a deterministic hash of:
  - dataset name
  - data generation parameters (noise, priors, seeds, etc.)
  - algorithm configuration (method, P, rho grid, M, n_iters, burn_in, etc.)
  - sweep configuration (if applicable)
- Store samples, metrics, diagnostics, and derived tables under the run-specific estimations directory.
- Store figures under the matching run-specific reports directory.
- Write the full configuration to config.json in the estimations directory.

## Figures
- Apply the publication style from reports/figure_style.py for all plots.

## Suggested helper
Use multiproposal.utils.run_paths.build_run_dirs to build:
- estimations_dir = repo_root/estimations/<dataset>/<algorithm>/<run_name>
- reports_dir = repo_root/reports/<dataset>/<algorithm>/<run_name>

The run name includes a short hash so changes to any config produce a new folder.
