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
- Reuse cached outputs when present: load existing samples/metrics from the run directory and skip re-running chains.

## Figures
- Apply the publication style from reports/figure_style.py for all plots.

## Naming conventions
- Save chain outputs as <method>_P{P}_rho{rho}_seed{seed}.npz (or <method>_rho{rho}_seed{seed}.npz for pCN).
- Save metrics next to the chain file as <method>_P{P}_rho{rho}_seed{seed}_metrics.json.
- Store algorithm-specific diagnostics (e.g., mPCN diagnostics) alongside the chain files.

## Variants and sub-experiments
- For secondary experiments that share the same base run configuration (e.g., random-start sweeps), save under a subfolder of the main run directory (for example, estimations_dir/random_start).

## Suggested helper
Use multiproposal.utils.run_paths.build_run_dirs to build:
- estimations_dir = repo_root/estimations/<dataset>/<algorithm>/<run_name>
- reports_dir = repo_root/reports/<dataset>/<algorithm>/<run_name>

The run name includes a short hash so changes to any config produce a new folder.
