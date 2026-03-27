# Run Output Policy

Goal: outputs are stored under run-specific folders that uniquely identify dataset generation and algorithm configuration. This enables cache reuse and prevents accidental reuse when hyperparameters change.

## Required practice
- Always build a run-specific directory using a deterministic hash of:
  - dataset name
  - data generation parameters (noise, priors, seeds, etc.)
  - algorithm configuration (method, P, rho grid, M, n_iters, burn_in, etc.)
  - sweep configuration (if applicable)
- Never write outputs under the notebooks folder. All outputs must live under the repo root
  in estimations/ and reports/ (or a user-specified root via MULTIPROPOSAL_RUN_ROOT).
- Store samples, metrics, diagnostics, and derived tables under the run-specific estimations directory.
- Store figures under the matching run-specific reports directory.
- Write the full configuration to config.json in the estimations directory.
- Reuse cached outputs when present: load existing samples/metrics from the run directory and skip re-running chains.

## Directory structure
Use a two-level identifier split:
- data_id: hash of a stable, minimal dataset-generation config (data hyperparameters,
  observation config, and data seeds). This should be identical across notebooks
  that use the same data, even if their sweep dimensions differ.
- run_id: hash of the algorithm/MCMC configuration (including sweep settings when applicable).

Required layout:
- estimations/<dataset>/<data_id>/<sweep|fixed>/<run_id>/...
- reports/<dataset>/<data_id>/<sweep|fixed>/<run_id>/...

## Figures
- Apply the publication style from reports/figure_style.py for all plots.

## Naming conventions
- Save chain outputs as <method>_P{P}_rho{rho}_seed{seed}.npz (or <method>_rho{rho}_seed{seed}.npz for pCN).
- Save metrics next to the chain file as <method>_P{P}_rho{rho}_seed{seed}_metrics.json.
- Store algorithm-specific diagnostics (e.g., mPCN diagnostics) alongside the chain files.

## Variants and sub-experiments
- For secondary experiments that share the same base run configuration (e.g., random-start sweeps), save under a subfolder of the main run directory (for example, estimations_dir/random_start).

## Suggested helper
When building paths, resolve repo_root from MULTIPROPOSAL_RUN_ROOT or by walking up to
the repo's pyproject.toml. Do not assume Path.cwd() points at the repo root when
executing notebooks.
