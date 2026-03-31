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

## Job scripts
- Store long-running sweep scripts under jobs/<dataset>/<run_id>_data_<data_id>/run.py.
- Keep notebooks focused on loading cached outputs and plotting, not running chains.
- Each job script should mirror the run_id/data_id used for estimations and reports.

## Grid sharding (P, rho sweeps)
- Prefer launching multiple job scripts with disjoint (P, rho) subsets over nested process pools.
- Use deterministic grid slicing (for example, grid_index/grid_count) so reruns are reproducible.
- Keep inner mPCN parallelization enabled; avoid additional per-grid process pools to prevent
  over-subscription unless explicitly managed.

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

## Parallelization policy
- Parallelization is an execution detail, not a statistical configuration. Do not include it in the run_id hash unless it changes the algorithmic output.
- Record parallel settings in config.json under a separate execution section when helpful (for example, n_jobs, backend, per-proposal parallelization).
- Prefer deterministic ordering of worker outputs (executor.map preserves input order) so results are stable across runs.
- Use threads when the problem object is not picklable. Use processes only if the problem and inputs can be serialized.

## Parallelization implementation pattern
- Keep the vectorized path as the default and add an opt-in switch (for example, parallelize_props) to enable proposal-level parallelism.
- Use SeedSequence to spawn per-proposal RNGs and avoid sharing RNG state across workers.
- Structure worker tasks to return both proposal and log-likelihood so the caller can assemble candidates and weights without extra passes.
- Example pattern:
  - Main thread: draw a master seed, spawn n_props SeedSequence objects.
  - Worker: sample z, build proposal, evaluate log-likelihood.
  - Main thread: collect props + log_likelihoods, then proceed with weighting and selection.

## Current recommended parallel setup (mPCN)
- Default to process-based log-likelihood evaluation with chunking.
- Keep proposal generation in the main process (vectorized).
- Use chunks sized at roughly ceil(P / n_jobs), or override explicitly.
- Disable proposal-level parallelization when using chunked log-likelihoods.
- Record the execution settings (backend, n_jobs, llh_chunk_size) in config.json and, where helpful, in chain metadata.

## Suggested helper
When building paths, resolve repo_root from MULTIPROPOSAL_RUN_ROOT or by walking up to
the repo's pyproject.toml. Do not assume Path.cwd() points at the repo root when
executing notebooks.
