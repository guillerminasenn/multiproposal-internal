# Plotting Policy

## General
- Use the spelling "mpCN" (not "mPCN") in titles and legends.
- Use LaTeX for symbols and indices: $\rho$, $x_1$, $x_2$.
- Prefer a single shared legend for multi-panel figures.
- Place legends outside the plotting area to avoid covering curves.

## Titles and Labels
- For per-parameter grids, use subplot titles as $x_1$ and $x_2$ only.
- Avoid redundant subplot titles when axis labels already indicate the metric.
- Keep figure-level titles short and consistent across notebooks.

## Trace Plots
- Use a fixed window of 30,000 iterations after burn-in for comparability.
- Use a horizontal shared legend above the subplot grid.
