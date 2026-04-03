# Plotting Policy

## General
- Use the spelling "mpCN" (not "mPCN") in titles and legends.
- Use LaTeX for symbols and indices: $\rho$, $x_1$, $x_2$.
- Prefer a single shared legend for multi-panel figures.
- Place legends outside the plotting area to avoid covering curves.
- For independent pCN aggregates, use a distinct linestyle (for example, dotted for mean and
	dashed for max) and label them explicitly as "pCN indep".

## ESS/MSJD Curves
- For independent pCN curves, plot ESS as a single dotted line per P (no markers).
- For MSJD, plot both mean and max for independent pCN (mean dotted, max dashed) using the
  same color per P to make the pair visually linked.
- Avoid markers on independent pCN curves so they remain distinguishable from mpCN and pCN.

## Titles and Labels
- For per-parameter grids, use subplot titles as $x_1$ and $x_2$ only.
- Avoid redundant subplot titles when axis labels already indicate the metric.
- Keep figure-level titles short and consistent across notebooks.

## Trace Plots
- Use a fixed window of 30,000 iterations after burn-in for comparability.
- Use a horizontal shared legend above the subplot grid.
