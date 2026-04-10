# Diagnostics

## RMSE/MAE/MSE diagnostics (solute transport)

For the solute transport convergence notebook RMSE section:

1. Choose component index i for the parameter vector and the max iteration counts for the bar and time-series diagnostics.
2. Compute per-chain MAE, MSE, and RMSE using the first n samples:
   - MAE = mean(abs(x_i - true_i))
   - MSE = mean((x_i - true_i)^2)
   - RMSE = sqrt(MSE)
3. For thinned pCN, thin each pCN chain by stride P, then limit the unthinned iterations so that unthinned length equals P times the thinned length used.
4. Bar plot: compute RMSE per chain (using the chosen iteration counts), then take the mean over chains for each algorithm (mPCN, pCN, pCN thinned).
5. Time-series plots: for k = 1..K, compute MAE/MSE/RMSE using the first k samples of each chain; average over chains within each algorithm to get the plotted series.

## Independent pCN ESS/MSJD computation

When plotting or tabulating independent pCN diagnostics for a given rho and P:

1. Load the P independent pCN chain files.
2. Thin each chain by taking every P-th sample (stride = P), then apply burn-in.
3. Compute ESS per chain on the thinned samples and sum across chains.
4. Compute MSJD per chain on the thinned samples, then aggregate:
   - mean of the per-chain MSJD means
   - max of the per-chain MSJD means

These aggregations are stored in the independent metrics payload:
- ess_mean_sum
- msjd_mean_mean
- msjd_mean_max
