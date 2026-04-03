# Diagnostics

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
