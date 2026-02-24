"""
Calculate autocorrelation values for the chain
Apply Geyer's monotone sequence criterion to determine where to truncate
Calculate the integrated autocorrelation time
Estimate ESS as N/tau (chain length divided by autocorrelation time)
"""

import numpy as np
import matplotlib.pyplot as plt

def estimate_effective_sample_size(chain, max_lag=None, tol=1e-15):
    """
    Estimate the effective sample size (ESS) of an MCMC chain using ACF method.
    
    Parameters:
    -----------
    chain : array-like
        1D or 2D array of MCMC samples. If 2D, rows are samples and columns are parameters.
    max_lag : int, optional
        Maximum lag to consider for autocorrelation. If None, will use min(N/5, 1000)
        where N is the chain length.
        
    Returns:
    --------
    ess : float or np.ndarray
        Effective sample size. If chain is 1D, returns a float.
        If chain is 2D, returns an array with the ESS for each parameter.
    """
    # Ensure chain is a numpy array
    chain = np.asarray(chain)
    
    # Handle multi-dimensional chains
    if len(chain.shape) > 1:
        if len(chain.shape) > 2:
            raise ValueError("Chain must be 1D or 2D")
        
        # For 2D chains, compute ESS for each parameter
        print(f'Estimating ESS for each parameter.')
        return np.array([estimate_effective_sample_size(chain[:, i], max_lag) 
                         for i in range(chain.shape[1])])
    
    # For 1D chains
    N = len(chain)
    
    # Set default max_lag if not provided
    if max_lag is None:
        max_lag = min(int(N/5), 1000)
    # print(f'Chain length = {N} and max_lag = {max_lag}.')

    # Center the chain (subtract mean)
    centered_chain = chain - np.mean(chain)

    # Compute variance of centered chain
    variance_centered = np.var(centered_chain)
    if variance_centered == 0:
        print('Warning: Centered chain variance is zero. Returning ESS = 0.')
        return 0
    
    # Compute autocorrelation and integrated autocorrelation time
    acf_values = compute_autocorrelation(centered_chain, max_lag)
    tau = integrated_autocorrelation_time(acf_values)
    # print(f'Estimated autocorrelation time (tau) = {tau:.2f}.')
    # if np.allclose(acf_values[1:], 1.0):
    #     print('Warning: All autocorrelation values are one. Returning ESS = 0.')
    #     return 0
    
    # ESS = N / tau
    ess = N / tau
    
    return ess

def compute_autocorrelation(x, max_lag, tol=1e-15):
    """
    Compute autocorrelation function up to max_lag.
    
    Parameters:
    -----------
    x : array-like
        Centered time series
    max_lag : int
        Maximum lag to compute
    
    Returns:
    --------
    acf : array
        Autocorrelation function values from lag 0 to max_lag
    """
    N = len(x)
    variance = np.var(x)
    # print(f'Variance of the chain = {variance}')
    
    if variance < tol:
        # print(f"Chain has variance 0. Autocor is 1 for all lags.")
        return np.ones(max_lag + 1)
    
    acf = np.zeros(max_lag + 1)
    
    # Lag 0 is always 1
    acf[0] = 1.0
    # print(f'lag=0, acf[lag]=1')
    
    # Compute autocorrelation for lags 1 to max_lag
    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(x[lag:] * x[:-lag]) / ((N - lag) * variance)
        # print(f'lag={lag}, acf[lag]={acf[lag]}')
    # print(f"Computed autocorrelation values up to lag {max_lag}, with results:{acf}")
    return acf

def integrated_autocorrelation_time(acf):
    """
    Compute integrated autocorrelation time using Geyer's monotone sequence.
    
    Parameters:
    -----------
    acf : array-like
        Autocorrelation function values
    
    Returns:
    --------
    tau : float
        Integrated autocorrelation time
    """
    # Test if autocorrelation=1 for all lags
    if np.sum(acf) == len(acf):
        return np.inf
    
    # Apply Geyer's initial monotone sequence criterion
    max_lag = len(acf) - 1

    # Compute the sum of consecutive pairs
    sums = np.zeros(max_lag // 2)
    for i in range(max_lag // 2):
        sums[i] = acf[2 * i + 1] + acf[2 * i + 2]

    if len(sums) == 0:
        return 1.0

    # Keep only the initial positive sequence
    first_nonpos = np.where(sums <= 0)[0]
    cutoff = int(first_nonpos[0]) if first_nonpos.size > 0 else len(sums)
    sums = sums[:cutoff]
    if len(sums) == 0:
        return 1.0

    # # Enforce monotone non-increasing sequence
    # sums = np.minimum.accumulate(sums)

    # Integrated autocorrelation time from monotone pair sums
    tau = 1.0 + 2.0 * np.sum(sums)
    # print(f'The integrated autocorrelation time is {tau}.')
    
    return max(1.0, tau)  # Ensure tau is at least 1

def compute_ess_efficiency(chains, runtime_minutes, max_lag):
    """
    Compute ESS efficiency (ESS per minute) for multiple MCMC chains or parameters.
    
    Parameters:
    -----------
    chains : array-like
        2D array where each column is a chain for a different parameter
        or a list of 1D arrays where each array is a chain for a parameter
    runtime_minutes : float
        Runtime of the sampler in minutes
        
    Returns:
    --------
    ess_values : array
        Array of ESS values for each parameter
    ess_per_minute_values : array
        Array of ESS per minute for each parameter
    """
    # Convert to numpy array if it's not already
    chains = np.asarray(chains)
    
    # Handle case where input is a 1D array (single parameter)
    if len(chains.shape) == 1:
        chains = chains.reshape(-1, 1)
    
    # Number of parameters
    n_params = chains.shape[1]
    
    # Compute ESS for each parameter
    ess_values = np.zeros(n_params)
    for i in range(n_params):
        ess_values[i] = estimate_effective_sample_size(chains[:, i], max_lag=max_lag)
    
    # Compute ESS per minute
    ess_per_minute_values = ess_values / runtime_minutes
    
    return ess_values, ess_per_minute_values

def plot_ess_diagnostics(chain, max_lag=None, figsize=(12, 8)):
    """
    Plot diagnostics for ESS estimation.
    
    Parameters:
    -----------
    chain : array-like
        1D array of MCMC samples
    max_lag : int, optional
        Maximum lag to plot for autocorrelation
    figsize : tuple, optional
        Figure size
    """
    chain = np.asarray(chain)
    if len(chain.shape) > 1:
        raise ValueError("This plotting function only works for 1D chains")
    
    N = len(chain)
    
    if max_lag is None:
        max_lag = min(int(N/5), 1000)
    
    # Center the chain
    centered_chain = chain - np.mean(chain)
    
    # Compute autocorrelation
    acf_values = compute_autocorrelation(centered_chain, max_lag)
    
    # Calculate ESS
    ess = estimate_effective_sample_size(chain, max_lag=max_lag)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Trace plot
    axes[0, 0].plot(chain)
    axes[0, 0].set_title('Trace Plot')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Value')
    
    # Plot 2: Autocorrelation function
    axes[0, 1].plot(acf_values)
    axes[0, 1].set_title('Autocorrelation Function')
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('Autocorrelation')
    axes[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Plot 3: Histogram
    axes[1, 0].hist(chain, bins=30, alpha=0.7, density=True)
    axes[1, 0].set_title('Histogram')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    
    # Plot 4: ESS information
    axes[1, 1].axis('off')
    text_info = (
        f"Chain Length: {N}\n\n"
        f"ESS (ACF method): {ess:.2f} ({ess/N:.2%} of chain)\n"
        f"Autocorrelation Time: {N/ess:.2f}"
    )
    axes[1, 1].text(0.1, 0.5, text_info, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

def plot_ess_histograms(ess_values, ess_per_minute_values, figsize=(12, 6), 
                       parameter_names=None, bins=10):
    """
    Plot histograms of ESS and ESS per minute for multiple parameters.
    
    Parameters:
    -----------
    ess_values : array-like
        Array of ESS values for each parameter
    ess_per_minute_values : array-like
        Array of ESS per minute values for each parameter
    figsize : tuple, optional
        Figure size
    parameter_names : list, optional
        List of parameter names for the plot
    bins : int, optional
        Number of bins for histograms
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ESS histogram
    axes[0].hist(ess_values, bins=bins, alpha=0.7, color='blue')
    axes[0].set_title('Effective Sample Size (ESS)')
    axes[0].set_xlabel('ESS')
    axes[0].set_ylabel('Frequency')
    
    # Add median and mean lines
    median_ess = np.median(ess_values)
    mean_ess = np.mean(ess_values)
    axes[0].axvline(median_ess, color='red', linestyle='--', label=f'Median: {median_ess:.2f}')
    axes[0].axvline(mean_ess, color='green', linestyle=':', label=f'Mean: {mean_ess:.2f}')
    axes[0].legend()
    
    # Plot ESS per minute histogram
    axes[1].hist(ess_per_minute_values, bins=bins, alpha=0.7, color='purple')
    axes[1].set_title('ESS per Minute')
    axes[1].set_xlabel('ESS / Minute')
    axes[1].set_ylabel('Frequency')
    
    # Add median and mean lines
    median_eff = np.median(ess_per_minute_values)
    mean_eff = np.mean(ess_per_minute_values)
    axes[1].axvline(median_eff, color='red', linestyle='--', label=f'Median: {median_eff:.2f}')
    axes[1].axvline(mean_eff, color='green', linestyle=':', label=f'Mean: {mean_eff:.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    
    # If parameter names are provided, create a table with values
    if parameter_names is not None:
        # Create a table with parameter names, ESS, and ESS/min
        data = []
        
        # Create header
        header = ['Parameter', 'ESS', 'ESS/min']
        
        # Fill the data rows
        for i, name in enumerate(parameter_names):
            data.append([name, f"{ess_values[i]:.2f}", f"{ess_per_minute_values[i]:.2f}"])
        
        # Summary statistics in the last row
        data.append(['Mean', f"{np.mean(ess_values):.2f}", f"{np.mean(ess_per_minute_values):.2f}"])
        data.append(['Median', f"{np.median(ess_values):.2f}", f"{np.median(ess_per_minute_values):.2f}"])
        data.append(['Min', f"{np.min(ess_values):.2f}", f"{np.min(ess_per_minute_values):.2f}"])
        data.append(['Max', f"{np.max(ess_values):.2f}", f"{np.max(ess_per_minute_values):.2f}"])
        
        # Create a new figure for the table
        plt.figure(figsize=(8, len(data)*0.5))
        table = plt.table(cellText=data, colLabels=header, loc='center',
                         cellLoc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        plt.axis('off')
        plt.title('ESS and Efficiency by Parameter', y=0.9 + 0.05*len(data))
        plt.tight_layout()
        
    plt.show()
    
    # Return summary statistics
    summary = {
        'mean_ess': mean_ess,
        'median_ess': median_ess,
        'mean_ess_per_minute': mean_eff,
        'median_ess_per_minute': median_eff,
        'min_ess': np.min(ess_values),
        'max_ess': np.max(ess_values),
        'min_ess_per_minute': np.min(ess_per_minute_values),
        'max_ess_per_minute': np.max(ess_per_minute_values)
    }
    
    return summary

def compute_squared_jumping_distance(chain):
    """
    Compute the squared jumping distance for an MCMC chain.
    
    The squared jumping distance is the squared Euclidean distance between
    consecutive samples in the chain.
    
    Parameters:
    -----------
    chain : array-like
        1D or 2D array of MCMC samples. If 2D, rows are samples and columns are parameters.
        
    Returns:
    --------
    sjd : array
        Squared jumping distances. If chain is 1D, returns a 1D array of length N-1.
        If chain is 2D, returns a 1D array of length N-1 with the squared distance
        computed across all parameters.
    """
    chain = np.asarray(chain)
    
    if len(chain.shape) == 1:
        # For 1D chains, squared jumping distance is just squared differences
        sjd = np.diff(chain) ** 2
    elif len(chain.shape) == 2:
        # For 2D chains, compute Euclidean distance between consecutive samples
        diffs = np.diff(chain, axis=0)
        sjd = np.sum(diffs ** 2, axis=1)
    else:
        raise ValueError("Chain must be 1D or 2D")
    
    return sjd

def compute_mean_squared_jumping_distance(chain):
    """
    Compute the mean squared jumping distance (MSJD) for an MCMC chain.
    
    This is a summary statistic that measures the average squared distance
    between consecutive samples.
    
    Parameters:
    -----------
    chain : array-like
        1D or 2D array of MCMC samples.
        
    Returns:
    --------
    msjd : float or np.ndarray
        Mean squared jumping distance. If chain is 1D, returns a float.
        If chain is 2D, returns an array with the MSJD for each parameter
        (computed using only that parameter's variance).
    """
    chain = np.asarray(chain)
    
    if len(chain.shape) > 1 and len(chain.shape) != 2:
        raise ValueError("Chain must be 1D or 2D")
    
    # Handle multi-dimensional chains by computing MSJD for each parameter
    if len(chain.shape) == 2:
        return np.array([compute_mean_squared_jumping_distance(chain[:, i]) 
                         for i in range(chain.shape[1])])
    
    # For 1D chains
    sjd = compute_squared_jumping_distance(chain)
    msjd = np.mean(sjd)
    
    return msjd

def compute_normalized_jumping_distance(chain):
    """
    Compute normalized jumping distance (jumping distance divided by variance).
    
    This normalizes the jumping distance by the empirical variance of the chain,
    providing a scale-invariant measure of mixing.
    
    Parameters:
    -----------
    chain : array-like
        1D or 2D array of MCMC samples.
        
    Returns:
    --------
    njd : float or np.ndarray
        Normalized jumping distance. If chain is 1D, returns a float.
        If chain is 2D, returns an array with the NJD for each parameter.
    """
    chain = np.asarray(chain)
    
    if len(chain.shape) > 1 and len(chain.shape) != 2:
        raise ValueError("Chain must be 1D or 2D")
    
    # Handle multi-dimensional chains
    if len(chain.shape) == 2:
        return np.array([compute_normalized_jumping_distance(chain[:, i]) 
                         for i in range(chain.shape[1])])
    
    # For 1D chains
    msjd = compute_mean_squared_jumping_distance(chain)
    variance = np.var(chain)
    
    if variance == 0:
        return 0.0
    
    njd = msjd / variance
    
    return njd