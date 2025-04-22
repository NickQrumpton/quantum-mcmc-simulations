import numpy as np
import matplotlib.pyplot as plt

def compute_autocorrelation(samples, lag=50):
    """
    Compute the autocorrelation of a chain of samples up to a given lag.
    
    Args:
        samples: List of samples
        lag: Maximum lag to compute autocorrelation for
        
    Returns:
        A list of autocorrelation values for each dimension
    """
    n_dims = len(samples[0])
    acf_by_dim = []
    
    for dim in range(n_dims):
        # Extract this dimension's values
        dim_values = [float(sample[dim]) for sample in samples]
        dim_values = np.array(dim_values)
        
        # Compute autocorrelation
        acf = []
        mean = np.mean(dim_values)
        var = np.var(dim_values)
        
        for l in range(lag + 1):
            if l == 0:
                acf.append(1.0)
            else:
                # Compute autocorrelation at lag l
                sum_corr = 0
                for t in range(len(dim_values) - l):
                    sum_corr += (dim_values[t] - mean) * (dim_values[t + l] - mean)
                
                acf.append(sum_corr / ((len(dim_values) - l) * var))
        
        acf_by_dim.append(acf)
    
    return acf_by_dim

def compute_ess(samples):
    """
    Compute the Effective Sample Size (ESS) for each dimension of the samples.
    
    Args:
        samples: List of samples
        
    Returns:
        A list of ESS values, one for each dimension
    """
    n_dims = len(samples[0])
    ess_values = []
    
    for dim in range(n_dims):
        # Extract this dimension's values
        dim_values = [float(sample[dim]) for sample in samples]
        dim_values = np.array(dim_values)
        
        # Compute autocorrelation for lags 1 to 50 (or less if the series is shorter)
        max_lag = min(50, len(dim_values) // 4)
        acf = compute_autocorrelation([samples[i] for i in range(len(samples))], max_lag)[dim]
        
        # Find the lag where ACF cuts off (absolute value < 0.05 or becomes negative)
        cutoff_lag = max_lag
        for i in range(1, max_lag + 1):
            if abs(acf[i]) < 0.05 or acf[i] < 0:
                cutoff_lag = i
                break
        
        # Compute ESS using the truncated sum of autocorrelations
        rho_sum = 2 * sum(acf[1:cutoff_lag])
        ess = len(dim_values) / (1 + rho_sum)
        ess_values.append(ess)
    
    return ess_values

def plot_trace(samples, filename, title=None):
    """
    Create trace plots for each dimension of the samples.
    
    Args:
        samples: List of samples
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(samples[0])
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    for dim in range(n_dims):
        # Extract this dimension's values
        dim_values = [float(sample[dim]) for sample in samples]
        
        # Plot trace
        axes[dim].plot(dim_values)
        axes[dim].set_ylabel(f'Dimension {dim+1}')
        
        # Add horizontal lines for mean and quantiles
        mean_val = np.mean(dim_values)
        quantile_25 = np.percentile(dim_values, 25)
        quantile_75 = np.percentile(dim_values, 75)
        
        axes[dim].axhline(mean_val, color='r', linestyle='--', alpha=0.5, label='Mean')
        axes[dim].axhline(quantile_25, color='g', linestyle=':', alpha=0.3, label='25% Quantile')
        axes[dim].axhline(quantile_75, color='g', linestyle=':', alpha=0.3, label='75% Quantile')
        
        # Annotate with statistics
        axes[dim].text(0.02, 0.95, 
                       f'Mean: {mean_val:.2f}\nStd: {np.std(dim_values):.2f}\n'
                       f'Min: {min(dim_values):.2f}\nMax: {max(dim_values):.2f}', 
                       transform=axes[dim].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if dim == 0:  # Only add legend to first plot to avoid repetition
            axes[dim].legend(loc='upper right')
    
    axes[-1].set_xlabel('Iteration')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Adjust for the title
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_autocorrelation(acf_by_dim, filename, title=None):
    """
    Plot the autocorrelation function for each dimension.
    
    Args:
        acf_by_dim: List of autocorrelation functions for each dimension
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(acf_by_dim)
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    for dim in range(n_dims):
        axes[dim].plot(acf_by_dim[dim])
        axes[dim].set_ylabel(f'ACF Dim {dim+1}')
        
        # Add horizontal lines
        axes[dim].axhline(0, color='r', linestyle='--', alpha=0.3)
        axes[dim].axhline(0.05, color='g', linestyle=':', alpha=0.3)
        axes[dim].axhline(-0.05, color='g', linestyle=':', alpha=0.3)
        
        # Calculate effective sample size
        max_lag = len(acf_by_dim[dim]) - 1
        cutoff_lag = max_lag
        for i in range(1, max_lag + 1):
            if abs(acf_by_dim[dim][i]) < 0.05 or acf_by_dim[dim][i] < 0:
                cutoff_lag = i
                break
        
        # Add annotation with ESS calculation info
        corr_sum = 2 * sum(acf_by_dim[dim][1:cutoff_lag])
        ess = len(acf_by_dim[dim]) / (1 + corr_sum)
        
        axes[dim].text(0.7, 0.95, 
                       f'ESS: {ess:.1f}\nCutoff lag: {cutoff_lag}', 
                       transform=axes[dim].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Lag')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Adjust for the title
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_acceptance_trace(accepts, filename, window_size=100):
    """
    Plot the acceptance rate over time using a moving window.
    
    Args:
        accepts: List of booleans indicating acceptance
        filename: Filename to save the plot
        window_size: Size of the moving window
        
    Returns:
        None (saves plot to file)
    """
    accepts_arr = np.array(accepts, dtype=float)
    
    # Calculate moving average
    cumsum = np.cumsum(np.insert(accepts_arr, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(moving_avg)) + window_size // 2
    ax.plot(x, moving_avg, label=f'Moving Avg (window={window_size})')
    
    # Add horizontal line for overall acceptance rate
    overall_rate = np.mean(accepts_arr)
    ax.axhline(overall_rate, color='r', linestyle='--', alpha=0.8, 
              label=f'Overall Rate: {overall_rate:.3f}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('IMHK Acceptance Rate Over Time')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}')
    plt.close()