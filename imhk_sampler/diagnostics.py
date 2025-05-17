import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Tuple, Any
from sage.structure.element import Vector
import os
from pathlib import Path

# Set the Seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def get_plot_dir():
    """Get the directory for saving plots with proper error handling."""
    from pathlib import Path
    
    plot_dir = Path(__file__).resolve().parent.parent / "results/plots"
    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create plot directory: {e}")
    return plot_dir


def _to_numpy(samples: List[Union[Vector, np.ndarray, List]]) -> np.ndarray:
    """
    Convert SageMath vectors or lists to a NumPy array.
    
    Args:
        samples: List of vectors to convert
        
    Returns:
        NumPy array of the samples
    """
    if not samples:
        raise ValueError("Empty samples list provided")
    
    # Create an empty array based on the sample shape
    sample = samples[0]
    if isinstance(sample, Vector):
        # SageMath vector
        n_dims = len(sample)
        result = np.zeros((len(samples), n_dims))
        for i, s in enumerate(samples):
            for j in range(n_dims):
                result[i, j] = float(s[j])
    elif isinstance(sample, (list, tuple)):
        # List or tuple
        result = np.array(samples, dtype=float)
    elif isinstance(sample, np.ndarray):
        # Already a NumPy array, just stack them
        result = np.stack(samples) if len(samples) > 1 else np.array(samples[0], dtype=float)
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")
    
    return result


def compute_autocorrelation(samples: List[Union[Vector, np.ndarray, List]], 
                           lag: int = 50) -> List[np.ndarray]:
    """
    Compute the autocorrelation of a chain of samples up to a given lag.
    
    Mathematical Basis:
    Autocorrelation measures the correlation between observations as a function of the time lag 
    between them. For a stationary process, the autocorrelation ρ at lag k is defined as:
    ρ(k) = Cov(X_t, X_{t+k}) / Var(X_t)
    
    Relevance to Lattice-based Cryptography:
    In MCMC sampling for lattice-based cryptography, autocorrelation analysis helps assess the 
    quality of the sampler. High autocorrelation indicates that consecutive samples are highly 
    dependent, potentially leading to biased estimates for cryptographic parameters.
    
    Args:
        samples: List of sample vectors (SageMath vectors, NumPy arrays, or lists)
        lag: Maximum lag to compute autocorrelation for (must be positive)
        
    Returns:
        A list of autocorrelation arrays, one for each dimension
        
    Raises:
        ValueError: If samples is empty or lag is not positive
        TypeError: If samples elements are not of a supported type
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    if lag <= 0:
        raise ValueError("Lag must be positive")
    
    # Convert to NumPy array
    samples_np = _to_numpy(samples)
    n_samples, n_dims = samples_np.shape
    
    # Ensure lag doesn't exceed sample size
    lag = min(lag, n_samples - 1)
    
    # Initialize results
    acf_by_dim = []
    
    # Vectorized autocorrelation computation for each dimension
    for dim in range(n_dims):
        # Extract this dimension's values
        dim_values = samples_np[:, dim]
        
        # Center the data
        centered_data = dim_values - np.mean(dim_values)
        
        # Compute variance (denominator for normalization)
        variance = np.var(dim_values, ddof=1)
        if variance < 1e-10:
            # Handle near-constant series
            acf = np.zeros(lag + 1)
            acf[0] = 1.0  # Autocorrelation at lag 0 is always 1
            acf_by_dim.append(acf)
            continue
        
        # Initialize autocorrelation array
        acf = np.zeros(lag + 1)
        acf[0] = 1.0  # Autocorrelation at lag 0 is always 1
        
        # Vectorized computation for lags 1 to lag
        for k in range(1, lag + 1):
            # Use dot product for efficiency
            acf[k] = np.sum(centered_data[:-k] * centered_data[k:]) / ((n_samples - k) * variance)
        
        acf_by_dim.append(acf)
    
    return acf_by_dim


def compute_initial_monotone_sequence(acf: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Compute the initial monotone sequence estimator for determining the
    cutoff lag in autocorrelation calculations.
    
    This implements Geyer's (1992) initial positive sequence and initial monotone 
    sequence estimators for MCMC standard errors.
    
    Args:
        acf: Array of autocorrelation values
        
    Returns:
        Tuple containing:
        - Cutoff lag determined by the initial monotone sequence
        - Modified autocorrelation values
    """
    # Skip lag 0, which is always 1
    acf_pairs = acf[1:]
    
    # Step 1: Create initial positive sequence
    # Group by pairs for even-odd correction (Γₙ = ρ₂ₙ₋₁ + ρ₂ₙ)
    n_pairs = len(acf_pairs) // 2
    gamma = np.zeros(n_pairs)
    for i in range(n_pairs):
        # Sum consecutive autocorrelations (for even-odd correction)
        if 2*i+1 < len(acf_pairs):
            gamma[i] = acf_pairs[2*i] + acf_pairs[2*i+1]
        else:
            gamma[i] = acf_pairs[2*i]
    
    # Step 2: Create initial positive sequence
    # Find first negative or near-zero value
    cutoff = n_pairs
    for i in range(n_pairs):
        if gamma[i] < 0 or abs(gamma[i]) < 1e-10:
            cutoff = i
            break
    
    # Truncate the sequence
    gamma = gamma[:cutoff]
    
    # Step 3: Create initial monotone sequence
    # Ensure the sequence is monotonically decreasing
    for i in range(1, len(gamma)):
        if gamma[i] > gamma[i-1]:
            gamma[i] = gamma[i-1]
    
    # The cutoff lag in the original autocorrelation terms
    effective_cutoff = min(2 * len(gamma), len(acf) - 1)
    
    # Return the cutoff lag and the modified gamma values
    return effective_cutoff, gamma


def compute_ess(samples: List[Union[Vector, np.ndarray, List]]) -> List[float]:
    """
    Compute the Effective Sample Size (ESS) for each dimension of the samples.
    
    Mathematical Basis:
    ESS estimates the equivalent number of independent samples represented by an autocorrelated
    chain. It's calculated as:
    ESS = n / (1 + 2*∑ρ(k))
    where n is the total sample size, and ∑ρ(k) is the sum of autocorrelations up to an appropriate lag.
    
    Relevance to Lattice-based Cryptography:
    In lattice-based cryptography, ensuring high ESS is crucial for:
    - Reliable parameter estimation of lattice-based schemes
    - Accurate security estimates for cryptographic primitives
    - Valid evaluation of sampling algorithms used in cryptographic protocols
    
    Args:
        samples: List of sample vectors (SageMath vectors, NumPy arrays, or lists)
        
    Returns:
        A list of ESS values, one for each dimension
        
    Raises:
        ValueError: If samples is empty
        TypeError: If samples elements are not of a supported type
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    
    # Convert to NumPy array
    samples_np = _to_numpy(samples)
    n_samples, n_dims = samples_np.shape
    
    # Determine max lag (recommended: min(n/5, 50))
    max_lag = min(50, n_samples // 5)
    
    # Compute autocorrelation for all dimensions
    acf_by_dim = compute_autocorrelation(samples, max_lag)
    
    # Compute ESS for each dimension
    ess_values = []
    
    for dim in range(n_dims):
        acf = acf_by_dim[dim]
        
        # Use initial monotone sequence estimator to determine cutoff lag
        cutoff_lag, gamma = compute_initial_monotone_sequence(acf)
        
        # Calculate ESS using the appropriate autocorrelation sum
        # ESS = n / (1 + 2*∑ρ(k))
        if len(gamma) > 0:
            # Sum of modified gamma values already accounts for pairs
            rho_sum = np.sum(gamma)
            ess = n_samples / (1 + 2 * rho_sum)
        else:
            # No significant autocorrelation detected
            ess = n_samples
        
        ess_values.append(ess)
    
    return ess_values


def plot_trace(samples: List[Union[Vector, np.ndarray, List]], 
              filename: str, 
              title: Optional[str] = None) -> None:
    """
    Create trace plots for each dimension of the samples.
    
    Trace plots visualize how the Markov chain explores the parameter space over time.
    Ideal trace plots should show good mixing (moving freely around parameter space)
    and stationarity (constant mean and variance over time).
    
    Relevance to Lattice-based Cryptography:
    In lattice-based cryptography, trace plots help assess:
    - Whether the sampler is correctly exploring the lattice space
    - If the chain has reached stationarity (crucial for security parameter estimation)
    - Quality of the sampling procedure for generating cryptographic keys or signatures
    
    Args:
        samples: List of sample vectors (SageMath vectors, NumPy arrays, or lists)
        filename: Filename to save the plot (without path)
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty
        TypeError: If samples elements are not of a supported type
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    
    # Convert to NumPy array
    samples_np = _to_numpy(samples)
    n_samples, n_dims = samples_np.shape
    
    # Get the plot directory
    plot_dir = get_plot_dir()
    
    # Create the figure
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    # Set the color palette
    colors = sns.color_palette("muted")
    
    for dim in range(n_dims):
        # Extract this dimension's values
        dim_values = samples_np[:, dim]
        
        # Calculate statistics
        mean_val = np.mean(dim_values)
        std_val = np.std(dim_values)
        min_val = np.min(dim_values)
        max_val = np.max(dim_values)
        quantile_25 = np.percentile(dim_values, 25)
        quantile_75 = np.percentile(dim_values, 75)
        
        # Plot trace
        axes[dim].plot(dim_values, color=colors[0], linewidth=1.0)
        axes[dim].set_ylabel(f'Dimension {dim+1}')
        
        # Add horizontal lines for mean and quantiles
        axes[dim].axhline(mean_val, color=colors[1], linestyle='--', 
                         alpha=0.7, label='Mean')
        axes[dim].axhline(quantile_25, color=colors[2], linestyle=':', 
                         alpha=0.5, label='25% Quantile')
        axes[dim].axhline(quantile_75, color=colors[2], linestyle=':', 
                         alpha=0.5, label='75% Quantile')
        
        # Annotate with statistics
        axes[dim].text(0.02, 0.95, 
                      f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\n'
                      f'Min: {min_val:.2f}\nMax: {max_val:.2f}\n'
                      f'ESS: {compute_ess([samples[i] for i in range(n_samples)])[dim]:.1f}', 
                      transform=axes[dim].transAxes, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        if dim == 0:  # Only add legend to first plot to avoid repetition
            axes[dim].legend(loc='upper right')
    
    # Add overall labels and title
    axes[-1].set_xlabel('Iteration')
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Finalize and save
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Adjust for the title
    
    try:
        plt.savefig(plot_dir / filename, dpi=300)
        print(f"Plot saved to {plot_dir / filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()


def plot_autocorrelation(acf_by_dim: List[np.ndarray], 
                        filename: str, 
                        title: Optional[str] = None,
                        ess_values: Optional[List[float]] = None) -> None:
    """
    Plot the autocorrelation function for each dimension.
    
    Mathematical Basis:
    Autocorrelation plots visualize the correlation between samples separated by different lags.
    Rapid decay towards zero indicates better mixing of the Markov chain.
    
    Relevance to Lattice-based Cryptography:
    In lattice-based cryptography, autocorrelation plots help:
    - Assess independence of samples used in security parameter estimation
    - Evaluate the efficiency of sampling algorithms
    - Determine if the algorithm needs longer burn-in periods or thinning
    
    Args:
        acf_by_dim: List of autocorrelation functions for each dimension
        filename: Filename to save the plot (without path)
        title: Optional title for the plot
        ess_values: Optional list of effective sample sizes to include in the plot
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If acf_by_dim is empty
    """
    # Input validation
    if not acf_by_dim:
        raise ValueError("Empty autocorrelation list provided")
    
    n_dims = len(acf_by_dim)
    
    # Get the plot directory
    plot_dir = get_plot_dir()
    
    # Create the figure
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    # Set the color palette
    colors = sns.color_palette("muted")
    
    for dim in range(n_dims):
        # Plot autocorrelation
        axes[dim].plot(acf_by_dim[dim], color=colors[0], marker='o', markersize=3, alpha=0.7)
        axes[dim].set_ylabel(f'ACF Dim {dim+1}')
        
        # Add horizontal lines
        axes[dim].axhline(0, color=colors[3], linestyle='-', alpha=0.3)
        axes[dim].axhline(0.05, color=colors[2], linestyle=':', alpha=0.5, 
                         label='Significance (±0.05)')
        axes[dim].axhline(-0.05, color=colors[2], linestyle=':', alpha=0.5)
        
        # Determine cutoff lag using initial monotone sequence estimator
        cutoff_lag, _ = compute_initial_monotone_sequence(acf_by_dim[dim])
        
        # Mark the cutoff point
        if cutoff_lag < len(acf_by_dim[dim]):
            axes[dim].axvline(cutoff_lag, color=colors[1], linestyle='--', alpha=0.5, 
                             label=f'Cutoff lag: {cutoff_lag}')
        
        # Add annotation with ESS calculation info
        if ess_values is not None:
            ess = ess_values[dim]
        else:
            # Compute ESS using the autocorrelation function
            n_samples = len(acf_by_dim[dim])
            _, gamma = compute_initial_monotone_sequence(acf_by_dim[dim])
            rho_sum = np.sum(gamma) if len(gamma) > 0 else 0
            ess = n_samples / (1 + 2 * rho_sum)
        
        axes[dim].text(0.7, 0.95, 
                      f'ESS: {ess:.1f}\nCutoff lag: {cutoff_lag}', 
                      transform=axes[dim].transAxes, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        if dim == 0:
            axes[dim].legend(loc='upper right')
    
    # Add overall labels and title
    axes[-1].set_xlabel('Lag')
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Finalize and save
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Adjust for the title
    
    try:
        plt.savefig(plot_dir / filename, dpi=300)
        print(f"Plot saved to {plot_dir / filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()


def plot_acceptance_trace(accepts: List[bool], 
                         filename: str, 
                         window_size: int = 100,
                         title: Optional[str] = "IMHK Acceptance Rate Over Time") -> None:
    """
    Plot the acceptance rate over time using a moving window.
    
    Mathematical Basis:
    The acceptance rate in Metropolis-Hastings algorithms measures the proportion of
    proposed moves that are accepted. It's calculated over a moving window to show
    how the acceptance behavior changes throughout the simulation.
    
    Relevance to Lattice-based Cryptography:
    In lattice-based cryptography, acceptance rates help:
    - Tune proposal distributions for optimal sampling efficiency
    - Ensure cryptographic parameter estimates are based on well-mixed chains
    - Diagnose issues with sampling algorithms used in lattice-based schemes
    
    Args:
        accepts: List of booleans indicating acceptance
        filename: Filename to save the plot (without path)
        window_size: Size of the moving window (must be positive and smaller than the length of accepts)
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If accepts is empty or window_size is inappropriate
    """
    # Input validation
    if not accepts:
        raise ValueError("Empty accepts list provided")
    
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    
    if window_size >= len(accepts):
        window_size = max(len(accepts) // 10, 1)  # Default to 1/10 of the data
    
    # Convert to NumPy array
    accepts_arr = np.array(accepts, dtype=float)
    
    # Get the plot directory
    plot_dir = get_plot_dir()
    
    # Calculate moving average
    cumsum = np.cumsum(np.insert(accepts_arr, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Set the color palette
    colors = sns.color_palette("muted")
    
    # Plot moving average
    x = np.arange(len(moving_avg)) + window_size // 2
    plt.plot(x, moving_avg, color=colors[0], 
             label=f'Moving Avg (window={window_size})')
    
    # Add horizontal line for overall acceptance rate
    overall_rate = np.mean(accepts_arr)
    plt.axhline(overall_rate, color=colors[1], linestyle='--', alpha=0.8, 
               label=f'Overall Rate: {overall_rate:.3f}')
    
    # Add optimal acceptance rate range for reference (typically 0.23-0.5 for MH)
    plt.axhspan(0.23, 0.5, alpha=0.2, color='green', 
               label='Optimal Range (0.23-0.5)')
    
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Acceptance Rate')
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Finalize and save
    plt.tight_layout()
    try:
        plt.savefig(plot_dir / filename, dpi=300)
        print(f"Plot saved to {plot_dir / filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()