"""
IMHK Sampler: A Python package for high-quality discrete Gaussian sampling over lattices

This package implements the Independent Metropolis-Hastings-Klein (IMHK) algorithm
for sampling from discrete Gaussian distributions over lattices, along with tools
for visualization, diagnostics, and statistical analysis. The package is designed
for research in lattice-based cryptography, where high-quality discrete Gaussian
sampling is essential for security and efficiency.

Key components:
- Samplers: IMHK and Klein algorithms for lattice-based discrete Gaussian sampling
- Utils: Core mathematical functions for discrete Gaussian distributions
- Diagnostics: Tools for assessing sample quality and Markov chain properties
- Visualization: Functions for creating publication-quality visualizations
- Stats: Statistical measures for comparing empirical and theoretical distributions
- Experiments: Frameworks for systematic parameter studies and analysis

References:
[1] Klein, P. (2000). Finding the closest lattice vector when it's unusually close.
    In Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms (pp. 937-941).
[2] Ducas, L., & Prest, T. (2016). Fast Fourier orthogonalization. In Proceedings
    of the ACM on International Symposium on Symbolic and Algebraic Computation (pp. 191-198).
[3] Gentry, C., Peikert, C., & Vaikuntanathan, V. (2008). Trapdoors for hard lattices
    and new cryptographic constructions. In Proceedings of the 40th Annual ACM
    Symposium on Theory of Computing (pp. 197-206).
"""

__version__ = '0.1.0'
__author__ = 'Lattice Cryptography Research Group'

from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from sage.all import Matrix, Vector, RealNumber

# Expose key functions for easy imports
from .samplers import imhk_sampler, klein_sampler, discrete_gaussian_sampler_1d
from .utils import discrete_gaussian_pdf, precompute_discrete_gaussian_probabilities
from .diagnostics import compute_autocorrelation, compute_ess, plot_trace
from .diagnostics import plot_autocorrelation, plot_acceptance_trace
from .visualization import plot_2d_samples, plot_3d_samples, plot_2d_projections, plot_pca_projection
from .stats import compute_total_variation_distance, compute_kl_divergence
from .experiments import run_experiment, parameter_sweep, plot_parameter_sweep_results, compare_convergence_times

# Type annotations and docstrings for exposed functions

# ---- Sampling Functions ----

def imhk_sampler(
    B: Matrix, 
    sigma: float, 
    num_samples: int, 
    center: Optional[Vector] = None, 
    burn_in: int = 1000
) -> Tuple[List[Vector], float, List[Vector], List[bool]]:
    """
    Independent Metropolis-Hastings-Klein algorithm for sampling from a discrete Gaussian over a lattice.
    
    Mathematical Context:
    The IMHK algorithm combines Klein's algorithm as a proposal distribution with
    Metropolis-Hastings to correct the distribution, ensuring the exact discrete
    Gaussian distribution is sampled. The algorithm is particularly useful for
    high-quality sampling when the lattice basis is not orthogonal.
    
    Cryptographic Relevance:
    High-quality discrete Gaussian sampling is essential for:
    - Security of lattice-based signature schemes (e.g., FALCON)
    - Parameter selection in encryption schemes
    - Trapdoor sampling with statistical indistinguishability
    - Side-channel resistant implementations
    
    Args:
        B: The lattice basis matrix (rows are basis vectors)
        sigma: The standard deviation of the Gaussian (must be positive)
        num_samples: The number of samples to generate (must be positive)
        center: The center of the Gaussian (default: origin)
        burn_in: The number of initial samples to discard (default: 1000)
        
    Returns:
        A tuple containing:
        - List of lattice point samples
        - Acceptance rate of the Metropolis-Hastings algorithm
        - All samples including burn-in (for diagnostics)
        - Boolean indicators of whether each proposal was accepted
        
    Raises:
        ValueError: If sigma is not positive, num_samples is not positive,
                   or dimensions are incompatible
        RuntimeError: If sampling fails due to numerical issues
        
    Example:
        >>> B = matrix.identity(RR, 2)  # 2D identity lattice
        >>> samples, acceptance_rate, _, _ = imhk_sampler(B, 1.0, 1000)
        >>> len(samples)
        1000
        >>> 0 <= acceptance_rate <= 1
        True
    """
    return imhk_sampler(B, sigma, num_samples, center, burn_in)


def klein_sampler(
    B: Matrix, 
    sigma: float, 
    center: Optional[Vector] = None
) -> Vector:
    """
    Klein's algorithm for sampling from a discrete Gaussian over a lattice.
    
    Mathematical Context:
    Klein's algorithm samples from an approximate discrete Gaussian by using
    Gram-Schmidt orthogonalization and recursive sampling of coordinates.
    It works by sampling one coordinate at a time, working backwards from
    the last coordinate to the first.
    
    Cryptographic Relevance:
    Klein's algorithm is used in lattice-based cryptography for:
    - Trapdoor sampling in signature schemes
    - Basis delegation techniques
    - Approximate sampling when exact sampling is too costly
    - Preprocessing steps in certain cryptographic constructions
    
    Args:
        B: The lattice basis matrix (rows are basis vectors)
        sigma: The standard deviation of the Gaussian (must be positive)
        center: The center of the Gaussian (default: origin)
        
    Returns:
        A lattice point sampled according to Klein's algorithm
        
    Raises:
        ValueError: If sigma is not positive or dimensions are incompatible
        RuntimeError: If Gram-Schmidt orthogonalization fails
        
    Example:
        >>> B = matrix.identity(RR, 2)  # 2D identity lattice
        >>> sample = klein_sampler(B, 1.0)
        >>> len(sample)
        2
    """
    return klein_sampler(B, sigma, center)


def discrete_gaussian_sampler_1d(
    center: float, 
    sigma: float
) -> int:
    """
    Sample from a 1D discrete Gaussian distribution.
    
    Mathematical Context:
    The discrete Gaussian distribution over integers is a probability distribution
    where the probability of sampling integer x is proportional to exp(-(x-center)²/(2σ²)).
    This function implements efficient sampling from this distribution.
    
    Cryptographic Relevance:
    1D discrete Gaussian sampling is a building block for:
    - Klein's algorithm and other lattice samplers
    - Noise generation in Ring-LWE encryption
    - Error terms in lattice-based cryptographic schemes
    - Secure implementation of lattice-based primitives
    
    Args:
        center: The center of the Gaussian
        sigma: The standard deviation of the Gaussian (must be positive)
        
    Returns:
        An integer sampled from the discrete Gaussian
        
    Raises:
        ValueError: If sigma is not positive
        RuntimeError: If sampling fails due to numerical underflow
        
    Example:
        >>> # Sample from a standard discrete Gaussian
        >>> sample = discrete_gaussian_sampler_1d(0.0, 1.0)
        >>> isinstance(sample, int)
        True
    """
    return discrete_gaussian_sampler_1d(center, sigma)


# ---- Utility Functions ----

def discrete_gaussian_pdf(
    x: Union[int, float, Vector], 
    sigma: float, 
    center: Optional[Union[float, Vector]] = None
) -> float:
    """
    Compute the probability density function of a discrete Gaussian.
    
    Mathematical Context:
    For a point x, the unnormalized PDF is exp(-||x-center||²/(2σ²)), where:
    - ||x-center|| is the Euclidean distance between x and center
    - σ is the standard deviation parameter
    
    Cryptographic Relevance:
    This function is used to:
    - Calculate acceptance probabilities in MCMC algorithms
    - Estimate quality of sampling algorithms
    - Compute statistical distances between distributions
    - Validate security assumptions in cryptographic proofs
    
    Args:
        x: The point at which to evaluate the PDF (scalar or vector)
        sigma: The standard deviation of the Gaussian (must be positive)
        center: The center of the Gaussian (default: origin)
        
    Returns:
        The probability density at point x
        
    Raises:
        ValueError: If sigma is not positive or dimensions don't match
        
    Example:
        >>> # PDF of a standard discrete Gaussian at x=0
        >>> density = discrete_gaussian_pdf(0, 1.0)
        >>> abs(density - 1.0) < 1e-10  # Should be very close to 1.0
        True
    """
    return discrete_gaussian_pdf(x, sigma, center)


def precompute_discrete_gaussian_probabilities(
    sigma: float, 
    center: Union[float, int] = 0, 
    radius: float = 6
) -> Dict[int, float]:
    """
    Precompute discrete Gaussian probabilities within radius*sigma of center.
    
    Mathematical Context:
    This function precomputes the normalized probability mass function for integers
    in the range [center-radius*sigma, center+radius*sigma]. This is useful for
    efficient repeated sampling from the same distribution.
    
    Cryptographic Relevance:
    Precomputation improves efficiency in:
    - Repeated sampling operations in cryptographic protocols
    - Constant-time implementations resistant to timing attacks
    - Performance-critical applications of lattice-based cryptography
    
    Args:
        sigma: The standard deviation of the Gaussian (must be positive)
        center: The center of the Gaussian (default: 0)
        radius: How many standard deviations to consider (default: 6)
        
    Returns:
        A dictionary mapping integer points to their probabilities
        
    Raises:
        ValueError: If sigma or radius is not positive, or if total probability 
                   mass is too small due to numerical underflow
        
    Example:
        >>> # Precompute probabilities for a standard discrete Gaussian
        >>> probs = precompute_discrete_gaussian_probabilities(1.0)
        >>> sum(probs.values())  # Should sum to approximately 1.0
        1.0
    """
    return precompute_discrete_gaussian_probabilities(sigma, center, radius)


# ---- Diagnostic Functions ----

def compute_autocorrelation(
    samples: List[Vector], 
    lag: int = 50
) -> List[np.ndarray]:
    """
    Compute the autocorrelation of a chain of samples up to a given lag.
    
    Mathematical Context:
    Autocorrelation measures the correlation between samples at different time lags.
    For a stationary process, autocorrelation ρ at lag k is:
    ρ(k) = Cov(X_t, X_{t+k}) / Var(X_t)
    
    Cryptographic Relevance:
    In lattice-based cryptography, autocorrelation analysis helps:
    - Assess quality of MCMC samplers used in cryptographic applications
    - Determine effective sample size for security parameter estimation
    - Validate independence assumptions in security proofs
    - Optimize sampling algorithms for cryptographic use
    
    Args:
        samples: List of sample vectors
        lag: Maximum lag to compute autocorrelation for (default: 50)
        
    Returns:
        A list of autocorrelation arrays, one for each dimension
        
    Raises:
        ValueError: If samples is empty or lag is not positive
        
    Example:
        >>> # Compute autocorrelation for a list of 2D samples
        >>> samples = [vector(RR, [0, 0]) for _ in range(100)]
        >>> acf = compute_autocorrelation(samples)
        >>> len(acf)  # Should have one array per dimension
        2
    """
    return compute_autocorrelation(samples, lag)


def compute_ess(
    samples: List[Vector]
) -> List[float]:
    """
    Compute the Effective Sample Size (ESS) for each dimension of the samples.
    
    Mathematical Context:
    ESS estimates the number of independent samples that would provide the
    same estimation accuracy as the autocorrelated MCMC samples:
    ESS = n / (1 + 2Σρ(k)) where Σρ(k) is the sum of autocorrelations.
    
    Cryptographic Relevance:
    ESS is crucial for lattice-based cryptography because:
    - It helps determine the actual number of independent samples generated
    - It impacts the statistical confidence in security parameter estimates
    - It allows fair comparison between different sampling algorithms
    - It helps optimize the trade-off between sampling quality and efficiency
    
    Args:
        samples: List of sample vectors
        
    Returns:
        A list of ESS values, one for each dimension
        
    Raises:
        ValueError: If samples is empty
        
    Example:
        >>> # Compute ESS for a list of 2D samples
        >>> samples = [vector(RR, [0, 0]) for _ in range(100)]
        >>> ess = compute_ess(samples)
        >>> len(ess)  # Should have one value per dimension
        2
    """
    return compute_ess(samples)


def plot_trace(
    samples: List[Vector], 
    filename: str, 
    title: Optional[str] = None
) -> None:
    """
    Create trace plots for each dimension of the samples.
    
    Mathematical Context:
    Trace plots visualize the values of each dimension over the length of the chain.
    They help assess stationarity (constant mean and variance) and mixing (how 
    well the chain explores the parameter space).
    
    Cryptographic Relevance:
    In lattice-based cryptography, trace plots help:
    - Visually inspect the quality of sampling algorithms
    - Identify potential issues with sampler convergence
    - Determine appropriate burn-in periods
    - Assess whether the sampler is exploring the target distribution properly
    
    Args:
        samples: List of sample vectors
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty
        IOError: If the plot cannot be saved to the specified file
        
    Example:
        >>> # Create trace plots for a list of 2D samples
        >>> samples = [vector(RR, [0, 0]) for _ in range(100)]
        >>> plot_trace(samples, "trace.png", "Trace Plot")
    """
    return plot_trace(samples, filename, title)


def plot_autocorrelation(
    acf_by_dim: List[np.ndarray], 
    filename: str, 
    title: Optional[str] = None
) -> None:
    """
    Plot the autocorrelation function for each dimension.
    
    Mathematical Context:
    Autocorrelation plots show how quickly the dependency between samples
    decays with increasing lag. Rapid decay indicates better mixing of the chain.
    
    Cryptographic Relevance:
    Autocorrelation plots are important for:
    - Assessing independence of samples used in security parameter estimation
    - Comparing efficiency of different sampling algorithms
    - Determining if thinning is necessary for cryptographic applications
    - Validating theoretical mixing time bounds in practice
    
    Args:
        acf_by_dim: List of autocorrelation arrays for each dimension
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If acf_by_dim is empty
        IOError: If the plot cannot be saved to the specified file
        
    Example:
        >>> # Plot autocorrelation for a 2D sample chain
        >>> acf = compute_autocorrelation([vector(RR, [0, 0]) for _ in range(100)])
        >>> plot_autocorrelation(acf, "acf.png", "Autocorrelation Plot")
    """
    return plot_autocorrelation(acf_by_dim, filename, title)


def plot_acceptance_trace(
    accepts: List[bool], 
    filename: str, 
    window_size: int = 100
) -> None:
    """
    Plot the acceptance rate over time using a moving window.
    
    Mathematical Context:
    The acceptance rate in Metropolis-Hastings algorithms measures the proportion
    of proposed moves that are accepted. It indicates how well the proposal
    distribution matches the target distribution.
    
    Cryptographic Relevance:
    Acceptance rate analysis helps:
    - Tune proposal distributions for optimal sampling efficiency
    - Diagnose issues with sampling algorithms used in cryptographic schemes
    - Balance between exploration and exploitation in MCMC sampling
    - Optimize implementation parameters for cryptographic applications
    
    Args:
        accepts: List of booleans indicating acceptance
        filename: Filename to save the plot
        window_size: Size of the moving window (default: 100)
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If accepts is empty or window_size is not positive
        IOError: If the plot cannot be saved to the specified file
        
    Example:
        >>> # Plot acceptance rate for a chain
        >>> accepts = [True, False, True, True, False] * 20
        >>> plot_acceptance_trace(accepts, "acceptance.png")
    """
    return plot_acceptance_trace(accepts, filename, window_size)


# ---- Visualization Functions ----

def plot_2d_samples(
    samples: List[Vector], 
    sigma: float, 
    filename: str, 
    lattice_basis: Optional[Matrix] = None, 
    title: Optional[str] = None, 
    center: Optional[Vector] = None
) -> None:
    """
    Create a 2D scatter plot of the samples with density contours.
    
    Mathematical Context:
    This visualization displays the empirical distribution of samples from a
    discrete Gaussian over a 2D lattice, along with theoretical contours and
    the fundamental domain of the lattice.
    
    Cryptographic Relevance:
    These visualizations help:
    - Verify that samplers produce the correct statistical distribution
    - Understand the geometry of the lattice and its fundamental domain
    - Demonstrate the relationship between sigma and the spread of samples
    - Validate security assumptions for 2D examples in publication figures
    
    Args:
        samples: List of 2D samples
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        lattice_basis: The lattice basis (for plotting the fundamental domain)
        title: Optional title for the plot
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty, contains non-2D points, or sigma is not positive
        IOError: If the plot cannot be saved to the specified file
        
    Example:
        >>> # Plot 2D samples
        >>> samples = [vector(RR, [0, 0]) for _ in range(100)]
        >>> B = matrix.identity(RR, 2)
        >>> plot_2d_samples(samples, 1.0, "samples_2d.png", B, "2D Samples")
    """
    return plot_2d_samples(samples, sigma, filename, lattice_basis, title, center)


def plot_3d_samples(
    samples: List[Vector], 
    sigma: float, 
    filename: str, 
    title: Optional[str] = None, 
    center: Optional[Vector] = None
) -> None:
    """
    Create a 3D scatter plot of the samples.
    
    Mathematical Context:
    This visualization shows the distribution of samples in 3D space, with
    multiple viewpoints to better understand the spatial distribution.
    
    Cryptographic Relevance:
    3D visualizations help:
    - Understand sampling behavior in higher dimensions
    - Verify that the distribution matches theoretical expectations
    - Identify potential biases in sampling algorithms
    - Create publication-quality figures for research papers
    
    Args:
        samples: List of 3D samples
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty, contains non-3D points, or sigma is not positive
        IOError: If the plot cannot be saved to the specified file
        
    Example:
        >>> # Plot 3D samples
        >>> samples = [vector(RR, [0, 0, 0]) for _ in range(100)]
        >>> plot_3d_samples(samples, 1.0, "samples_3d.png", "3D Samples")
    """
    return plot_3d_samples(samples, sigma, filename, title, center)


def plot_2d_projections(
    samples: List[Vector], 
    sigma: float, 
    filename: str, 
    title: Optional[str] = None, 
    center: Optional[Vector] = None
) -> None:
    """
    Create 2D projections of higher-dimensional samples.
    
    Mathematical Context:
    For samples in dimensions > 3, this creates pairwise 2D projections to
    visualize relationships between different dimensions of the distribution.
    
    Cryptographic Relevance:
    Projection plots help:
    - Visualize high-dimensional lattice samples relevant to cryptography
    - Identify correlations between dimensions that might affect security
    - Verify that high-dimensional samplers maintain expected properties
    - Study the behavior of cryptographic constructions in lower dimensions
    
    Args:
        samples: List of samples (3D or higher)
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty, or sigma is not positive
        IOError: If the plot cannot be saved to the specified file
        
    Example:
        >>> # Plot 2D projections of 4D samples
        >>> samples = [vector(RR, [0, 0, 0, 0]) for _ in range(100)]
        >>> plot_2d_projections(samples, 1.0, "projections.png", "4D Sample Projections")
    """
    return plot_2d_projections(samples, sigma, filename, title, center)


def plot_pca_projection(
    samples: List[Vector], 
    sigma: float, 
    filename: str, 
    title: Optional[str] = None
) -> None:
    """
    Create a PCA projection of higher-dimensional samples to 2D.
    
    Mathematical Context:
    Principal Component Analysis (PCA) finds the directions of maximum
    variance in the data and projects the samples onto these principal components,
    allowing visualization of high-dimensional data in 2D.
    
    Cryptographic Relevance:
    PCA projections help:
    - Visualize high-dimensional lattice samples used in cryptography
    - Understand the effective dimensionality of the sampling distribution
    - Identify dominant directions of variance in cryptographic constructions
    - Analyze sampler behavior for high-dimensional lattice-based schemes
    
    Args:
        samples: List of samples (3D or higher)
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty or sigma is not positive
        IOError: If the plot cannot be saved to the specified file
        
    Example:
        >>> # PCA projection of 5D samples
        >>> samples = [vector(RR, [0, 0, 0, 0, 0]) for _ in range(100)]
        >>> plot_pca_projection(samples, 1.0, "pca.png", "PCA Projection")
    """
    return plot_pca_projection(samples, sigma, filename, title)


# ---- Statistical Functions ----

def compute_total_variation_distance(
    samples: List[Vector], 
    sigma: float, 
    lattice_basis: Matrix, 
    center: Optional[Vector] = None
) -> float:
    """
    Compute the total variation distance between the empirical and ideal distributions.
    
    Mathematical Context:
    The total variation distance is defined as:
    TV(μ, ν) = (1/2) * sum_x |μ(x) - ν(x)|
    where μ is the empirical distribution and ν is the theoretical discrete Gaussian.
    
    Cryptographic Relevance:
    Total variation distance is critical because:
    - It directly relates to security assumptions in lattice-based cryptography
    - It quantifies how well a sampler approximates the ideal distribution
    - It helps validate statistical indistinguishability claims
    - It allows comparison of different sampling algorithms for cryptographic use
    
    Args:
        samples: List of lattice point samples
        sigma: The standard deviation used for the Gaussian
        lattice_basis: The lattice basis
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        The estimated total variation distance (between 0 and 1)
        
    Raises:
        ValueError: If samples is empty, sigma is not positive, or dimensions don't match
        
    Example:
        >>> # Compute total variation distance for a set of samples
        >>> samples = [vector(RR, [0, 0]) for _ in range(100)]
        >>> B = matrix.identity(RR, 2)
        >>> tv_dist = compute_total_variation_distance(samples, 1.0, B)
        >>> 0 <= tv_dist <= 1
        True
    """
    return compute_total_variation_distance(samples, sigma, lattice_basis, center)


def compute_kl_divergence(
    samples: List[Vector], 
    sigma: float, 
    lattice_basis: Matrix, 
    center: Optional[Vector] = None
) -> float:
    """
    Compute the KL divergence between the empirical and ideal distributions.
    
    Mathematical Context:
    The Kullback-Leibler divergence is defined as:
    KL(μ||ν) = sum_x μ(x) * log(μ(x)/ν(x))
    where μ is the empirical distribution and ν is the theoretical discrete Gaussian.
    
    Cryptographic Relevance:
    KL divergence is important because:
    - It measures information-theoretic difference between distributions
    - It is related to distinguishing advantage in cryptographic security proofs
    - It penalizes sampling errors differently than total variation distance
    - It provides additional insight into sampler quality for security applications
    
    Args:
        samples: List of lattice point samples
        sigma: The standard deviation used for the Gaussian
        lattice_basis: The lattice basis
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        The estimated KL divergence (non-negative)
        
    Raises:
        ValueError: If samples is empty, sigma is not positive, or dimensions don't match
        
    Example:
        >>> # Compute KL divergence for a set of samples
        >>> samples = [vector(RR, [0, 0]) for _ in range(100)]
        >>> B = matrix.identity(RR, 2)
        >>> kl_div = compute_kl_divergence(samples, 1.0, B)
        >>> kl_div >= 0
        True
    """
    return compute_kl_divergence(samples, sigma, lattice_basis, center)


# ---- Experiment Functions ----

def run_experiment(
    dim: int, 
    sigma: float, 
    num_samples: int, 
    basis_type: str = 'identity', 
    compare_with_klein: bool = True, 
    center: Optional[Vector] = None
) -> Dict[str, Any]:
    """
    Run a complete experiment with IMHK sampling and analysis.
    
    Mathematical Context:
    This function executes a comprehensive experiment to evaluate the quality of 
    discrete Gaussian sampling over lattices using the IMHK algorithm, with optional
    comparison to Klein's algorithm.
    
    Cryptographic Relevance:
    Comprehensive experiments help:
    - Validate sampling algorithms for use in cryptographic schemes
    - Measure the impact of different parameters on security and efficiency
    - Generate data for publication-quality research papers
    - Benchmark performance for practical cryptographic implementations
    
    Args:
        dim: The dimension of the lattice (must be ≥ 2)
        sigma: The standard deviation of the Gaussian (must be positive)
        num_samples: The number of samples to generate (must be positive)
        basis_type: The type of lattice basis to use (default: 'identity')
        compare_with_klein: Whether to compare with Klein's algorithm (default: True)
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        A dictionary containing the experiment results
        
    Raises:
        ValueError: If inputs are invalid (e.g., dim < 2, sigma <= 0)
        
    Example:
        >>> # Run a simple 2D experiment
        >>> results = run_experiment(2, 1.0, 100)
        >>> 'imhk_acceptance_rate' in results
        True
    """
    return run_experiment(dim, sigma, num_samples, basis_type, compare_with_klein, center)


def parameter_sweep(
    dimensions: Optional[List[int]] = None, 
    sigmas: Optional[List[float]] = None, 
    basis_types: Optional[List[str]] = None, 
    centers: Optional[Dict[int, List[Vector]]] = None, 
    num_samples: int = 1000
) -> Dict[Tuple, Dict[str, Any]]:
    """
    Perform a parameter sweep across dimensions, sigmas, basis types, and centers.
    
    Mathematical Context:
    This function systematically explores the parameter space to evaluate
    sampler performance across different configurations, generating comparative
    statistics and visualizations.
    
    Cryptographic Relevance:
    Parameter sweeps are essential for:
    - Determining optimal parameters for lattice-based cryptographic schemes
    - Understanding the trade-off between security and efficiency
    - Establishing confidence in security parameters across different scenarios
    - Identifying edge cases where sampling quality might degrade
    
    Args:
        dimensions: List of dimensions to test (default: [2, 3, 4])
        sigmas: List of sigma values to test (default: [0.5, 1.0, 2.0, 5.0])
        basis_types: List of basis types (default: ['identity', 'skewed', 'ill-conditioned'])
        centers: Dictionary mapping dimensions to lists of centers (default: origin)
        num_samples: Number of samples for each configuration (default: 1000)
        
    Returns:
        A dictionary of results indexed by configuration
        
    Raises:
        ValueError: If input parameters are invalid
        
    Example:
        >>> # Run a minimal parameter sweep
        >>> results = parameter_sweep(dimensions=[2], sigmas=[1.0], num_samples=100)
        >>> len(results) > 0
        True
    """
    return parameter_sweep(dimensions, sigmas, basis_types, centers, num_samples)


def plot_parameter_sweep_results(
    results: Dict[Tuple, Dict[str, Any]], 
    dimensions: List[int], 
    sigmas: List[float], 
    basis_types: List[str]
) -> None:
    """
    Create comparative plots for the parameter sweep results.
    
    Mathematical Context:
    This function generates visualizations to compare the performance of
    sampling algorithms across different parameter configurations, including
    acceptance rates, total variation distances, and quality ratios.
    
    Cryptographic Relevance:
    These comparative visualizations help:
    - Identify optimal parameters for lattice-based cryptography
    - Understand how sampling quality varies with sigma and dimension
    - Evaluate the impact of basis quality on security properties
    - Generate publication-quality figures for research papers
    
    Args:
        results: Dictionary of results from parameter_sweep
        dimensions: List of dimensions tested
        sigmas: List of sigma values tested
        basis_types: List of basis types tested
        
    Returns:
        None (saves plots to files)
        
    Raises:
        ValueError: If inputs are invalid or incompatible with results
        IOError: If plots cannot be saved
        
    Example:
        >>> # Plot results from a parameter sweep
        >>> results = parameter_sweep(dimensions=[2], sigmas=[1.0], num_samples=100)
        >>> plot_parameter_sweep_results(results, [2], [1.0], ['identity'])
    """
    return plot_parameter_sweep_results(results, dimensions, sigmas, basis_types)


def compare_convergence_times(
    results: Dict[Tuple, Dict[str, Any]]
) -> None:
    """
    Analyze and compare convergence times across different configurations.
    
    Mathematical Context:
    This function analyzes the effective time required to generate statistically
    independent samples, accounting for autocorrelation in MCMC methods through
    the Effective Sample Size (ESS) adjustment.
    
    Cryptographic Relevance:
    Convergence time analysis helps:
    - Determine the computational cost of high-quality sampling
    - Optimize implementation choices for practical cryptographic systems
    - Balance security (sampling quality) with efficiency (speed)
    - Identify parameter regimes where algorithms are most efficient
    
    Args:
        results: Dictionary of results from parameter_sweep
        
    Returns:
        None (saves plots to files)
        
    Raises:
        ValueError: If results is empty or has invalid structure
        IOError: If plots cannot be saved
        
    Example:
        >>> # Compare convergence times from a parameter sweep
        >>> results = parameter_sweep(dimensions=[2], sigmas=[1.0], num_samples=100)
        >>> compare_convergence_times(results)
    """
    return compare_convergence_times(results)