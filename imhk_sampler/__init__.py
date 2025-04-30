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
    In Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms 
    (pp. 937-941).
[2] Ducas, L., & Prest, T. (2016). Fast Fourier orthogonalization. 
    In Proceedings of the ACM on International Symposium on Symbolic and 
    Algebraic Computation (pp. 191-198).
[3] Gentry, C., Peikert, C., & Vaikuntanathan, V. (2008). Trapdoors for hard 
    lattices and new cryptographic constructions. In Proceedings of the 40th 
    Annual ACM Symposium on Theory of Computing (pp. 197-206).
"""

__version__ = '0.1.0'
__author__ = 'Nick Qrumpton, Lattice Cryptography Research Group'

from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from pathlib import Path
# Correct SageMath imports for version 10.5 compatibility
from sage.all import matrix, vector, RR

# Create necessary directories for results
def setup_directories():
    """Create necessary directories for results, logs, and plots."""
    dirs = [
        Path('results'),
        Path('results/plots'),
        Path('results/logs'),
        Path('data')
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

# Function to dynamically import modules to avoid circular dependencies
def get_function(module_name, function_name):
    """
    Dynamically import a function from a module to avoid circular dependencies.
    
    Args:
        module_name: Name of the module (e.g., 'samplers', 'utils')
        function_name: Name of the function to import
        
    Returns:
        The imported function
    """
    import importlib
    
    # Full module path
    full_module_name = f"imhk_sampler.{module_name}"
    
    # Import the module
    module = importlib.import_module(full_module_name)
    
    # Get and return the function
    return getattr(module, function_name)

# Create necessary directories on import
setup_directories()

# Define wrapper functions that use dynamic imports to avoid circular references
def imhk_sampler(B, sigma, num_samples, center=None, burn_in=1000):
    """Independent Metropolis-Hastings-Klein algorithm for sampling from a discrete Gaussian."""
    _imhk_sampler = get_function('samplers', 'imhk_sampler')
    return _imhk_sampler(B, sigma, num_samples, center, burn_in)

def klein_sampler(B, sigma, center=None):
    """Klein's algorithm for sampling from a discrete Gaussian over a lattice."""
    _klein_sampler = get_function('samplers', 'klein_sampler')
    return _klein_sampler(B, sigma, center)

def discrete_gaussian_sampler_1d(center, sigma):
    """Sample from a 1D discrete Gaussian distribution."""
    _discrete_gaussian_sampler_1d = get_function('samplers', 'discrete_gaussian_sampler_1d')
    return _discrete_gaussian_sampler_1d(center, sigma)

def discrete_gaussian_pdf(x, sigma, center=None):
    """Compute the probability density function of a discrete Gaussian."""
    _discrete_gaussian_pdf = get_function('utils', 'discrete_gaussian_pdf')
    return _discrete_gaussian_pdf(x, sigma, center)

def precompute_discrete_gaussian_probabilities(sigma, center=0, radius=6):
    """Precompute discrete Gaussian probabilities within radius*sigma of center."""
    _precompute = get_function('utils', 'precompute_discrete_gaussian_probabilities')
    return _precompute(sigma, center, radius)

def compute_autocorrelation(samples, lag=50):
    """Compute the autocorrelation of a chain of samples up to a given lag."""
    _compute_autocorrelation = get_function('diagnostics', 'compute_autocorrelation')
    return _compute_autocorrelation(samples, lag)

def compute_ess(samples):
    """Compute the Effective Sample Size (ESS) for each dimension of the samples."""
    _compute_ess = get_function('diagnostics', 'compute_ess')
    return _compute_ess(samples)

def plot_trace(samples, filename, title=None):
    """Create trace plots for each dimension of the samples."""
    _plot_trace = get_function('diagnostics', 'plot_trace')
    return _plot_trace(samples, filename, title)

def plot_autocorrelation(acf_by_dim, filename, title=None):
    """Plot the autocorrelation function for each dimension."""
    _plot_autocorrelation = get_function('diagnostics', 'plot_autocorrelation')
    return _plot_autocorrelation(acf_by_dim, filename, title)

def plot_acceptance_trace(accepts, filename, window_size=100):
    """Plot the acceptance rate over time using a moving window."""
    _plot_acceptance_trace = get_function('diagnostics', 'plot_acceptance_trace')
    return _plot_acceptance_trace(accepts, filename, window_size)

def plot_2d_samples(samples, sigma, filename, lattice_basis=None, title=None, center=None):
    """Create a 2D scatter plot of the samples with density contours."""
    _plot_2d_samples = get_function('visualization', 'plot_2d_samples')
    return _plot_2d_samples(samples, sigma, filename, lattice_basis, title, center)

def plot_3d_samples(samples, sigma, filename, title=None, center=None):
    """Create a 3D scatter plot of the samples."""
    _plot_3d_samples = get_function('visualization', 'plot_3d_samples')
    return _plot_3d_samples(samples, sigma, filename, title, center)

def plot_2d_projections(samples, sigma, filename, title=None, center=None):
    """Create 2D projections of higher-dimensional samples."""
    _plot_2d_projections = get_function('visualization', 'plot_2d_projections')
    return _plot_2d_projections(samples, sigma, filename, title, center)

def plot_pca_projection(samples, sigma, filename, title=None):
    """Create a PCA projection of higher-dimensional samples to 2D."""
    _plot_pca_projection = get_function('visualization', 'plot_pca_projection')
    return _plot_pca_projection(samples, sigma, filename, title)

def compute_total_variation_distance(samples, sigma, lattice_basis, center=None):
    """Compute the total variation distance between empirical and ideal distributions."""
    _compute_tv = get_function('stats', 'compute_total_variation_distance')
    return _compute_tv(samples, sigma, lattice_basis, center)

def compute_kl_divergence(samples, sigma, lattice_basis, center=None):
    """Compute the KL divergence between the empirical and ideal distributions."""
    _compute_kl = get_function('stats', 'compute_kl_divergence')
    return _compute_kl(samples, sigma, lattice_basis, center)

def run_experiment(dim, sigma, num_samples, basis_type='identity', compare_with_klein=True, center=None):
    """Run a complete experiment with IMHK sampling and analysis."""
    _run_experiment = get_function('experiments', 'run_experiment')
    return _run_experiment(dim, sigma, num_samples, basis_type, compare_with_klein, center)

def parameter_sweep(dimensions=None, sigmas=None, basis_types=None, centers=None, num_samples=1000):
    """Perform a parameter sweep across dimensions, sigmas, basis types, and centers."""
    _parameter_sweep = get_function('experiments', 'parameter_sweep')
    return _parameter_sweep(dimensions, sigmas, basis_types, centers, num_samples)

def plot_parameter_sweep_results(results, dimensions, sigmas, basis_types):
    """Create comparative plots for the parameter sweep results."""
    _plot_results = get_function('experiments', 'plot_parameter_sweep_results')
    return _plot_results(results, dimensions, sigmas, basis_types)

def compare_convergence_times(results):
    """Analyze and compare convergence times across different configurations."""
    _compare_times = get_function('experiments', 'compare_convergence_times')
    return _compare_times(results)
