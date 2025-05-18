"""
Adaptive parameter configuration for lattice experiments.

This module handles computation of smoothing parameters and
ratio-based experiment configuration.
"""

import numpy as np
from sage.all import matrix, vector, sqrt, log, pi, RDF
import logging

logger = logging.getLogger(__name__)


def compute_smoothing_parameter(lattice_basis, epsilon=0.01):
    """
    Compute the smoothing parameter η_ε(Λ) for a given lattice.
    
    The smoothing parameter is the smallest σ such that the statistical
    distance between the discrete Gaussian D_{Λ,σ} and the continuous
    Gaussian is at most ε.
    
    Args:
        lattice_basis: Sage matrix representing the lattice basis
        epsilon: Statistical distance parameter (default: 0.01)
        
    Returns:
        float: The smoothing parameter η_ε(Λ)
        
    Raises:
        ValueError: If lattice_basis is invalid
    """
    if not hasattr(lattice_basis, 'nrows'):
        lattice_basis = matrix(lattice_basis)
    
    n = lattice_basis.nrows()
    
    if lattice_basis.ncols() != n:
        raise ValueError("Lattice basis must be square")
    
    if lattice_basis.rank() < n:
        raise ValueError("Lattice basis must be full rank")
    
    # Compute Gram-Schmidt orthogonalization
    try:
        B_gso, mu = lattice_basis.gram_schmidt()
        
        # Find minimum Gram-Schmidt vector length
        min_gs_norm = min(vector(b).norm() for b in B_gso)
        
        # Upper bound for smoothing parameter
        eta = sqrt(log(2 * n * (1 + 1/epsilon)) / pi) / min_gs_norm
        
        logger.info(f"Computed smoothing parameter: η_{epsilon}(Λ) = {float(eta):.6f}")
        return float(eta)
        
    except Exception as e:
        logger.error(f"Failed to compute smoothing parameter: {e}")
        # Fallback to a conservative estimate
        return float(sqrt(log(2 * n / epsilon) / pi))


def get_experiment_ratios():
    """
    Get the standard σ/η ratios for experiments.
    
    Returns:
        list: Ratios to test
    """
    return [0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]


def compute_sigma_from_ratio(ratio, smoothing_param):
    """
    Compute sigma value from ratio and smoothing parameter.
    
    Args:
        ratio: The σ/η ratio
        smoothing_param: The smoothing parameter η
        
    Returns:
        float: The sigma value
    """
    sigma = ratio * smoothing_param
    
    # Ensure minimum sigma for numerical stability
    min_sigma = 0.1
    if sigma < min_sigma:
        logger.warning(f"Sigma {sigma:.6f} below minimum {min_sigma}, adjusting")
        sigma = min_sigma
    
    return sigma


def should_skip_experiment(ratio, tv_distance=None):
    """
    Determine if an experiment should be skipped based on ratio or results.
    
    Args:
        ratio: The σ/η ratio
        tv_distance: Optional TV distance from previous run
        
    Returns:
        bool: True if experiment should be skipped
        str: Reason for skipping (or empty string)
    """
    # Skip very small ratios
    if ratio < 0.5:
        return True, f"Ratio {ratio} below minimum threshold 0.5"
    
    # Skip if TV distance is extremely small (perfect mixing)
    if tv_distance is not None and tv_distance < 1e-6:
        return True, f"TV distance {tv_distance} indicates perfect mixing"
    
    # Skip extremely large ratios that would be instantaneous
    if ratio > 10.0:
        return True, f"Ratio {ratio} too large for meaningful results"
    
    return False, ""


def get_basis_info(basis_type, dimension):
    """
    Get standardized basis information for experiments.
    
    Args:
        basis_type: Type of basis ('identity', 'skewed', 'ill_conditioned')
        dimension: Lattice dimension
        
    Returns:
        dict: Basis configuration information
    """
    config = {
        'type': basis_type,
        'dimension': dimension,
        'name': f"{basis_type}_{dimension}d"
    }
    
    # Add specific parameters for different basis types
    if basis_type == 'ill_conditioned':
        config['condition_number'] = 10 ** dimension
    elif basis_type == 'skewed':
        config['skew_factor'] = 0.4
    
    return config


def get_experiment_parameters(dimension, basis_types):
    """
    Generate experiment parameters for given dimension and basis types.
    
    Parameters
    ----------
    dimension : int
        Lattice dimension
    basis_types : list of str
        List of basis types to test
    
    Returns
    -------
    list of dict
        Experiment configurations
    """
    configs = []
    ratios = get_experiment_ratios()
    
    for basis_type in basis_types:
        basis_info = get_basis_info(basis_type, dimension)
        
        for ratio in ratios:
            config = {
                'dimension': dimension,
                'basis_type': basis_type,
                'basis_info': basis_info,
                'ratio': ratio
            }
            configs.append(config)
    
    return configs


def generate_experiment_configs(dimensions, basis_types, ratios):
    """
    Generate all experiment configurations.
    
    Parameters
    ----------
    dimensions : list of int
        Dimensions to test
    basis_types : list of str
        Basis types to test
    ratios : list of float
        σ/η ratios to test
    
    Returns
    -------
    list of dict
        Experiment configurations
    """
    configs = []
    
    for dim in dimensions:
        for basis_type in basis_types:
            for ratio in ratios:
                config = {
                    'dimension': dim,
                    'basis_type': basis_type,
                    'ratio': ratio
                }
                configs.append(config)
    
    return configs


if __name__ == "__main__":
    # Example usage
    from utils import create_lattice_basis
    
    basis = create_lattice_basis(2, "identity")
    eta = compute_smoothing_parameter(basis)
    print(f"Smoothing parameter: {eta:.4f}")
    
    configs = get_experiment_parameters(2, ["identity", "skewed"])
    for config in configs[:5]:
        print(config)