"""Statistical utilities for IMHK sampler."""

import numpy as np
from sage.all import vector, matrix, RDF, RR
import logging

# Set up module logger
logger = logging.getLogger("imhk_stats")

# Import optimized version
try:
    from stats_optimized import (
        compute_total_variation_distance as compute_total_variation_distance_optimized,
        estimate_tv_distance_sample_size,
        diagnose_sampling_quality
    )
    USE_OPTIMIZED = True
    logger.info("Using optimized TV distance computation")
except ImportError:
    try:
        from .stats_optimized import (
            compute_total_variation_distance as compute_total_variation_distance_optimized,
            estimate_tv_distance_sample_size,
            diagnose_sampling_quality
        )
        USE_OPTIMIZED = True
        logger.info("Using optimized TV distance computation")
    except ImportError:
        USE_OPTIMIZED = False
        logger.warning("Optimized version not available, using standard implementation")


def tv_distance_discrete_gaussian(lattice_basis, base_sigma, samples, max_radius=5):
    """
    Compute total variation distance between sampled distribution and true discrete Gaussian
    
    Args:
        lattice_basis: Lattice basis vectors as a matrix
        base_sigma: Base standard deviation for the Gaussian 
        samples: Array of samples from the lattice
        max_radius: Maximum radius to consider for probability computation (default: 5)
    
    Returns:
        Total variation distance (float between 0 and 1)
    """
    n = samples.shape[1]
    sigma = base_sigma * np.sqrt(2 * np.pi)
    num_samples = len(samples)
    
    # Convert lattice basis to numpy array for efficiency
    basis_matrix = np.array(lattice_basis)
    
    # Compute empirical distribution from samples
    sample_counts = {}
    for sample in samples:
        key = tuple(sample)
        sample_counts[key] = sample_counts.get(key, 0) + 1
    
    # Enumerate lattice points within radius
    lattice_points = []
    
    # Handle the 2D case
    if n == 2:
        for i in range(-max_radius, max_radius + 1):
            for j in range(-max_radius, max_radius + 1):
                point = i * basis_matrix[0] + j * basis_matrix[1]
                if np.linalg.norm(point) <= max_radius * sigma:
                    lattice_points.append((point, (i, j)))
    else:
        # For higher dimensions, use a more general approach
        from itertools import product
        ranges = [range(-max_radius, max_radius + 1) for _ in range(n)]
        
        for coeffs in product(*ranges):
            # Compute the lattice point
            point = np.zeros(n)
            for i, c in enumerate(coeffs):
                point += c * basis_matrix[i]
            
            if np.linalg.norm(point) <= max_radius * sigma:
                lattice_points.append((point, coeffs))
    
    # Compute true probabilities for each lattice point
    total_true_prob = 0
    true_probs = {}
    
    for point, coeffs in lattice_points:
        # Get the center of the distribution (origin)
        center = np.zeros(n)
        
        # Compute the field-consistent lattice point
        base_field = RDF
        
        # Start with the center
        sage_point = vector(base_field, center)
        
        # Add contributions from basis vectors
        for i, c in enumerate(coeffs):
            basis_row = vector(base_field, lattice_basis.row(i))
            sage_point = sage_point + base_field(c) * basis_row
        
        # Compute probability
        # Note: We use the squared norm directly rather than converting back to numpy
        norm_squared = sum(x*x for x in sage_point - vector(base_field, center))
        prob = np.exp(-float(norm_squared) / (2 * sigma**2))
        
        point_tuple = tuple(point)
        true_probs[point_tuple] = prob
        total_true_prob += prob
    
    # Normalize true probabilities
    if total_true_prob > 0:
        for key in true_probs:
            true_probs[key] /= total_true_prob
    
    # Compute empirical probabilities
    empirical_probs = {}
    for key, count in sample_counts.items():
        empirical_probs[key] = count / num_samples
    
    # Compute TV distance
    tv_dist = 0
    
    # Add contribution from points in true distribution
    for point_tuple, true_prob in true_probs.items():
        emp_prob = empirical_probs.get(point_tuple, 0)
        tv_dist += abs(true_prob - emp_prob)
    
    # Add contribution from points only in empirical distribution
    for point_tuple, emp_prob in empirical_probs.items():
        if point_tuple not in true_probs:
            tv_dist += emp_prob
    
    return tv_dist / 2  # TV distance formula includes a factor of 1/2


def compute_total_variation_distance(samples, sigma, lattice_basis, center=None, max_radius=5, **kwargs):
    """
    Compute total variation distance between sampled distribution and true discrete Gaussian.
    
    This function serves as the primary interface for TV distance calculations, matching
    the signature expected by the experiments module.
    
    Args:
        samples: List or array of samples from the lattice
        sigma: Standard deviation of the Gaussian, must be positive
        lattice_basis: Lattice basis vectors as a matrix
        center: Center of the Gaussian distribution (default: origin)
        max_radius: Maximum radius to consider for probability computation
        **kwargs: Additional arguments for optimized implementation
    
    Returns:
        float: Total variation distance between 0 and 1, or np.nan if computation fails
        
    Raises:
        ValueError: If sigma <= 0 or other invalid parameters
    """
    # Use optimized version if available
    if USE_OPTIMIZED:
        return compute_total_variation_distance_optimized(
            samples, sigma, lattice_basis, center, 
            max_radius=max_radius, **kwargs
        )
    
    # Fallback to original implementation
    # Parameter validation
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    
    try:
        # Convert samples to numpy array if needed
        if not isinstance(samples, np.ndarray):
            samples = np.array([list(s) for s in samples])
        
        # Check for empty samples
        if len(samples) == 0:
            return np.nan
        
        # Use the existing tv_distance_discrete_gaussian function
        tv_dist = tv_distance_discrete_gaussian(lattice_basis, sigma, samples, max_radius)
        
        # Ensure result is in valid range
        tv_dist = np.clip(tv_dist, 0.0, 1.0)
        
        return tv_dist
        
    except Exception as e:
        logger.warning(f"TV distance computation failed: {e}")
        return np.nan


def compute_kl_divergence(samples, sigma, lattice_basis, center=None):
    """
    Deprecated: KL divergence computation disabled.
    
    Args:
        samples: List or array of samples from the lattice
        sigma: Standard deviation of the Gaussian
        lattice_basis: Lattice basis vectors as a matrix  
        center: Center of the Gaussian distribution
    
    Returns:
        None (disabled for this implementation)
    """
    # KL divergence disabled - focusing on TV distance only
    return None