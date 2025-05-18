"""
Optimized statistical utilities for IMHK sampler with performance improvements.

This module provides efficient implementations of statistical functions with:
- Early stopping criteria
- Progress logging
- Adaptive sampling strategies
- Interrupt handling
"""

import numpy as np
import logging
import time
import signal
from sage.all import vector, matrix, RDF, RR
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, List

# Configure module logger
logger = logging.getLogger("imhk_stats_optimized")

# Global flag for interrupt handling
_interrupted = False

def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _interrupted
    _interrupted = True
    logger.warning("Interrupt received. Attempting graceful termination...")

@contextmanager
def interrupt_handler():
    """Context manager for handling interrupts during computation."""
    global _interrupted
    _interrupted = False
    old_handler = signal.signal(signal.SIGINT, _signal_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)
        _interrupted = False

def tv_distance_discrete_gaussian_optimized(
    lattice_basis, 
    base_sigma, 
    samples, 
    max_radius=None,
    convergence_threshold=1e-4,
    progress_interval=5.0,
    max_points=10000,
    adaptive_sampling=True
):
    """
    Compute total variation distance with optimization and progress tracking.
    
    Key optimizations:
    1. Adaptive radius selection based on dimension
    2. Early stopping when convergence is achieved
    3. Importance sampling for high dimensions
    4. Progress logging for monitoring
    5. Interrupt handling for graceful termination
    
    Args:
        lattice_basis: Lattice basis vectors as a matrix
        base_sigma: Base standard deviation for the Gaussian 
        samples: Array of samples from the lattice
        max_radius: Maximum radius to consider (auto-computed if None)
        convergence_threshold: Early stopping threshold
        progress_interval: Seconds between progress logs
        max_points: Maximum lattice points to consider
        adaptive_sampling: Use adaptive sampling for high dimensions
    
    Returns:
        Total variation distance (float between 0 and 1)
    """
    n = samples.shape[1]
    sigma = base_sigma * np.sqrt(2 * np.pi)
    num_samples = len(samples)
    
    # Auto-compute appropriate radius based on dimension
    if max_radius is None:
        # Use smaller radius for higher dimensions to keep computation tractable
        max_radius = max(2, int(5.0 / np.sqrt(n)))
        logger.info(f"Auto-selected max_radius={max_radius} for dimension {n}")
    
    logger.info(f"Starting TV distance computation: dim={n}, sigma={sigma:.4f}, "
                f"samples={num_samples}, max_radius={max_radius}")
    
    # Convert lattice basis to numpy array for efficiency
    basis_matrix = np.array(lattice_basis)
    
    # Compute empirical distribution from samples
    sample_counts = {}
    for sample in samples:
        key = tuple(sample)
        sample_counts[key] = sample_counts.get(key, 0) + 1
    
    # Initialize tracking variables
    lattice_points = []
    points_checked = 0
    last_progress_time = time.time()
    
    with interrupt_handler():
        # For high dimensions, use Monte Carlo sampling of lattice points
        if n > 8 and adaptive_sampling:
            logger.info(f"Using adaptive Monte Carlo sampling for dimension {n}")
            
            # Sample lattice points randomly
            num_mc_samples = min(max_points, (2 * max_radius + 1) ** min(n, 8))
            
            for i in range(num_mc_samples):
                if _interrupted:
                    logger.warning("Computation interrupted")
                    break
                
                # Generate random coefficients
                coeffs = np.random.randint(-max_radius, max_radius + 1, size=n)
                point = np.zeros(n)
                for j, c in enumerate(coeffs):
                    point += c * basis_matrix[j]
                
                if np.linalg.norm(point) <= max_radius * sigma:
                    lattice_points.append((point, tuple(coeffs)))
                
                points_checked += 1
                
                # Progress logging
                current_time = time.time()
                if current_time - last_progress_time > progress_interval:
                    logger.info(f"Progress: {points_checked}/{num_mc_samples} points checked, "
                               f"{len(lattice_points)} valid points found")
                    last_progress_time = current_time
        
        else:
            # Enumerate lattice points systematically for low dimensions
            logger.info(f"Using systematic enumeration for dimension {n}")
            
            from itertools import product
            ranges = [range(-max_radius, max_radius + 1) for _ in range(n)]
            
            # Count total combinations for progress tracking
            total_combinations = (2 * max_radius + 1) ** n
            
            for coeffs in product(*ranges):
                if _interrupted:
                    logger.warning("Computation interrupted")
                    break
                
                if points_checked >= max_points:
                    logger.warning(f"Reached maximum point limit ({max_points})")
                    break
                
                # Compute the lattice point
                point = np.zeros(n)
                for i, c in enumerate(coeffs):
                    point += c * basis_matrix[i]
                
                if np.linalg.norm(point) <= max_radius * sigma:
                    lattice_points.append((point, coeffs))
                
                points_checked += 1
                
                # Progress logging
                current_time = time.time()
                if current_time - last_progress_time > progress_interval:
                    progress_pct = (points_checked / total_combinations) * 100
                    logger.info(f"Progress: {progress_pct:.1f}% ({points_checked}/{total_combinations}), "
                               f"{len(lattice_points)} valid points found")
                    last_progress_time = current_time
    
    if _interrupted:
        logger.warning(f"Computation interrupted after checking {points_checked} points")
        # Continue with partial results
    
    logger.info(f"Found {len(lattice_points)} lattice points within radius")
    
    # Compute true probabilities for each lattice point
    total_true_prob = 0
    true_probs = {}
    
    for idx, (point, coeffs) in enumerate(lattice_points):
        if _interrupted:
            break
            
        # Use numpy computation for efficiency
        norm_squared = np.sum(point ** 2)
        prob = np.exp(-norm_squared / (2 * sigma**2))
        
        point_tuple = tuple(point)
        true_probs[point_tuple] = prob
        total_true_prob += prob
        
        # Early stopping check
        if idx > 100 and idx % 100 == 0:
            # Check if remaining probability mass is negligible
            remaining_prob = np.exp(-(max_radius * sigma)**2 / (2 * sigma**2))
            if remaining_prob < convergence_threshold:
                logger.info(f"Early stopping: remaining probability mass < {convergence_threshold}")
                break
    
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
    
    final_tv = tv_dist / 2  # TV distance formula includes a factor of 1/2
    
    logger.info(f"TV distance computation complete: {final_tv:.6f} "
                f"(checked {points_checked} points, found {len(lattice_points)} valid)")
    
    return final_tv


def compute_total_variation_distance(
    samples, 
    sigma, 
    lattice_basis, 
    center=None, 
    **kwargs
):
    """
    Compute total variation distance with optimizations.
    
    This is the main interface that replaces the original function.
    
    Args:
        samples: List or array of samples from the lattice
        sigma: Standard deviation of the Gaussian
        lattice_basis: Lattice basis vectors as a matrix
        center: Center of the Gaussian distribution (default: origin)
        **kwargs: Additional arguments passed to optimized implementation
            - max_radius: Maximum radius to consider
            - convergence_threshold: Early stopping threshold
            - progress_interval: Seconds between progress logs
            - max_points: Maximum lattice points to consider
            - adaptive_sampling: Use adaptive sampling for high dimensions
    
    Returns:
        float: Total variation distance between 0 and 1
    """
    # Parameter validation
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    
    try:
        # Convert samples to numpy array if needed
        if not isinstance(samples, np.ndarray):
            samples = np.array([list(s) for s in samples])
        
        # Check for empty samples
        if len(samples) == 0:
            logger.warning("No samples provided, returning NaN")
            return np.nan
        
        # Handle center offset if provided
        if center is not None:
            logger.info("Adjusting samples for non-zero center")
            center_array = np.array(center)
            samples = samples - center_array
        
        # Get dimension and log computational complexity
        n = samples.shape[1]
        max_radius = kwargs.get('max_radius')
        if max_radius is None:
            max_radius = max(2, int(5.0 / np.sqrt(n)))
        
        estimated_points = (2 * max_radius + 1) ** n
        logger.info(f"Dimension: {n}, Estimated search space: {estimated_points:.2e} points")
        
        # Warn for high-dimensional cases
        if n > 16:
            logger.warning(f"High dimension ({n}) detected. Using adaptive sampling. "
                          "Results may be approximate.")
        
        # Use the optimized function
        tv_dist = tv_distance_discrete_gaussian_optimized(
            lattice_basis, 
            sigma, 
            samples,
            **kwargs
        )
        
        # Ensure result is in valid range
        tv_dist = np.clip(tv_dist, 0.0, 1.0)
        
        return tv_dist
        
    except Exception as e:
        logger.error(f"Error computing TV distance: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


# Additional utility functions

def estimate_tv_distance_sample_size(dimension, sigma, target_accuracy=0.01):
    """
    Estimate the number of samples needed for accurate TV distance computation.
    
    Args:
        dimension: Lattice dimension
        sigma: Standard deviation parameter
        target_accuracy: Desired accuracy for TV distance
    
    Returns:
        Recommended number of samples
    """
    # Heuristic based on dimension and sigma
    base_samples = 1000
    dimension_factor = 2 ** (dimension / 4)
    sigma_factor = max(1, 1 / sigma)
    
    recommended = int(base_samples * dimension_factor * sigma_factor)
    
    # Cap at reasonable maximum
    max_samples = 100000
    recommended = min(recommended, max_samples)
    
    logger.info(f"Recommended sample size for dim={dimension}, sigma={sigma}: {recommended}")
    
    return recommended


def diagnose_sampling_quality(samples, lattice_basis, sigma):
    """
    Diagnose potential issues with sampling quality.
    
    Args:
        samples: Array of samples
        lattice_basis: Lattice basis matrix
        sigma: Gaussian parameter
    
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {}
    
    # Check sample uniqueness
    unique_samples = len(set(map(tuple, samples)))
    diagnostics['unique_ratio'] = unique_samples / len(samples)
    
    # Check sample spread
    sample_norms = np.linalg.norm(samples, axis=1)
    diagnostics['mean_norm'] = np.mean(sample_norms)
    diagnostics['std_norm'] = np.std(sample_norms)
    diagnostics['expected_norm'] = sigma * np.sqrt(samples.shape[1])
    
    # Check for concentration
    center_samples = np.sum(np.all(samples == 0, axis=1))
    diagnostics['center_concentration'] = center_samples / len(samples)
    
    logger.info(f"Sampling diagnostics: {diagnostics}")
    
    return diagnostics