"""
Utility functions for discrete Gaussian calculations in the IMHK Sampler framework.

This module provides core mathematical functions required by the IMHK Sampler,
particularly for discrete Gaussian distribution calculations with a focus
on numerical stability.
"""

from sage.structure.element import Vector
import numpy as np
from sage.all import *
from typing import Union, Tuple, Optional, Dict, List
from math import floor, ceil, exp, log
import logging

# Set up module logger
logger = logging.getLogger("imhk_utils")

def import_module_function(module_name, function_name):
    """
    Dynamically import a function to avoid circular imports.
    
    Args:
        module_name (str): Name of the module within imhk_sampler
        function_name (str): Name of the function to import
        
    Returns:
        Imported function
        
    Example:
        >>> discrete_gaussian_pdf = import_module_function('samplers', 'discrete_gaussian_pdf')
    """
    import importlib
    
    module = importlib.import_module(f"imhk_sampler.{module_name}")
    return getattr(module, function_name)


def get_config():
    """Get the Config class from config module."""
    import importlib
    try:
        config_module = importlib.import_module("imhk_sampler.config")
        return config_module.Config
    except ImportError:
        logger.warning("Config module not found, using default paths")
        return None


def discrete_gaussian_pdf(x: Union[int, float, Vector, list, tuple, np.ndarray],
                         sigma: float,
                         center: Optional[Union[int, float, Vector, list, tuple, np.ndarray]] = None) -> float:
    """
    Compute the probability density function of a discrete Gaussian distribution.
    
    Mathematical Formula:
    ρ_σ,c(x) = exp(-||x - c||² / (2σ²))
    
    This is the unnormalized probability density function where:
    - ||x - c|| is the Euclidean distance between x and center c
    - σ is the standard deviation of the distribution
    
    Cryptographic Relevance:
    Discrete Gaussian distributions are fundamental in lattice-based cryptography.
    They are used in schemes like:
    - Ring-LWE and LWE-based encryption
    - Lattice-based signatures (e.g., FALCON, CRYSTALS-Dilithium)
    - Trapdoor sampling for security reductions
    - Generating error terms with provable security properties
    
    Args:
        x: The point to evaluate (can be a vector or scalar)
        sigma: The standard deviation of the Gaussian
        center: The center of the Gaussian (default: origin)
        
    Returns:
        The probability density at point x
        
    Raises:
        ValueError: If sigma is not positive or if dimensions of x and center don't match
    """
    # Validate sigma
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    
    # Handle default center
    if center is None:
        if isinstance(x, (list, tuple, np.ndarray)) or isinstance(x, Vector):
            center = vector(RR, [0] * len(x))
        else:
            center = 0
    
    # Convert lists/tuples/arrays to Sage vectors
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            x = vector(RR, x)
        except Exception as e:
            logger.error(f"Error converting input to Sage vector: {e}")
            raise ValueError(f"Failed to convert input to Sage vector: {e}") from e
    
    if isinstance(center, (list, tuple, np.ndarray)):
        try:
            center = vector(RR, center)
        except Exception as e:
            logger.error(f"Error converting center to Sage vector: {e}")
            raise ValueError(f"Failed to convert center to Sage vector: {e}") from e
    
    # Check dimensions match if both are vectors
    if isinstance(x, Vector) and isinstance(center, Vector) and len(x) != len(center):
        raise ValueError(f"Dimension mismatch: x has dimension {len(x)}, center has dimension {len(center)}")
    
    try:
        if isinstance(x, Vector):
            # Compute the squared norm of (x - center) using dot product for numerical stability
            diff = vector(RR, x) - vector(RR, center)
            squared_norm = diff.dot_product(diff)
            
            # Handle potential numerical issues
            if squared_norm < 0:
                logger.warning(f"Negative squared norm detected: {squared_norm}, using absolute value")
                squared_norm = abs(squared_norm)
                
            # Check for potential overflow
            if squared_norm > 700 * sigma * sigma:
                logger.debug(f"Large squared norm detected: {squared_norm}, result will be close to zero")
                return 1e-300  # Effectively zero but not exactly zero to avoid division issues
                
            return exp(-squared_norm / (2 * sigma * sigma))
        else:
            # Scalar case
            diff_squared = (x - center) ** 2
            
            # Check for potential overflow
            if diff_squared > 700 * sigma * sigma:
                logger.debug(f"Large squared difference detected: {diff_squared}, result will be close to zero")
                return 1e-300  # Effectively zero but not exactly zero
                
            return exp(-diff_squared / (2 * sigma * sigma))
            
    except Exception as e:
        logger.error(f"Error computing discrete Gaussian PDF: {e}")
        raise RuntimeError(f"Failed to compute discrete Gaussian PDF: {e}") from e


def precompute_discrete_gaussian_probabilities(sigma: float,
                                              center: Union[int, float] = 0,
                                              radius: float = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute discrete Gaussian probabilities for integers within radius*sigma of center.
    
    Mathematical Formula:
    ρ_σ,c(x) = exp(-||x - c||² / (2σ²))
    
    Normalized to ensure the sum of probabilities equals 1.
    
    Cryptographic Relevance:
    Precomputing these probabilities is essential for efficient sampling in:
    - Lattice-based signature schemes
    - Encryption schemes requiring high-performance Gaussian sampling
    - Security-critical applications where constant-time operations are needed
    
    Assumptions:
    - The radius parameter is assumed to cover the significant mass of the distribution. 
      Typically, 6σ is sufficient to capture >99.999% of the distribution's mass.
    - For cryptographic applications, larger radius values may be needed to ensure
      statistical indistinguishability in security proofs.
    
    Args:
        sigma: The standard deviation of the Gaussian
        center: The center of the Gaussian (default: 0)
        radius: How many standard deviations to consider (default: 6)
        
    Returns:
        A tuple containing:
        - Array of integer points
        - Array of corresponding probabilities (normalized to sum to 1)
        
    Raises:
        ValueError: If sigma or radius is not positive, or if total probability mass is zero
    """
    # Input validation
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    if radius <= 0:
        raise ValueError("Radius must be positive")
    
    try:
        # Compute bounds
        lower_bound = int(floor(center - radius * sigma))
        upper_bound = int(ceil(center + radius * sigma))
        
        # Create arrays for points and probabilities
        points = np.arange(lower_bound, upper_bound + 1)
        probs = np.zeros(len(points), dtype=np.float64)
        
        # Calculate probabilities using vectorized operations where possible
        for i, x in enumerate(points):
            try:
                probs[i] = discrete_gaussian_pdf(x, sigma, center)
            except Exception as e:
                logger.warning(f"Error computing probability for x={x}: {e}, using fallback")
                # Fallback to standard formula with safeguards
                diff_squared = (x - center) ** 2
                if diff_squared > 700 * sigma * sigma:
                    probs[i] = 1e-300
                else:
                    probs[i] = exp(-diff_squared / (2 * sigma * sigma))
        
        # Check for underflow
        total = np.sum(probs)
        if total <= 0 or not np.isfinite(total):
            logger.error("Total probability mass is zero or invalid due to numerical underflow")
            raise ValueError("Total probability mass is zero or invalid due to numerical underflow; "
                            "increase sigma or use extended precision")
        
        # Normalize probabilities in a single operation
        probs = probs / total
        
        # Final validation
        if not np.all(np.isfinite(probs)):
            logger.warning("Non-finite values detected in normalized probabilities")
            # Replace non-finite values with zeros and renormalize
            probs[~np.isfinite(probs)] = 0
            new_total = np.sum(probs)
            if new_total > 0:
                probs = probs / new_total
            else:
                raise ValueError("Cannot recover from non-finite probability values")
        
        return points, probs
        
    except Exception as e:
        logger.error(f"Error precomputing discrete Gaussian probabilities: {e}")
        raise RuntimeError(f"Failed to precompute discrete Gaussian probabilities: {e}") from e


def calculate_smoothing_parameter(dim: int, epsilon: float = 0.01) -> float:
    """
    Calculate the smoothing parameter for a lattice.
    
    Mathematical Context:
    The smoothing parameter η_ε(Λ) is a fundamental concept in lattice-based cryptography.
    It represents the Gaussian parameter at which the discrete Gaussian distribution over
    the dual lattice Λ* appears nearly uniform modulo the lattice.
    
    For an identity basis, the smoothing parameter is approximated as:
    η_ε(Λ) ≈ sqrt(ln(2n/ε)/π)
    
    Where:
    - n is the lattice dimension
    - ε is a small constant, typically 2^(-n)
    
    Cryptographic Relevance:
    In lattice-based cryptography, sampling with σ > η_ε(Λ) ensures statistical
    properties required for security proofs in schemes like:
    - Ring-LWE encryption
    - Lattice-based signature schemes
    - Trapdoor functions
    
    Args:
        dim: Dimension of the lattice
        epsilon: Small constant (default: 0.01)
        
    Returns:
        The approximate smoothing parameter
    """
    if dim <= 0:
        raise ValueError("Dimension must be positive")
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("Epsilon must be between 0 and 1")
    
    try:
        # Use logarithm for numerical stability
        return sqrt(log(2*dim/epsilon)/pi)
    except Exception as e:
        logger.error(f"Error calculating smoothing parameter: {e}")
        raise RuntimeError(f"Failed to calculate smoothing parameter: {e}") from e


def is_positive_definite(matrix: Matrix) -> bool:
    """
    Check if a matrix is positive definite.
    
    A symmetric matrix is positive definite if all its eigenvalues are positive.
    This is important for covariance matrices in multivariate Gaussian distributions.
    
    Args:
        matrix: A symmetric matrix to check
        
    Returns:
        True if the matrix is positive definite, False otherwise
    """
    try:
        # Check if matrix is symmetric
        if not matrix.is_symmetric():
            return False
        
        # Check if all eigenvalues are positive
        eigenvalues = matrix.eigenvalues()
        return all(eigval > 0 for eigval in eigenvalues)
    except Exception as e:
        logger.error(f"Error checking positive definiteness: {e}")
        return False
