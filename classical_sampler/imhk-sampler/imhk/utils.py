from sage.structure.element import Vector
import numpy as np
from sage.all import *
from typing import Union, Tuple, Optional
from math import floor, ceil, exp


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
        x = vector(RR, x)
    
    if isinstance(center, (list, tuple, np.ndarray)):
        center = vector(RR, center)
    
    # Check dimensions match if both are vectors
    if isinstance(x, Vector) and isinstance(center, Vector) and len(x) != len(center):
        raise ValueError(f"Dimension mismatch: x has dimension {len(x)}, center has dimension {len(center)}")
    
    if isinstance(x, Vector):
        # Compute the squared norm of (x - center) using dot product for numerical stability
        diff = vector(RR, x) - vector(RR, center)
        squared_norm = diff.dot_product(diff)
        return exp(-squared_norm / (2 * sigma * sigma))
    else:
        # Scalar case
        return exp(-(x - center) ** 2 / (2 * sigma * sigma))


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
    
    # Compute bounds
    lower_bound = int(floor(center - radius * sigma))
    upper_bound = int(ceil(center + radius * sigma))
    
    # Create arrays for points and probabilities
    points = np.arange(lower_bound, upper_bound + 1)
    probs = np.zeros(len(points), dtype=np.float64)
    
    # Calculate probabilities using vectorized operations where possible
    for i, x in enumerate(points):
        probs[i] = discrete_gaussian_pdf(x, sigma, center)
    
    # Check for underflow
    total = np.sum(probs)
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Total probability mass is zero or invalid due to numerical underflow; "
                         "increase sigma or use extended precision")
    
    # Normalize probabilities in a single operation
    probs = probs / total
    
    return points, probs