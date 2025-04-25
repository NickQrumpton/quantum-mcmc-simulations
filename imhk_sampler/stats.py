from sage.structure.element import Vector
import numpy as np
from sage.all import *
from typing import List, Union, Optional, Tuple, Any
from math import log, pi, sqrt, ceil
from itertools import product
import random


def _compute_normalization_constant(lattice_basis: Matrix, sigma: float, center: Vector, 
                                   radius: float = 10.0, discrete_gaussian_pdf=None) -> float:
    """
    Compute a better approximation of the normalization constant Z for the discrete Gaussian.
    
    Uses systematic enumeration of lattice points within radius*sigma of the center,
    with adaptations for high-dimensional lattices.
    
    Args:
        lattice_basis: The lattice basis
        sigma: The standard deviation of the Gaussian
        center: Center of the Gaussian
        radius: Radius in standard deviations to consider (default: 10.0)
        discrete_gaussian_pdf: Function to compute PDF (if None, will be imported)
        
    Returns:
        Approximation of the normalization constant Z
    """
    if discrete_gaussian_pdf is None:
        from utils import discrete_gaussian_pdf
    
    n_dims = lattice_basis.nrows()
    
    # For very high dimensions, use theoretical approximation
    if n_dims > 20:
        # Z ≈ (det(Λ))^(-1) * (2πσ²)^(n/2)
        det_lattice = abs(lattice_basis.determinant())
        Z_approx = (det_lattice**(-1)) * ((2 * pi * sigma**2)**(n_dims/2))
        return Z_approx
    
    # Compute the Gram-Schmidt orthogonalization to determine enumeration bounds
    GSO = lattice_basis.gram_schmidt()[0]
    
    # Find the shortest vector in the GSO
    min_gs_norm = min(sqrt(sum(v_i^2 for v_i in v)) for v in GSO)
    
    # Determine the enumeration bounds based on sigma and GSO
    enum_radius = int(ceil(radius * sigma / min_gs_norm))
    
    # Limit radius for higher dimensions to avoid combinatorial explosion
    if n_dims > 10:
        enum_radius = min(enum_radius, 3)
    
    # Generate lattice points: sample randomly for high dimensions, enumerate for low dimensions
    coordinates_list = []
    if n_dims > 15:
        # Random sampling for high dimensions
        num_samples = 50000
        for _ in range(num_samples):
            coords = tuple(random.randint(-enum_radius, enum_radius) for _ in range(n_dims))
            coordinates_list.append(coords)
    else:
        # Enumeration for lower dimensions
        coordinates_list = list(product(range(-enum_radius, enum_radius+1), repeat=n_dims))
    
    # Compute the unnormalized probability sum
    Z = 0.0
    for coords in coordinates_list:
        # Convert integer coordinates to a lattice point
        point = center
        for i, c in enumerate(coords):
            if c != 0:  # Skip zero coefficients for efficiency
                point = point + c * vector(lattice_basis.row(i))
        
        # Add the unnormalized probability
        Z += discrete_gaussian_pdf(point, sigma, center)
    
    # If Z is too small (numerical underflow), fall back to theoretical approximation
    if Z < 1e-10:
        det_lattice = abs(lattice_basis.determinant())
        Z = (det_lattice**(-1)) * ((2 * pi * sigma**2)**(n_dims/2))
    
    return Z


def compute_total_variation_distance(samples: List[Vector], 
                                    sigma: float, 
                                    lattice_basis: Matrix, 
                                    center: Optional[Vector] = None,
                                    discrete_gaussian_pdf=None) -> float:
    """
    Compute the total variation distance between the empirical distribution of samples
    and the ideal discrete Gaussian distribution over a lattice.
    
    Mathematical Definition:
    TV(μ, ν) = (1/2) * sum_x |μ(x) - ν(x)|
    
    Where:
    - μ is the empirical distribution
    - ν is the theoretical discrete Gaussian distribution
    
    Relevance to Lattice-based Cryptography:
    The total variation distance measures how close the sampler's output is to the
    ideal discrete Gaussian distribution, which is critical for security proofs in
    lattice-based cryptographic schemes. Smaller distances indicate better sampling quality,
    which directly impacts the security guarantees of lattice-based encryption and
    signature schemes like FALCON, CRYSTALS-Dilithium, and NTRU.
    
    Args:
        samples: List of lattice point samples (SageMath vectors)
        sigma: The standard deviation of the Gaussian (positive real)
        lattice_basis: The lattice basis (full-rank SageMath matrix)
        center: Center of the Gaussian distribution (default: origin)
        discrete_gaussian_pdf: Function to compute PDF (if None, will be imported)
        
    Returns:
        The estimated total variation distance (float between 0 and 1)
        
    Assumptions:
    - All samples are points in the lattice defined by lattice_basis
    - The lattice basis is full-rank
    - The discrete Gaussian is centered at 'center' with standard deviation 'sigma'
    
    Raises:
        ValueError: If samples is empty, sigma is not positive, or lattice_basis is not full-rank
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    if not lattice_basis:
        raise ValueError("Lattice basis must be provided")
    
    # Check lattice basis is full-rank
    n_dims = lattice_basis.nrows()
    if lattice_basis.rank() < n_dims:
        raise ValueError("Lattice basis must be full-rank")
    
    # Import discrete_gaussian_pdf if not provided
    if discrete_gaussian_pdf is None:
        from utils import discrete_gaussian_pdf
    
    # Set default center if not provided
    if center is None:
        center = vector(RR, [0] * n_dims)
    else:
        # Verify center dimension matches samples
        if len(center) != n_dims:
            raise ValueError(f"Center dimension ({len(center)}) must match lattice dimension ({n_dims})")
    
    # Compute the empirical distribution using NumPy arrays for efficiency
    # Convert samples to hashable tuples for counting
    sample_tuples = np.array([tuple(float(x_i) for x_i in x) for x in samples])
    
    # Use NumPy's unique function to efficiently count occurrences
    unique_points, counts = np.unique(sample_tuples, axis=0, return_counts=True)
    
    # Normalize to get empirical probabilities
    total_samples = len(samples)
    empirical_probs = counts / total_samples
    
    # Convert unique points to SageMath vectors for theoretical probability computation
    unique_vectors = [vector(RR, point) for point in unique_points]
    
    # Compute improved normalization constant Z using the lattice structure
    Z = _compute_normalization_constant(lattice_basis, sigma, center, 
                                       radius=10.0, discrete_gaussian_pdf=discrete_gaussian_pdf)
    
    # Compute theoretical probabilities for observed points
    theoretical_densities = np.array([discrete_gaussian_pdf(point, sigma, center) 
                                     for point in unique_vectors])
    theoretical_probs = theoretical_densities / Z
    
    # Compute the total variation distance: TV = (1/2) * sum(|empirical - theoretical|)
    tv_distance = 0.5 * np.sum(np.abs(empirical_probs - theoretical_probs))
    
    return tv_distance


def compute_kl_divergence(samples: List[Vector], 
                         sigma: float, 
                         lattice_basis: Matrix, 
                         center: Optional[Vector] = None,
                         discrete_gaussian_pdf=None) -> float:
    """
    Compute the Kullback-Leibler divergence between the empirical distribution of samples
    and the ideal discrete Gaussian distribution over a lattice.
    
    Mathematical Definition:
    KL(μ||ν) = sum_x μ(x) * log(μ(x)/ν(x))
    
    Where:
    - μ is the empirical distribution
    - ν is the theoretical discrete Gaussian distribution
    
    Relevance to Lattice-based Cryptography:
    KL divergence quantifies the information-theoretic difference between the empirical 
    and ideal distributions. In lattice-based cryptography, this metric:
    - Helps evaluate the security of sampling algorithms used in trapdoor constructions
    - Provides a measure for the statistical quality of noise generation in encryption schemes
    - Validates sampling procedures in formal security proofs for LWE/Ring-LWE constructions
    - Assesses potential vulnerabilities to statistical attacks on the sampling process
    
    Args:
        samples: List of lattice point samples (SageMath vectors)
        sigma: The standard deviation of the Gaussian (positive real)
        lattice_basis: The lattice basis (full-rank SageMath matrix)
        center: Center of the Gaussian distribution (default: origin)
        discrete_gaussian_pdf: Function to compute PDF (if None, will be imported)
        
    Returns:
        The estimated KL divergence (non-negative real)
        
    Assumptions:
    - All samples are points in the lattice defined by lattice_basis
    - The lattice basis is full-rank
    - The discrete Gaussian is centered at 'center' with standard deviation 'sigma'
    
    Raises:
        ValueError: If samples is empty, sigma is not positive, or lattice_basis is not full-rank
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    if not lattice_basis:
        raise ValueError("Lattice basis must be provided")
    
    # Check lattice basis is full-rank
    n_dims = lattice_basis.nrows()
    if lattice_basis.rank() < n_dims:
        raise ValueError("Lattice basis must be full-rank")
    
    # Import discrete_gaussian_pdf if not provided
    if discrete_gaussian_pdf is None:
        from utils import discrete_gaussian_pdf
    
    # Set default center if not provided
    if center is None:
        center = vector(RR, [0] * n_dims)
    else:
        # Verify center dimension matches samples
        if len(center) != n_dims:
            raise ValueError(f"Center dimension ({len(center)}) must match lattice dimension ({n_dims})")
    
    # Compute the empirical distribution using NumPy arrays for efficiency
    sample_tuples = np.array([tuple(float(x_i) for x_i in x) for x in samples])
    unique_points, counts = np.unique(sample_tuples, axis=0, return_counts=True)
    
    # Normalize to get empirical probabilities
    total_samples = len(samples)
    empirical_probs = counts / total_samples
    
    # Convert unique points to SageMath vectors for theoretical probability computation
    unique_vectors = [vector(RR, point) for point in unique_points]
    
    # Compute improved normalization constant Z using the lattice structure
    Z = _compute_normalization_constant(lattice_basis, sigma, center, 
                                       radius=10.0, discrete_gaussian_pdf=discrete_gaussian_pdf)
    
    # Compute theoretical probabilities for observed points
    theoretical_densities = np.array([discrete_gaussian_pdf(point, sigma, center) 
                                     for point in unique_vectors])
    theoretical_probs = theoretical_densities / Z
    
    # Add small epsilon to prevent log(0) - important for numerical stability
    epsilon = 1e-10
    theoretical_probs = np.maximum(theoretical_probs, epsilon)
    
    # Compute the KL divergence: KL = sum(empirical * log(empirical / theoretical))
    # Use log from NumPy for vectorized operation
    log_ratio = np.log(empirical_probs / theoretical_probs)
    kl_terms = empirical_probs * log_ratio
    kl_divergence = np.sum(kl_terms)
    
    return max(0.0, kl_divergence)  # Ensure non-negative due to potential numerical issues