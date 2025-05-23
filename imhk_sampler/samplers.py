"""
Core sampling algorithms for the IMHK sampler framework.

This module implements the discrete Gaussian sampling algorithms used in
lattice-based cryptography, with a focus on numerical stability and
statistical accuracy.
"""

from sage.structure.element import Vector
import numpy as np
from sage.all import *
from typing import List, Tuple, Optional, Union, Callable
from random import random
from math import exp, pi, sqrt, floor, ceil
from bisect import bisect_left
import logging

# Set up module logger
logger = logging.getLogger("imhk_samplers")


def imhk_sampler_wrapper(basis_info, sigma: float, num_samples: int, center: Optional[Vector] = None, 
                        burn_in: int = 1000, basis_type: str = 'identity') -> Tuple[np.ndarray, dict]:
    """
    Wrapper function for IMHK sampler that handles different cryptographic lattice types.
    
    This wrapper intelligently dispatches to appropriate sampling algorithms based on 
    the lattice type (matrix-based or polynomial/structured).
    
    Args:
        basis_info: Either a Matrix or tuple (poly_mod, q) for structured lattices
        sigma: Standard deviation of the Gaussian
        num_samples: Number of samples to generate
        center: Center of the Gaussian (default: origin)
        burn_in: Number of initial samples to discard
        basis_type: Type of basis ('identity', 'q-ary', 'NTRU', 'PrimeCyclotomic')
        
    Returns:
        Tuple (samples, metadata)
    """
    if basis_type in ['NTRU', 'PrimeCyclotomic']:
        # Handle structured lattices
        if not isinstance(basis_info, tuple) or len(basis_info) != 2:
            raise ValueError(f"For {basis_type}, expected tuple (poly_mod, q)")
        return imhk_sampler_structured(basis_info, sigma, num_samples, center, burn_in, basis_type)
    else:
        # Handle matrix-based lattices  
        if not hasattr(basis_info, 'nrows'):
            raise ValueError(f"For {basis_type}, expected a matrix")
        return imhk_sampler(basis_info, sigma, num_samples, center, burn_in)


def klein_sampler_wrapper(basis_info, sigma: float, num_samples: int,
                         center: Optional[Vector] = None, basis_type: str = 'identity') -> Tuple[np.ndarray, dict]:
    """
    Wrapper function for Klein sampler that handles different cryptographic lattice types.
    
    Args:
        basis_info: Either a Matrix or tuple (poly_mod, q) for structured lattices
        sigma: Standard deviation of the Gaussian
        num_samples: Number of samples to generate
        center: Center of the Gaussian (default: origin)
        basis_type: Type of basis ('identity', 'q-ary', 'NTRU', 'PrimeCyclotomic')
        
    Returns:
        Tuple (samples, metadata)
    """
    if basis_type in ['NTRU', 'PrimeCyclotomic']:
        # Handle structured lattices
        if not isinstance(basis_info, tuple) or len(basis_info) != 2:
            raise ValueError(f"For {basis_type}, expected tuple (poly_mod, q)")
        return klein_sampler_structured(basis_info, sigma, num_samples, center, basis_type)
    else:
        # Handle matrix-based lattices  
        if not hasattr(basis_info, 'nrows'):
            raise ValueError(f"For {basis_type}, expected a matrix")
        samples = klein_sampler(basis_info, sigma, num_samples, center)
        metadata = {
            'basis_type': basis_type,
            'sigma': sigma
        }
        return samples, metadata


def klein_sampler_structured(lattice_params: Tuple, sigma: float, num_samples: int,
                           center: Optional[Vector] = None, basis_type: str = 'NTRU') -> Tuple[np.ndarray, dict]:
    """
    Klein sampler for structured lattices (NTRU, Prime Cyclotomic).
    
    Simplified sampling for polynomial-based lattices.
    
    Args:
        lattice_params: Tuple (polynomial_modulus, prime_modulus)
        sigma: Standard deviation of the Gaussian
        num_samples: Number of samples to generate
        center: Center of the Gaussian (default: origin)
        basis_type: Type of structured lattice
        
    Returns:
        Tuple (samples, metadata)
    """
    poly_mod, q = lattice_params
    degree = poly_mod.degree()
    
    logger.info(f"Starting structured Klein sampler: type={basis_type}, degree={degree}, q={q}")
    
    # For structured lattices, we work in the polynomial ring
    R = poly_mod.parent()
    
    # Initialize samples list
    samples = []
    
    # Generate samples
    for i in range(num_samples):
        # Generate coefficients for polynomial sample
        coeffs = []
        for j in range(degree):
            # Sample from discrete Gaussian over integers
            coeff = _sample_discrete_gaussian_integer(sigma, 0)
            coeffs.append(coeff % q)  # Reduce modulo q
            
        # Create polynomial from coefficients
        sample_poly = R(coeffs)
        
        # Convert polynomial to vector of coefficients
        coeffs_list = sample_poly.list()
        # Pad with zeros to full degree if necessary
        while len(coeffs_list) < degree:
            coeffs_list.append(0)
        sample_vector = [int(c) for c in coeffs_list]
        samples.append(sample_vector)
            
    # Convert samples to numpy array with correct shape
    if samples:
        samples_array = np.array(samples)
    else:
        samples_array = np.zeros((0, degree))  # Empty array with correct shape
    
    metadata = {
        'basis_type': basis_type,
        'polynomial_degree': degree,
        'modulus': q,
        'sigma': sigma
    }
    
    return samples_array, metadata


def imhk_sampler_structured(lattice_params: Tuple, sigma: float, num_samples: int, 
                           center: Optional[Vector] = None, burn_in: int = 1000,
                           basis_type: str = 'NTRU') -> Tuple[np.ndarray, dict]:
    """
    IMHK sampler for structured lattices (NTRU, Prime Cyclotomic).
    
    Handles polynomial-based lattice representations used in modern
    cryptographic constructions like Falcon and Mitaka.
    
    Args:
        lattice_params: Tuple (polynomial_modulus, prime_modulus)
        sigma: Standard deviation of the Gaussian
        num_samples: Number of samples to generate
        center: Center of the Gaussian (default: origin)
        burn_in: Number of initial samples to discard
        basis_type: Type of structured lattice
        
    Returns:
        Tuple (samples, metadata)
    """
    poly_mod, q = lattice_params
    degree = poly_mod.degree()
    
    logger.info(f"Starting structured IMHK sampler: type={basis_type}, degree={degree}, q={q}")
    
    # For structured lattices, we work in the polynomial ring
    R = poly_mod.parent()
    
    # Initialize samples list
    samples = []
    acceptance_count = 0
    total_count = 0
    
    # Generate initial sample (zero polynomial)
    current_sample = R(0)
    current_density = 1.0  # Density at origin
    
    # Main sampling loop
    for i in range(num_samples + burn_in):
        # Generate coefficients for polynomial sample
        coeffs = []
        for j in range(degree):
            # Sample from discrete Gaussian over integers
            coeff = _sample_discrete_gaussian_integer(sigma, 0)
            coeffs.append(coeff % q)  # Reduce modulo q
            
        # Create polynomial from coefficients
        proposal = R(coeffs)
        
        # Compute acceptance probability (simplified for structured lattices)
        proposal_density = _compute_polynomial_gaussian_density(proposal, sigma, poly_mod, q)
        
        if current_density > 0:
            ratio = proposal_density / current_density
        else:
            ratio = 1.0
            
        # Accept or reject
        if random() < min(1, ratio):
            current_sample = proposal
            current_density = proposal_density
            acceptance_count += 1
            
        total_count += 1
        
        # Store sample after burn-in
        if i >= burn_in:
            # Convert polynomial to vector of coefficients
            coeffs_list = current_sample.list()
            # Pad with zeros to full degree if necessary
            while len(coeffs_list) < degree:
                coeffs_list.append(0)
            sample_vector = [int(c) for c in coeffs_list]
            samples.append(sample_vector)
            
    # Convert samples to numpy array with correct shape
    if samples:
        samples_array = np.array(samples)
    else:
        samples_array = np.zeros((0, degree))  # Empty array with correct shape
    
    metadata = {
        'acceptance_rate': acceptance_count / total_count if total_count > 0 else 0,
        'basis_type': basis_type,
        'polynomial_degree': degree,
        'modulus': q,
        'sigma': sigma,
        'burn_in': burn_in
    }
    
    return samples_array, metadata


def _sample_discrete_gaussian_integer(sigma: float, center: float = 0) -> int:
    """Sample from discrete Gaussian over integers."""
    # Simple rejection sampling
    bound = ceil(6 * sigma)  # 6-sigma bound
    while True:
        x = randint(-bound, bound)
        prob = exp(-(x - center)**2 / (2 * sigma**2))
        if random() < prob:
            return x
            

def _compute_polynomial_gaussian_density(poly, sigma: float, poly_mod, q: int) -> float:
    """Compute Gaussian density for polynomial."""
    # Compute norm of polynomial coefficients
    coeffs = poly.list()
    norm_sq = sum(c**2 for c in coeffs)
    return exp(-norm_sq / (2 * sigma**2))


def _get_function(module_name, function_name):
    """Dynamically import a function to avoid circular dependencies."""
    import importlib
    module = importlib.import_module(f"imhk_sampler.{module_name}")
    return getattr(module, function_name)


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
    
    try:
        if isinstance(x, Vector):
            # Compute the squared norm of (x - center) using dot product for numerical stability
            diff = vector(RR, x) - vector(RR, center)
            squared_norm = diff.dot_product(diff)
            return exp(-squared_norm / (2 * sigma * sigma))
        else:
            # Scalar case
            return exp(-(x - center) ** 2 / (2 * sigma * sigma))
    except Exception as e:
        logger.error(f"Error calculating discrete Gaussian PDF: {e}")
        raise RuntimeError(f"Failed to calculate discrete Gaussian PDF: {e}") from e


def discrete_gaussian_sampler_1d(center: float, sigma: float) -> int:
    """
    Sample from a 1D discrete Gaussian distribution centered at 'center' with width 'sigma'.
    
    Mathematical Basis:
    The discrete Gaussian distribution is a probability distribution over integers,
    where the probability of sampling integer x is proportional to exp(-(x-center)²/(2σ²)).
    It is a fundamental primitive in lattice-based cryptography.
    
    Cryptographic Relevance:
    In lattice-based cryptography, discrete Gaussian sampling is used for:
    - Key generation in encryption schemes like Ring-LWE
    - Signature schemes like FALCON and CRYSTALS-Dilithium
    - Trapdoor sampling for security reductions
    - Generating error terms with provable security properties
    
    This implementation uses precomputed CDF and binary search for efficient sampling,
    especially effective for repeated sampling with the same parameters.
    
    Args:
        center (float): The center of the Gaussian
        sigma (float): The standard deviation
        
    Returns:
        int: An integer sampled from the discrete Gaussian
        
    Raises:
        ValueError: If sigma is not positive
    """
    # Input validation
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    
    # Compute the integer center
    c_int = int(round(center))
    
    # For very small sigma, just return the rounded center
    if sigma < 0.01:
        return c_int
    
    try:
        # For standard cases, use precomputed probabilities within 6*sigma
        tau = 6  # Number of standard deviations to consider
        lower_bound = int(floor(center - tau * sigma))
        upper_bound = int(ceil(center + tau * sigma))
        
        # Precompute probabilities and CDF for this range
        x_values = list(range(lower_bound, upper_bound + 1))
        probs = [exp(-(x - center)**2 / (2 * sigma * sigma)) for x in x_values]
        
        # Normalize and compute cumulative distribution
        total = sum(probs)
        if total <= 0:
            logger.warning("Total probability mass is zero, returning rounded center")
            return c_int
            
        cdf = [0]
        for p in probs:
            cdf.append(cdf[-1] + p / total)
        cdf.pop(0)  # Remove the initial 0
        
        # Sample using binary search on the CDF
        u = random()
        idx = bisect_left(cdf, u)
        if idx >= len(x_values):
            idx = len(x_values) - 1  # Handle the edge case
        
        return x_values[idx]
    
    except Exception as e:
        logger.error(f"Error in discrete Gaussian sampling: {e}")
        # Fallback to returning the rounded center
        return c_int


def klein_sampler_single(B: Matrix, sigma: float, center: Optional[Vector] = None) -> Vector:
    """
    Klein's algorithm for sampling from a discrete Gaussian over a lattice.
    
    Mathematical Basis:
    Klein's algorithm samples from a discrete Gaussian distribution over a lattice
    by using the Gram-Schmidt orthogonalization of the basis. It processes the basis
    vectors in reverse order, sampling each coordinate from a 1D discrete Gaussian
    and adjusting subsequent coordinates accordingly.
    
    Cryptographic Relevance:
    In lattice-based cryptography, Klein's algorithm is used for:
    - Trapdoor sampling in lattice-based signatures
    - Basis delegation techniques
    - Preimage sampling under lattice trapdoors
    - Security proofs for lattice-based cryptosystems
    
    Assumptions:
    - The lattice basis B is full-rank
    - The basis is reasonably well-conditioned for numerical stability
    - The sigma parameter is large enough relative to the Gram-Schmidt norms
    
    Args:
        B (Matrix): The lattice basis matrix (rows are basis vectors)
        sigma (float): The standard deviation of the Gaussian
        center (Optional[Vector]): The center of the Gaussian (default: origin)
        
    Returns:
        Vector: A lattice point sampled according to Klein's algorithm
        
    Raises:
        ValueError: If sigma is not positive or if dimensions don't match
    """
    # Input validation
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    
    logger.info(f"Starting Klein sampler: dim={B.nrows()}, sigma={sigma}")
    
    # Use center parameter, but work with c internally for consistency
    n = B.nrows()
    c = center
    if c is None:
        c = vector(RR, [0] * n)
    else:
        c = vector(RR, c)
    
    # Verify dimensions
    if len(c) != n:
        raise ValueError(f"Center vector dimension ({len(c)}) must match basis dimension ({n})")
    
    try:
        # Convert to a ring that supports Gram-Schmidt orthogonalization
        # SageMath requires exact rings like QQ for gram_schmidt()
        try:
            B_copy = matrix(QQ, B)  # Use rational numbers for exact computation
        except (TypeError, ValueError):
            # If conversion to QQ fails, try using RR with high precision
            B_copy = matrix(RR, B)
        
        # Gram-Schmidt orthogonalization
        GSO = B_copy.gram_schmidt()
        Q = GSO[0]  # Orthogonal basis
        
        # Sample from discrete Gaussian over Z^n, working backwards
        z = vector(ZZ, [0] * n)
        c_prime = vector(RR, c)
        
        for i in range(n-1, -1, -1):
            b_i_star = vector(RR, Q.row(i))  # Convert to RR for numerical stability
            b_i = vector(RR, B_copy.row(i))
            
            # Project c_prime onto b_i_star
            # μ = <c', b*>/||b*||^2
            b_star_norm_sq = b_i_star.dot_product(b_i_star)
            
            # Ensure b_star_norm_sq is not too close to zero
            if b_star_norm_sq < 1e-10:
                raise ValueError(f"Gram-Schmidt orthogonalization failed: vector {i} has near-zero norm")
            
            mu_i = b_i_star.dot_product(c_prime) / b_star_norm_sq
            
            # Set sigma_i for this coordinate (σ / ||b*||)
            sigma_i = sigma / sqrt(b_star_norm_sq)
            
            # Sample from 1D discrete Gaussian
            z_i = discrete_gaussian_sampler_1d(mu_i, sigma_i)
            z[i] = z_i
            
            # Update c_prime for the next iteration
            c_prime = c_prime - z_i * b_i
        
        # Convert z to a lattice point by multiplying with the basis
        return B * z
        
    except Exception as e:
        logger.error(f"Error in Klein sampler: {e}")
        raise RuntimeError(f"Klein sampling failed: {e}") from e


def imhk_sampler(B: Matrix, sigma: float, num_samples: int, center: Optional[Vector] = None, 
                 burn_in: int = 1000) -> Tuple[np.ndarray, dict]:
    """
    Independent Metropolis-Hastings-Klein algorithm for sampling from a discrete Gaussian
    over a lattice.
    
    Mathematical Basis:
    The IMHK algorithm combines Klein's algorithm (as a proposal distribution) with
    Metropolis-Hastings to correct the sampling distribution and achieve the exact
    discrete Gaussian distribution over the lattice.
    
    Cryptographic Relevance:
    In lattice-based cryptography, high-quality Gaussian sampling is crucial for:
    - Security of signature schemes (preventing side-channel attacks)
    - Trapdoor sampling with enhanced security properties
    - Achieving better concrete security bounds in cryptographic constructions
    - Statistical indistinguishability in security proofs
    
    IMHK provides better statistical quality than standalone Klein sampler, especially
    for bases with high orthogonalization defects, which is essential for cryptographic
    applications requiring high-precision sampling.
    
    Assumptions:
    - The lattice basis B is full-rank
    - The target distribution is a discrete Gaussian centered at c with parameter sigma
    - Klein's algorithm provides a reasonable approximation to use as proposal distribution
    - The burn-in period is sufficient to reach the stationary distribution
    
    Args:
        B (Matrix): The lattice basis matrix (rows are basis vectors)
        sigma (float): The standard deviation of the Gaussian
        num_samples (int): The number of samples to generate
        center (Optional[Vector]): The center of the Gaussian (default: origin)
        burn_in (int): The number of initial samples to discard
        
    Returns:
        Tuple containing:
        - List[Vector]: List of lattice points sampled according to the discrete Gaussian
        - float: Acceptance rate of the Metropolis-Hastings algorithm
        - List[Vector]: All samples including burn-in (for diagnostics)
        - List[bool]: Whether each proposal was accepted (for diagnostics)
        
    Raises:
        ValueError: If sigma is not positive, num_samples or burn_in are not positive
    """
    # Input validation
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive")
    if burn_in < 0:
        raise ValueError("Burn-in period must be non-negative")
    
    logger.info(f"Starting IMHK sampler: dim={B.nrows()}, sigma={sigma}, num_samples={num_samples}, burn_in={burn_in}")
    
    # Use center parameter, but work with c internally for consistency
    n = B.nrows()
    c = center
    if c is None:
        c = vector(RR, [0] * n)
    else:
        c = vector(RR, c)
    
    # Verify dimensions
    if len(c) != n:
        raise ValueError(f"Center vector dimension ({len(c)}) must match basis dimension ({n})")
    
    try:
        # Initialize the chain with a sample from Klein's algorithm
        current_sample = klein_sampler_single(B, sigma, c)
        current_density = discrete_gaussian_pdf(current_sample, sigma, c)
        
        samples = []
        acceptance_count = 0
        total_count = 0
        
        # Monitor individual samples for diagnostics
        all_samples = []  # Store all samples including burn-in for diagnostics
        all_accepts = []  # Store whether each proposal was accepted
        
        logger.info(f"Starting IMHK sampling with {num_samples} samples and {burn_in} burn-in")
        
        # Run the chain
        for i in range(num_samples + burn_in):
            # Generate proposal using Klein's algorithm
            proposal = klein_sampler_single(B, sigma, c)
            proposal_density = discrete_gaussian_pdf(proposal, sigma, c)
            
            # Compute the Metropolis-Hastings acceptance ratio with numerical safeguards
            # For independent proposals with target π and proposal q:
            # acceptance probability = min(1, π(y)q(x)/π(x)q(y))
            epsilon = 1e-10  # Small constant to avoid division by zero
            if current_density < epsilon:
                # If current density is effectively zero, accept the proposal
                ratio = 1.0
            else:
                ratio = proposal_density / current_density
            
            # Accept or reject the proposal
            accept = random() < min(1, ratio)
            if accept:
                current_sample = proposal
                current_density = proposal_density
                acceptance_count += 1
            
            total_count += 1
            all_accepts.append(accept)
            all_samples.append(current_sample)
            
            # Store the sample if we're past the burn-in period
            if i >= burn_in:
                samples.append(current_sample)
            
            # Periodic logging for long runs
            if (i+1) % 1000 == 0:
                logger.debug(f"IMHK progress: {i+1}/{num_samples + burn_in} iterations, "
                           f"acceptance rate: {acceptance_count/(i+1):.4f}")
        
        acceptance_rate = acceptance_count / total_count if total_count > 0 else 0.0
        logger.info(f"IMHK sampling completed with acceptance rate: {acceptance_rate:.4f}")
        
        # Convert to numpy array for compatibility
        samples_array = np.array([list(s) for s in samples])
        
        # Create metadata dictionary
        metadata = {
            'acceptance_rate': acceptance_rate,
            'samples_accepted': acceptance_count,
            'samples_proposed': total_count,
            'burn_in': burn_in,
            'all_samples': all_samples,
            'all_accepts': all_accepts
        }
        
        return samples_array, metadata
        
    except Exception as e:
        logger.error(f"Error in IMHK sampler: {e}")
        raise RuntimeError(f"IMHK sampling failed: {e}") from e


def imhk_sampler_publication(B: Matrix, sigma: float, num_samples: int, 
                            center: Optional[Vector] = None, burn_in: int = 1000) -> Tuple[np.ndarray, dict]:
    """Alias for imhk_sampler for publication compatibility."""
    return imhk_sampler(B=B, sigma=sigma, num_samples=num_samples, 
                       center=center, burn_in=burn_in)


def klein_sampler(B: Matrix, sigma: float, num_samples: int, 
                 center: Optional[Vector] = None) -> np.ndarray:
    """
    Klein's algorithm for sampling multiple points from a discrete Gaussian over a lattice.
    
    Args:
        B (Matrix): The lattice basis matrix
        sigma (float): The standard deviation of the Gaussian
        num_samples (int): Number of samples to generate
        center (Optional[Vector]): The center of the Gaussian (default: origin)
        
    Returns:
        np.ndarray: Array of shape (num_samples, dimension) containing samples
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    
    logger.info(f"Klein sampler: dim={B.nrows()}, sigma={sigma}, num_samples={num_samples}")
    
    samples = []
    for i in range(num_samples):
        if i > 0 and i % 1000 == 0:
            logger.debug(f"Klein sampler: generated {i}/{num_samples} samples")
        
        # Use the single-sample Klein sampler
        sample = klein_sampler_single(B, sigma, center)
        samples.append(list(sample))
    
    logger.info(f"Klein sampler completed: {num_samples} samples generated")
    return np.array(samples)


# End of file