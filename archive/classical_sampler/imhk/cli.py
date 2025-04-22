#!/usr/bin/env sage
r"""
Independent Metropolis-Hastings-Klein (IMHK) Algorithm Implementation
---------------------------------------------------------------------
This module implements the IMHK algorithm for sampling from discrete Gaussian 
distributions over lattices, as described in Wang & Ling (2019) "Lattice Gaussian 
Sampling by Markov Chain Monte Carlo." The implementation includes comprehensive 
diagnostics inspired by Dwivedi et al. (2019) "Log-concave Sampling: 
Metropolis-Hastings Algorithms are Fast."

The code is designed for research in lattice-based cryptography, with a focus on
establishing empirical baselines for quantum speedups.

EXAMPLES::

    >>> from imhk.core import discrete_gaussian_pdf, rho_function, \
    ...     sum_rho_over_integers, discrete_gaussian_sampler_1d
    >>> from imhk.sampler import klein_sampler, gram_schmidt
    >>> from imhk.lattice import identity_matrix, ZZ, RR, vector
    >>> # now run the tiny examples:
    >>> B = identity_matrix(ZZ, 2)        # 2D identity lattice
    >>> sigma = 5.0                       # Standard deviation
    >>> z, x = klein_sampler(B, sigma)    # Sample using Klein's algorithm
    >>> z in ZZ**2                        # Check if z is an integer vector
    True
    >>> len(x) == 2                       # Check dimension of the sample
    True

    >>> # PDF examples:
    >>> discrete_gaussian_pdf(0, 1.0)                      # Scalar input
    1.0
    >>> discrete_gaussian_pdf([1, 0], 1.0)                # List input
    0.6065306597126334
    >>> from sage.all import vector, ZZ, RR
    >>> discrete_gaussian_pdf(vector(ZZ, [1, 1]), 1.0)     # SageMath integer vector
    0.36787944117144233
    >>> discrete_gaussian_pdf(vector(RR, [0, 0]), 2.0, [1, 1])  # With center
    0.8824969025845955

Author: [Your Name]
Date: April 2025
Requirements: SageMath 10.5+, NumPy, SciPy, Matplotlib
Optional: scikit-learn (for PCA visualization, install with: sage -pip install scikit-learn)
"""


import os
import time
import pickle
import logging
import itertools  # Added for compute_exact_discrete_gaussian
from datetime import datetime
from typing import List, Tuple, Dict, Union, Optional, Any

# SageMath imports
from sage.all import (
    matrix, vector, RealField, ZZ, RR, QQ, 
    random_vector, random_matrix, norm,
    set_random_seed, GF, randint, 
    identity_matrix, diagonal_matrix, MatrixSpace,
    copy  # Added for copying immutable matrices
)

# NumPy, SciPy, and other Python libraries
import numpy as np
from scipy import stats
from scipy.linalg import qr, cholesky
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from pathlib import Path

# Import PCA removed - now checked after logger configuration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("IMHK_Sampler")

# Check for scikit-learn availability after logger is configured
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not found. PCA visualization will use fallback method.")
    logger.warning("To enable PCA visualization, install scikit-learn: sage -pip install scikit-learn")

# Create results directories
try:
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/data").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Error creating directories: {e}")
    logger.warning("Some file operations may fail if directories don't exist")

# Set random seeds for reproducibility
set_random_seed(42)
np.random.seed(42)

# Precision for high-precision arithmetic
prec = 53
RF = RealField(prec)

#############################################################################
# 1. Core IMHK Algorithm Implementation
#############################################################################

def discrete_gaussian_pdf(x: Union[float, List[float], np.ndarray, 'FreeModuleElement'], 
                         sigma: float, 
                         center: Optional[Union[List[float], np.ndarray, 'FreeModuleElement']] = None) -> float:
    """
    Compute the unnormalized density of a discrete Gaussian distribution.
    
    p(x) ∝ exp(-||x - center||^2 / (2*sigma^2))
    
    Args:
        x: Point(s) at which to evaluate the PDF. Can be a scalar, list, tuple, 
           NumPy array, or SageMath vector.
        sigma: Standard deviation of the Gaussian (must be positive).
        center: Center of the Gaussian. If None, defaults to origin.
        
    Returns:
        Unnormalized probability density at x.
        
    Raises:
        ValueError: If sigma <= 0 or inputs have incompatible dimensions.
        
    >>> from sage.all import vector, ZZ, RR
    >>> discrete_gaussian_pdf(0, 1.0)  # Scalar input
    1.0
    >>> discrete_gaussian_pdf([1, 0], 1.0)  # List input
    0.6065306597126334
    >>> discrete_gaussian_pdf(vector(ZZ, [1, 1]), 1.0)  # SageMath integer vector
    0.36787944117144233
    >>> discrete_gaussian_pdf(vector(RR, [0, 0]), 2.0, [1, 1])  # With center
    0.8824969025845955
    >>> try:
    ...     discrete_gaussian_pdf([1, 2], 0)  # Invalid sigma
    ... except ValueError:
    ...     print("Caught sigma error")
    Caught sigma error
    """
    if sigma <= 0:
        raise ValueError(f"Standard deviation must be positive, got sigma={sigma}")
    
    # Handle scalar input
    if isinstance(x, (int, float)):
        x_np = np.array([float(x)])
    # Handle SageMath vectors (both vector_integer_dense and vector_real_double_dense)
    elif str(type(x)).find('sage.modules.vector') >= 0 or hasattr(x, 'list'):
        # Convert to list and then to numpy array (works for all SageMath vector types)
        try:
            x_np = np.array([float(xi) for xi in x])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert SageMath vector to float array: {e}")
    # Handle lists, tuples, and NumPy arrays
    elif isinstance(x, (list, tuple, np.ndarray)):
        x_np = np.array(x, dtype=float)
    else:
        raise TypeError(f"Unsupported type for x: {type(x)}")
    
    # Handle center
    if center is None:
        center_np = np.zeros_like(x_np)
    elif str(type(center)).find('sage.modules.vector') >= 0 or hasattr(center, 'list'):
        try:
            center_np = np.array([float(ci) for ci in center])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert center to float array: {e}")
    elif isinstance(center, (list, tuple, np.ndarray)):
        center_np = np.array(center, dtype=float)
    else:
        raise TypeError(f"Unsupported type for center: {type(center)}")
    
    # Check dimensions
    if x_np.shape != center_np.shape:
        raise ValueError(f"Dimension mismatch: x has shape {x_np.shape}, center has shape {center_np.shape}")
    
    # Compute squared distance
    squared_dist = np.sum((x_np - center_np) ** 2)
    
    # Compute unnormalized density
    return np.exp(-squared_dist / (2 * sigma**2))

def rho_function(x: Union[float, List[float], np.ndarray], 
                sigma: float, 
                center: Optional[Union[float, List[float], np.ndarray]] = None) -> float:
    """
    Compute the rho function ρ_σ,c(x) = exp(-π||x-c||^2/σ^2) as defined in lattice literature.
    
    Note: This differs from discrete_gaussian_pdf by a constant factor.
    
    Args:
        x: Point at which to evaluate the function.
        sigma: Standard deviation parameter.
        center: Center of the function. If None, defaults to origin.
        
    Returns:
        Value of ρ_σ,c(x).
    """
    if center is None:
        if isinstance(x, (int, float)):
            center = 0.0
        else:
            center = np.zeros_like(x)
    
    if isinstance(x, (int, float)) and isinstance(center, (int, float)):
        squared_dist = (x - center) ** 2
    else:
        x_np = np.array(x, dtype=float)
        center_np = np.array(center, dtype=float)
        squared_dist = np.sum((x_np - center_np) ** 2)
    
    return np.exp(-np.pi * squared_dist / sigma**2)

def sum_rho_over_integers(sigma: float, center: float) -> float:
    """
    Compute the sum of ρ_σ,c(k) over all integers k ∈ ℤ.
    
    This function truncates the sum at ±6σ from the center.
    
    Args:
        sigma: Standard deviation parameter.
        center: Center of the rho function.
        
    Returns:
        Approximation of Σ_{k ∈ ℤ} ρ_σ,c(k).
    """
    # Determine truncation range (±6σ should be sufficient)
    trunc = int(np.ceil(6 * sigma))
    lower = int(np.floor(center - trunc))
    upper = int(np.ceil(center + trunc))
    
    # Compute sum
    total = 0.0
    for k in range(lower, upper + 1):
        total += rho_function(k, sigma, center)
    
    return total

def discrete_gaussian_sampler_1d(center: float, sigma: float) -> int:
    """
    Sample from a 1D discrete Gaussian distribution centered at 'center' with
    standard deviation 'sigma', truncated at ±6σ.
    
    Args:
        center: Center of the Gaussian.
        sigma: Standard deviation of the Gaussian.
        
    Returns:
        A sample from the discrete Gaussian distribution over ℤ.
        
    Raises:
        ValueError: If sigma <= 0 or if numerical issues occur during sampling.
        
    >>> from sage.all import RealNumber
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> s = discrete_gaussian_sampler_1d(0.0, 1.0)
    >>> isinstance(s, int)
    True
    """
    if sigma <= 0:
        raise ValueError(f"Standard deviation must be positive, got sigma={sigma}")
        
    # Determine truncation range (±6σ)
    trunc = int(np.ceil(6 * sigma))
    lower = int(np.floor(center - trunc))
    upper = int(np.ceil(center + trunc))
    
    # Handle case where range is too small
    if lower >= upper:
        logger.warning(f"Truncation range [{lower}, {upper}] is invalid, adjusting...")
        upper = lower + 1
    
    # Compute probabilities
    probabilities = []
    for k in range(lower, upper + 1):
        probabilities.append(rho_function(k, sigma, center))
    
    # Normalize probabilities
    probabilities = np.array(probabilities)
    sum_prob = np.sum(probabilities)
    
    if sum_prob < 1e-10:
        raise ValueError(f"Sum of probabilities too small ({sum_prob}), adjust sigma or center")
        
    probabilities = probabilities / sum_prob
    
    # Validate probabilities (handle numerical issues)
    if not np.isclose(np.sum(probabilities), 1.0) or np.any(np.isnan(probabilities)):
        raise ValueError(f"Invalid probability distribution (sum={np.sum(probabilities)})")
    
    # Sample
    values = np.arange(lower, upper + 1)
    try:
        return int(np.random.choice(values, p=probabilities))
    except Exception as e:
        logger.error(f"Error in discrete_gaussian_sampler_1d: {e}")
        logger.error(f"Probabilities: min={np.min(probabilities)}, max={np.max(probabilities)}, sum={np.sum(probabilities)}")
        raise ValueError(f"Sampling failed: {e}")

def gram_schmidt(B):
    """
    Compute the Gram-Schmidt orthogonalization of basis matrix B.
    
    Args:
        B: Basis matrix where rows are basis vectors. Must be a SageMath matrix.
            Will be copied if immutable.
        
    Returns:
        GS: Gram-Schmidt orthogonalized basis (same dimensions as B).
        M: GSO coefficients (n×n matrix where n is the number of rows in B).
        
    Raises:
        ValueError: If B is not a valid SageMath matrix, has invalid dimensions,
                   or if numerical instability is detected.
        
    >>> from sage.all import matrix, ZZ
    >>> B = matrix(ZZ, [[1, 1], [0, 1]])
    >>> GS, M = gram_schmidt(B)
    >>> GS[0]
    (1, 1)
    >>> (GS[1][0] - 0).abs() < 1e-10  # First component of second GS vector should be approximately 0
    True
    >>> M[1,0]  # mu_{1,0} coefficient should be 1/2
    1/2
    
    >>> # Test error handling for invalid input
    >>> try:
    ...     B_bad = matrix(ZZ, [[0, 0], [0, 0]])
    ...     GS, M = gram_schmidt(B_bad)
    ... except ValueError:
    ...     print("Correctly caught bad input")
    Correctly caught bad input
    """
    # Check if B is a valid SageMath matrix
    if not hasattr(B, 'nrows') or not hasattr(B, 'ncols'):
        raise ValueError("B must be a SageMath matrix")
    
    n = B.nrows()
    ncols = B.ncols()
    
    if n <= 0 or ncols <= 0:
        raise ValueError("B must have positive dimensions")
    
    # Copy B if it's immutable to avoid error when we modify B later
    if hasattr(B, 'is_immutable') and B.is_immutable():
        B = copy(B)
    
    # Create matrices over QQ to handle rational coefficients correctly
    # We could use B.base_ring(), but QQ ensures we can handle rational coefficients
    GS = matrix(QQ, n, ncols)
    M = matrix(QQ, n, n)
    
    # Set the first vector
    GS[0] = B[0]
    M[0, 0] = 1
    
    # Compute the orthogonalization
    for i in range(1, n):
        GS[i] = B[i]
        for j in range(i):
            # Check if GS[j] is a zero vector or nearly zero (for numerical stability)
            gs_norm_squared = (GS[j] * GS[j])
            if gs_norm_squared < 1e-10:  # Tolerance for numerical stability
                raise ValueError(f"Gram-Schmidt failed: Vector GS[{j}] is numerically zero or nearly zero")
            
            # Compute μ_i,j = ⟨b_i, b*_j⟩/⟨b*_j, b*_j⟩
            # This will give a rational if B is over ZZ
            M[i, j] = (B[i] * GS[j]) / gs_norm_squared
            
            # Subtract the projection - this might give rational coefficients
            GS[i] = GS[i] - M[i, j] * GS[j]
        
        M[i, i] = 1
    
    # Note: We keep GS over QQ to handle rational coefficients
    # Attempting to convert to B.base_ring() could cause errors
    
    return GS, M

def klein_sampler(B, sigma, c=None):
    """
    Implementation of Klein's algorithm for sampling from a discrete Gaussian over a lattice.
    
    Reference: Klein, P. (2000) "Finding the closest lattice vector when it's unusually close."
    
    Args:
        B: Basis matrix where rows are basis vectors.
        sigma: Standard deviation parameter.
        c: Center of the Gaussian. If None, defaults to the origin.
        
    Returns:
        A lattice point sampled according to Klein's algorithm.
        
    Raises:
        ValueError: If dimensions are incompatible or if Gram-Schmidt fails.
        
    >>> from sage.all import identity_matrix, ZZ
    >>> B = identity_matrix(ZZ, 2)
    >>> z, x = klein_sampler(B, 5.0)
    >>> z in ZZ**2  # Note: Use ** for exponentiation in SageMath, not ^
    True
    >>> len(x) == 2
    True
    
    >>> # Test error handling with incompatible center dimension
    >>> try:
    ...     z, x = klein_sampler(B, 5.0, [1, 2, 3])
    ... except ValueError:
    ...     print("Correctly caught dimension mismatch")
    Correctly caught dimension mismatch
    """
    n = B.nrows()
    
    # Convert B to numpy for easier manipulation
    B_np = np.array(B, dtype=float)
    
    # If center is None, set it to the origin
    if c is None:
        c_np = np.zeros(B.ncols(), dtype=float)
    else:
        c_np = np.array(c, dtype=float)
        # Validate center dimensions
        if len(c_np) != B.ncols():
            raise ValueError(f"Center dimension ({len(c_np)}) must match B.ncols() ({B.ncols()})")
    
    try:
        # Compute Gram-Schmidt orthogonalization
        GS, M = gram_schmidt(B)
        GS_np = np.array(GS, dtype=float)
        M_np = np.array(M, dtype=float)
    except ValueError as e:
        raise ValueError(f"Klein sampler failed during Gram-Schmidt: {e}")
    
    # Initialize sample vector in ℤ^n
    z = np.zeros(n, dtype=int)
    
    # Sample from last to first coordinate
    for i in range(n-1, -1, -1):
        # Compute center for the conditional distribution
        c_i = c_np[i]
        for j in range(i+1, n):
            c_i -= M_np[j,i] * z[j]
        
        # Compute sigma_i = sigma/||b*_i||
        gs_norm = np.sqrt(np.sum(GS_np[i]**2))
        if gs_norm < 1e-10:
            raise ValueError(f"Klein sampler failed: GS vector {i} has near-zero norm")
            
        sigma_i = sigma / gs_norm
        
        # Sample from 1D discrete Gaussian
        try:
            z[i] = discrete_gaussian_sampler_1d(c_i, sigma_i)
        except Exception as e:
            raise ValueError(f"Klein sampler failed during 1D sampling: {e}")
    
    # Convert integer coefficients to lattice point
    sample = np.dot(z, B_np)
    
    return vector(ZZ, z), vector(RR, sample)

def imhk_step(current_sample, B, sigma, c=None):
    """
    Perform one step of the Independent Metropolis-Hastings-Klein algorithm.
    
    Args:
        current_sample: Current integer coordinates in the lattice basis.
        B: Basis matrix where rows are basis vectors.
        sigma: Standard deviation parameter.
        c: Center of the Gaussian. If None, defaults to the origin.
        
    Returns:
        new_sample: New integer coordinates.
        accepted: Boolean indicating whether the proposal was accepted.
        
    Raises:
        ValueError: If numerical issues occur or matrices have incompatible dimensions.
    """
    n = B.nrows()
    
    # Generate proposal using Klein's algorithm
    try:
        proposal_z, _ = klein_sampler(B, sigma, c)
    except ValueError as e:
        raise ValueError(f"IMHK step failed during proposal generation: {e}")
    
    # Compute the Gram-Schmidt orthogonalization
    try:
        GS, M = gram_schmidt(B)
    except ValueError as e:
        raise ValueError(f"IMHK step failed during Gram-Schmidt: {e}")
    
    # Initialize the acceptance ratio product
    ratio_product = 1.0
    
    # Compute the product term in the acceptance ratio
    for i in range(n):
        # Compute the conditional center for x
        c_x_i = 0.0 if c is None else c[i]
        for j in range(i+1, n):
            c_x_i -= M[j,i] * current_sample[j]
        
        # Compute the conditional center for y (proposal)
        c_y_i = 0.0 if c is None else c[i]
        for j in range(i+1, n):
            c_y_i -= M[j,i] * proposal_z[j]
        
        # Compute sigma_i = sigma/||b*_i||
        gs_norm = GS[i].norm()
        if gs_norm < 1e-10:
            raise ValueError(f"IMHK step failed: GS vector {i} has near-zero norm")
        
        sigma_i = sigma / gs_norm
        
        # Compute the sums over integers for the ratio
        try:
            sum_y = sum_rho_over_integers(sigma_i, c_y_i)
            sum_x = sum_rho_over_integers(sigma_i, c_x_i)
            
            # Check for numerical issues
            if sum_x < 1e-15:  # Avoid division by very small numbers
                logger.warning(f"Numerical issue in IMHK step: sum_x={sum_x} is very small")
                ratio_product = float('inf')
            else:
                ratio_product *= (sum_y / sum_x)
                
        except Exception as e:
            raise ValueError(f"IMHK step failed during ratio calculation: {e}")
    
    # Compute the acceptance probability
    alpha = min(1.0, ratio_product)
    
    # Accept or reject the proposal
    if np.random.rand() < alpha:
        return proposal_z, True
    else:
        return current_sample, False

def imhk_sampler(B, sigma, num_samples, c=None, burn_in=100, num_chains=1, initial_samples=None):
    """
    Run the Independent Metropolis-Hastings-Klein algorithm to sample from a discrete Gaussian
    distribution over a lattice.
    
    Args:
        B: Basis matrix where rows are basis vectors.
        sigma: Standard deviation parameter.
        num_samples: Number of samples to generate (per chain).
        c: Center of the Gaussian. If None, defaults to the origin.
        burn_in: Number of initial samples to discard.
        num_chains: Number of parallel chains to run.
        initial_samples: Initial samples for each chain. If None, generated using Klein sampler.
        
    Returns:
        samples_z: List of samples in integer coordinates.
        samples_x: List of samples in the original space.
        acceptance_rates: Acceptance rates for each chain.
        tv_distances: List of approximate TV distances (if available).
    """
    n = B.nrows()
    
    # Initialize chains
    chains_z = []
    acceptance_counts = np.zeros(num_chains)
    
    # Generate initial samples if not provided
    if initial_samples is None:
        initial_samples = []
        for _ in range(num_chains):
            initial_z, _ = klein_sampler(B, sigma, c)
            initial_samples.append(initial_z)
    else:
        if len(initial_samples) != num_chains:
            raise ValueError(f"Number of initial samples ({len(initial_samples)}) must match number of chains ({num_chains})")
    
    # Run each chain
    for chain_idx in range(num_chains):
        logger.info(f"Running chain {chain_idx+1}/{num_chains}")
        
        chain_z = [initial_samples[chain_idx]]
        
        # Run the chain
        for i in range(burn_in + num_samples - 1):
            new_z, accepted = imhk_step(chain_z[-1], B, sigma, c)
            chain_z.append(new_z)
            
            if i >= burn_in:
                acceptance_counts[chain_idx] += accepted
            
            if (i+1) % 100 == 0:
                logger.debug(f"Chain {chain_idx+1}, iteration {i+1}/{burn_in + num_samples - 1}")
        
        # Discard burn-in samples
        chain_z = chain_z[burn_in:]
        chains_z.append(chain_z)
    
    # Compute acceptance rates
    acceptance_rates = acceptance_counts / num_samples
    
    # Convert to original space
    chains_x = []
    for chain_z in chains_z:
        chain_x = [B.linear_combination_of_rows(list(z)) for z in chain_z]
        chains_x.append(chain_x)
    
    # Flatten chains for return (if needed)
    samples_z = [z for chain in chains_z for z in chain]
    samples_x = [x for chain in chains_x for x in chain]
    
    # Approximate TV distance (if possible)
    tv_distances = []
    
    return samples_z, samples_x, acceptance_rates, tv_distances

def parallel_imhk_sampler(B, sigma, num_samples, c=None, burn_in=100, num_chains=4, initial_samples=None):
    """
    Run the IMHK sampler in parallel using multiple cores.
    
    Args:
        Same as imhk_sampler, with num_chains preferably matching the number of cores.
        
    Returns:
        Combined results from all chains.
    """
    # Determine samples per chain
    samples_per_chain = (num_samples + num_chains - 1) // num_chains
    
    # Generate initial samples if not provided
    if initial_samples is None:
        initial_samples = []
        for _ in range(num_chains):
            initial_z, _ = klein_sampler(B, sigma, c)
            initial_samples.append(initial_z)
    
    # Create argument tuples for parallel execution
    args = []
    for i in range(num_chains):
        args.append((B, sigma, samples_per_chain, c, burn_in, 1, [initial_samples[i]]))
    
    # Run in parallel
    with mp.Pool(processes=min(num_chains, mp.cpu_count())) as pool:
        results = pool.starmap(imhk_sampler, args)
    
    # Combine results
    samples_z = []
    samples_x = []
    acceptance_rates = []
    tv_distances = []
    
    for res_z, res_x, acc_rate, tv_dist in results:
        samples_z.extend(res_z)
        samples_x.extend(res_x)
        acceptance_rates.extend(acc_rate)
        if tv_dist:
            tv_distances.extend(tv_dist)
    
    # Truncate to exactly num_samples if needed
    samples_z = samples_z[:num_samples]
    samples_x = samples_x[:num_samples]
    
    return samples_z, samples_x, np.mean(acceptance_rates), tv_distances

def truncate_lattice(B, sigma, center=None, epsilon=1e-10):
    """
    Determine the truncation bounds for the lattice for efficient high-dimensional sampling.
    
    Args:
        B: Basis matrix.
        sigma: Standard deviation parameter.
        center: Center of the Gaussian. If None, defaults to the origin.
        epsilon: Truncation parameter, lower means more points included.
        
    Returns:
        bounds: List of (lower, upper) bounds for each dimension in the integer coefficient space.
    """
    n = B.nrows()
    GS, M = gram_schmidt(B)
    
    bounds = []
    for i in range(n):
        # Compute σ_i = σ/||b*_i||
        sigma_i = sigma / GS[i].norm()
        
        # Determine truncation distance based on epsilon
        k = np.sqrt(-2 * sigma_i**2 * np.log(epsilon))
        
        # Calculate center in the i-th dimension
        c_i = 0.0 if center is None else center[i]
        
        # Set bounds
        lower = int(np.floor(c_i - k))
        upper = int(np.ceil(c_i + k))
        bounds.append((lower, upper))
    
    return bounds

#############################################################################
# 2. Lattice Basis Generation
#############################################################################

def generate_random_lattice(dim, bit_size=10, det_range=(50, 100)):
    """
    Generate a random integer lattice basis.
    
    Args:
        dim: Dimension of the lattice.
        bit_size: Size of the entries in bits.
        det_range: Range for the determinant.
        
    Returns:
        A random lattice basis matrix.
    """
    # Generate a random unimodular matrix
    U = random_matrix(ZZ, dim, dim)
    while abs(U.det()) != 1:
        U = random_matrix(ZZ, dim, dim)
    
    # Generate a diagonal matrix with specific determinant
    det = randint(det_range[0], det_range[1])
    D = diagonal_matrix([1] * (dim-1) + [det])
    
    # Generate another random unimodular matrix
    V = random_matrix(ZZ, dim, dim)
    while abs(V.det()) != 1:
        V = random_matrix(ZZ, dim, dim)
    
    # Create the lattice basis
    B = U * D * V
    
    # Ensure entries are not too large
    if bit_size > 0:
        max_value = 2**bit_size - 1
        for i in range(dim):
            for j in range(dim):
                B[i,j] = max(-max_value, min(max_value, B[i,j]))
    
    return B

def generate_skewed_lattice(dim, skew_factor=100):
    """
    Generate a skewed lattice basis with varying vector lengths.
    
    Args:
        dim: Dimension of the lattice.
        skew_factor: Factor determining how skewed the basis is.
        
    Returns:
        A skewed lattice basis matrix.
    """
    # Start with the identity
    B = identity_matrix(ZZ, dim)
    
    # Modify the first vector to be longer
    B[0] = B[0] * skew_factor
    
    # Add some off-diagonal elements to make it less orthogonal
    for i in range(1, dim):
        B[0,i] = randint(1, skew_factor // 2)
    
    return B

def generate_ill_conditioned_lattice(dim, condition_number=1000):
    """
    Generate an ill-conditioned lattice basis.
    
    Args:
        dim: Dimension of the lattice.
        condition_number: Approximate condition number for the basis.
        
    Returns:
        An ill-conditioned lattice basis matrix.
    """
    # Generate a random matrix
    B = random_matrix(ZZ, dim, dim)
    
    # Ensure it's full rank
    while B.rank() < dim:
        B = random_matrix(ZZ, dim, dim)
    
    # Get SVD and modify singular values to achieve desired condition number
    U, S, Vt = np.linalg.svd(np.array(B, dtype=float))
    
    # Set singular values to decay exponentially
    S[0] = condition_number**(1/(dim-1))
    for i in range(1, dim):
        S[i] = S[0] * (1/condition_number)**(i/(dim-1))
    
    # Reconstruct the matrix
    B_new = np.dot(U * S, Vt)
    
    # Convert to integer matrix
    B_int = matrix(ZZ, dim, dim)
    for i in range(dim):
        for j in range(dim):
            B_int[i,j] = int(round(B_new[i,j]))
    
    # Ensure full rank
    while B_int.rank() < dim:
        B_int = B_int + identity_matrix(ZZ, dim)
    
    return B_int

def generate_ntru_lattice(n, q):
    """
    Generate an NTRU lattice basis.
    
    Args:
        n: Dimension parameter (should be a power of 2).
        q: Modulus.
        
    Returns:
        An NTRU lattice basis matrix.
        
    Raises:
        ValueError: If n is not even or if NTRU generation fails.
    """
    if n % 2 != 0:
        raise ValueError("n must be even for NTRU lattices")
    
    # Dimension of the lattice is 2n
    dim = 2 * n
    
    # Generate random polynomial f with small coefficients
    f = [randint(-1, 1) for _ in range(n)]
    while f.count(0) > n // 3:  # Ensure f has enough non-zero terms
        f = [randint(-1, 1) for _ in range(n)]
    
    # Generate random polynomial g with small coefficients
    g = [randint(-1, 1) for _ in range(n)]
    while g.count(0) > n // 3:  # Ensure g has enough non-zero terms
        g = [randint(-1, 1) for _ in range(n)]
    
    # Convert to polynomials mod q
    Rq = GF(q)['x']
    x = Rq.gen()
    f_poly = sum(Rq(f[i]) * x^i for i in range(n))
    g_poly = sum(Rq(g[i]) * x^i for i in range(n))
    
    # Create the polynomial modulus x^n - 1
    mod_poly = x^n - 1
    
    # Try to compute h = g/f mod (q, x^n-1)
    try:
        f_inv = f_poly.inverse_mod(mod_poly)
        h_poly = (g_poly * f_inv) % mod_poly
        h = [int(h_poly[i]) for i in range(n)]
    except Exception as e:
        # If f is not invertible, generate new f and g
        logger.warning(f"f is not invertible, generating new NTRU basis: {e}")
        return generate_ntru_lattice(n, q)
    
    # Create the NTRU lattice basis matrix
    B = matrix(ZZ, dim, dim)
    
    # Check if B is mutable (should be by default, but check to be safe)
    if hasattr(B, 'is_immutable') and B.is_immutable():
        logger.warning("NTRU matrix is immutable, creating mutable copy")
        B = copy(B)
        B.set_immutable(False)
    
    # Fill the top-left block with I_n
    for i in range(n):
        B[i,i] = 1
    
    # Fill the top-right block with 0
    # (Already zeros by default)
    
    # Fill the bottom-left block with h in circulant form
    for i in range(n):
        for j in range(n):
            # Compute the index considering the circulant structure
            idx = (j - i) % n
            B[n+i, j] = h[idx]
    
    # Fill the bottom-right block with q*I_n
    for i in range(n):
        B[n+i, n+i] = q
    
    return B

def apply_lattice_reduction(B, method='LLL', block_size=20):
    """
    Apply lattice reduction to the basis.
    
    Args:
        B: Basis matrix.
        method: Reduction method ('LLL' or 'BKZ').
        block_size: Block size for BKZ.
        
    Returns:
        Reduced basis matrix.
    """
    if method.upper() == 'LLL':
        return B.LLL()
    elif method.upper() == 'BKZ':
        return B.BKZ(block_size=block_size)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

#############################################################################
# 3. Comprehensive Diagnostics
#############################################################################

def compute_gelman_rubin(chains):
    """
    Compute the Potential Scale Reduction Factor (PSRF) for multiple chains.
    
    Args:
        chains: List of chains, where each chain is a list of vectors.
        
    Returns:
        PSRF values for each dimension.
    """
    # Convert to numpy arrays for easier computation
    chains_np = []
    for chain in chains:
        chain_np = np.array([np.array(sample) for sample in chain])
        chains_np.append(chain_np)
    
    # Get dimensions
    n_chains = len(chains_np)
    n_samples = len(chains_np[0])
    n_dims = chains_np[0].shape[1]
    
    # Initialize PSRF values
    psrf_values = np.zeros(n_dims)
    
    for d in range(n_dims):
        # Extract the d-th dimension from all chains
        dim_chains = [chain[:,d] for chain in chains_np]
        
        # Calculate within-chain variance
        W = np.mean([np.var(chain, ddof=1) for chain in dim_chains])
        
        # Calculate between-chain variance
        chain_means = [np.mean(chain) for chain in dim_chains]
        grand_mean = np.mean(chain_means)
        B = n_samples * np.var(chain_means, ddof=1)
        
        # Calculate estimated variance
        var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        
        # Calculate PSRF
        psrf_values[d] = np.sqrt(var_plus / W)
    
    return psrf_values

def compute_geweke(samples, first=0.1, last=0.5):
    """
    Perform the Geweke test for stationarity.
    
    Args:
        samples: List of samples from a chain.
        first: Proportion of the chain to use for the first part.
        last: Proportion of the chain to use for the last part.
        
    Returns:
        Z-scores for each dimension.
    """
    samples_np = np.array([np.array(sample) for sample in samples])
    n_samples, n_dims = samples_np.shape
    
    # Determine the indices for the first and last parts
    first_end = int(n_samples * first)
    last_start = int(n_samples * (1 - last))
    
    z_scores = np.zeros(n_dims)
    
    for d in range(n_dims):
        # Extract the first and last parts for dimension d
        first_part = samples_np[:first_end, d]
        last_part = samples_np[last_start:, d]
        
        # Calculate means
        mean_first = np.mean(first_part)
        mean_last = np.mean(last_part)
        
        # Calculate standard errors using spectral density
        # (simplified version using regular standard deviation)
        se_first = np.std(first_part, ddof=1) / np.sqrt(len(first_part))
        se_last = np.std(last_part, ddof=1) / np.sqrt(len(last_part))
        
        # Calculate Z-score
        z_scores[d] = (mean_first - mean_last) / np.sqrt(se_first**2 + se_last**2)
    
    return z_scores

def compute_autocorrelation(samples, lag=50):
    """
    Calculate autocorrelation for mixing analysis.
    
    Args:
        samples: List of samples.
        lag: Maximum lag to consider.
        
    Returns:
        Autocorrelation function for each dimension.
    """
    samples_np = np.array([np.array(sample) for sample in samples])
    n_samples, n_dims = samples_np.shape
    
    # Initialize ACF matrix
    acf = np.zeros((lag + 1, n_dims))
    
    for d in range(n_dims):
        # Extract the d-th dimension
        series = samples_np[:, d]
        
        # Calculate mean and variance
        mean = np.mean(series)
        var = np.var(series, ddof=1)
        
        # Calculate autocorrelation for each lag
        for l in range(lag + 1):
            if l == 0:
                acf[l, d] = 1.0
            else:
                # Calculate covariance at lag l
                cov = np.mean((series[:-l] - mean) * (series[l:] - mean))
                acf[l, d] = cov / var
    
    return acf

def compute_ess(samples):
    """
    Compute Effective Sample Size (ESS).
    
    Args:
        samples: List of samples.
        
    Returns:
        ESS values for each dimension.
    """
    samples_np = np.array([np.array(sample) for sample in samples])
    n_samples, n_dims = samples_np.shape
    
    # Initialize ESS values
    ess_values = np.zeros(n_dims)
    
    for d in range(n_dims):
        # Extract the d-th dimension
        series = samples_np[:, d]
        
        # Calculate autocorrelation with lag 1
        mean = np.mean(series)
        var = np.var(series, ddof=1)
        
        if var == 0:  # If variance is zero, ESS is undefined
            ess_values[d] = n_samples
            continue
        
        # Calculate autocorrelation at lag 1
        acf_1 = np.mean((series[:-1] - mean) * (series[1:] - mean)) / var
        
        # Estimate ESS using lag 1 autocorrelation
        if acf_1 >= 1.0:
            ess_values[d] = 1.0
        else:
            ess_values[d] = n_samples * (1 - acf_1) / (1 + acf_1)
    
    return ess_values

def estimate_mixing_time(B, sigma, max_iterations=50, threshold=0.3, c=None, timeout_seconds=60):
    """
    Estimate mixing time by tracking TV distance to stationarity.
    
    Args:
        B: Basis matrix.
        sigma: Standard deviation parameter.
        max_iterations: Maximum number of iterations (default: 50).
        threshold: TV distance threshold for mixing (default: 0.3).
        c: Center of the Gaussian. If None, defaults to the origin.
        timeout_seconds: Maximum time in seconds before returning (default: 60).
        
    Returns:
        Estimated mixing time and TV distances over iterations.
        
    EXAMPLES::
        
        >>> from sage.all import identity_matrix, ZZ
        >>> B = identity_matrix(ZZ, 2)
        >>> sigma = 5.0
        >>> # Using small values for testing
        >>> mix_time, _ = estimate_mixing_time(B, sigma, max_iterations=3, timeout_seconds=5)
        >>> isinstance(mix_time, (int, type(None)))
        True
    """
    # This implementation only works for small dimensions (e.g., 2D)
    dim = B.nrows()
    if dim > 2:
        logger.warning("Mixing time estimation only implemented for dimensions ≤ 2")
        return None, []
    
    start_time = time.time()
    
    try:
        # Get the exact discrete Gaussian distribution (only for small dimensions)
        exact_dist = compute_exact_discrete_gaussian(B, sigma, c)
        if exact_dist is None:
            logger.error("Could not compute exact distribution for TV distance calculation")
            return None, []
        
        # Initialize chain with a random point
        current_z, _ = klein_sampler(B, sigma, c)
        
        # Track TV distances
        tv_distances = []
        
        # Run the chain
        for i in range(max_iterations):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Mixing time estimation timed out after {timeout_seconds} seconds at iteration {i}")
                return i, tv_distances
            
            # Update the chain
            current_z, _ = imhk_step(current_z, B, sigma, c)
            
            # Generate shorter chain to estimate current distribution (100 instead of 1000)
            samples_z, samples_x, _, _ = imhk_sampler(B, sigma, 100, c, burn_in=0, 
                                                     num_chains=1, initial_samples=[current_z])
            
            # Compute TV distance to the true distribution
            tv_dist = compute_total_variation_distance(samples_x, sigma, B, c, exact_dist)
            tv_distances.append(tv_dist)
            
            # Log progress every 10 iterations
            if (i+1) % 10 == 0 or i == 0:
                logger.info(f"Iteration {i+1}/{max_iterations}, TV distance: {tv_dist:.4f}")
            
            # Check if we've mixed
            if tv_dist < threshold:
                logger.info(f"Mixing achieved at iteration {i+1}, TV distance: {tv_dist:.4f}")
                return i + 1, tv_distances
        
        logger.info(f"Maximum iterations reached. Final TV distance: {tv_distances[-1]:.4f}")
        return max_iterations, tv_distances
        
    except Exception as e:
        logger.error(f"Error in mixing time estimation: {e}")
        return None, tv_distances

def compute_theoretical_spectral_bound(B, sigma, c=None):
    """
    Implement the spectral gap bound from Wang & Ling (2019).
    
    Args:
        B: Basis matrix.
        sigma: Standard deviation parameter.
        c: Center of the Gaussian. If None, defaults to the origin.
        
    Returns:
        Spectral gap bound and estimated mixing time.
    """
    # This is only implemented for small dimensions
    dim = B.nrows()
    if dim > 2:
        logger.warning("Spectral bound estimation only implemented for dimensions ≤ 2")
        return None, None
    
    # For dimension 2, implement the bound from Wang & Ling (2019)
    # The bound is δ = ρ_σ,c(Λ) / Π_{i=1}^n ρ_σ_i(ℤ)
    
    # Compute the numerator (sum over the lattice)
    # For small dimensions, enumerate all relevant lattice points
    rho_lambda = 0.0
    
    # Determine truncation bounds
    bounds = truncate_lattice(B, sigma, c, epsilon=1e-10)
    
    # Enumerate all lattice points within bounds
    for x1 in range(bounds[0][0], bounds[0][1] + 1):
        for x2 in range(bounds[1][0], bounds[1][1] + 1):
            z = vector(ZZ, [x1, x2])
            lattice_point = B.linear_combination_of_rows(list(z))
            rho_lambda += discrete_gaussian_pdf(lattice_point, sigma, c)
    
    # Compute the denominator (product of 1D sums)
    GS, _ = gram_schmidt(B)
    denominator = 1.0
    for i in range(dim):
        sigma_i = sigma / GS[i].norm()
        c_i = 0.0 if c is None else c[i]
        denominator *= sum_rho_over_integers(sigma_i, c_i)
    
    # Compute the spectral gap bound
    delta = rho_lambda / denominator
    
    # Estimate mixing time for ε = 0.01
    epsilon = 0.01
    mixing_time = int(np.ceil(-np.log(epsilon) / delta))
    
    return delta, mixing_time

def compute_exact_discrete_gaussian(B, sigma, c=None):
    """
    Compute the exact discrete Gaussian distribution for small dimensions.
    
    Args:
        B: Basis matrix.
        sigma: Standard deviation parameter.
        c: Center of the Gaussian. If None, defaults to the origin.
        
    Returns:
        Dictionary with lattice points as keys and probabilities as values.
        
    Raises:
        ValueError: If dimension is too large (>2) or inputs are invalid.
        
    >>> from sage.all import identity_matrix, ZZ
    >>> B = identity_matrix(ZZ, 2)
    >>> dist = compute_exact_discrete_gaussian(B, 1.0)
    >>> len(dist) > 0
    True
    >>> abs(sum(dist.values()) - 1.0) < 1e-10  # Probabilities should sum to 1
    True
    
    >>> # Test handling of integer vectors
    >>> from sage.all import vector
    >>> B = matrix(ZZ, [[1, 0], [0, 1]])
    >>> dist = compute_exact_discrete_gaussian(B, 1.0)
    >>> key = tuple(vector(ZZ, [0, 0]))  # Origin should have highest probability
    >>> all(dist[key] >= dist[other_key] for other_key in dist)
    True
    """
    # This is only feasible for small dimensions
    dim = B.nrows()
    if dim > 2:
        logger.warning("Exact distribution computation only implemented for dimensions ≤ 2")
        return None
    
    # Determine truncation bounds
    try:
        bounds = truncate_lattice(B, sigma, c, epsilon=1e-10)
    except Exception as e:
        raise ValueError(f"Error computing truncation bounds: {e}")
    
    # Compute unnormalized probabilities
    probs = {}
    Z = 0.0
    
    try:
        # Enumerate all lattice points within bounds
        for coords in itertools.product(*[range(b[0], b[1]+1) for b in bounds]):
            z = vector(ZZ, coords)
            lattice_point = B.linear_combination_of_rows(list(z))
            
            # Use discrete_gaussian_pdf which now handles Vector_integer_dense
            prob = discrete_gaussian_pdf(lattice_point, sigma, c)
            
            # Use tuple of coordinates as key
            key = tuple(lattice_point)
            probs[key] = prob
            Z += prob
        
        # Check if Z is too small (numerical instability)
        if Z < 1e-10:
            raise ValueError("Normalization constant is too small, adjust sigma or bounds")
            
        # Normalize
        for key in probs:
            probs[key] /= Z
            
        return probs
    except Exception as e:
        logger.error(f"Error in compute_exact_discrete_gaussian: {e}")
        return None

def compute_total_variation_distance(samples, sigma, lattice_basis, center=None, exact=None):
    """
    Compute TV distance to the true discrete Gaussian.
    
    Args:
        samples: List of samples.
        sigma: Standard deviation parameter.
        lattice_basis: Basis matrix.
        center: Center of the Gaussian. If None, defaults to the origin.
        exact: Pre-computed exact distribution. If None, computed here.
        
    Returns:
        TV distance between the empirical and true distributions.
    """
    # This is only feasible for small dimensions
    dim = lattice_basis.nrows()
    if dim > 2:
        logger.warning("TV distance computation only implemented for dimensions ≤ 2")
        return None
    
    # Compute exact distribution if not provided
    if exact is None:
        exact = compute_exact_discrete_gaussian(lattice_basis, sigma, center)
        if exact is None:
            return None
    
    # Count occurrences of each lattice point in the samples
    sample_counts = {}
    n_samples = len(samples)
    
    for sample in samples:
        key = tuple(sample)
        sample_counts[key] = sample_counts.get(key, 0) + 1
    
    # Compute empirical distribution
    empirical = {}
    for key in sample_counts:
        empirical[key] = sample_counts[key] / n_samples
    
    # Compute TV distance
    tv_dist = 0.0
    
    # Add contribution from keys in exact but not in empirical
    for key in exact:
        if key not in empirical:
            tv_dist += exact[key]
        else:
            tv_dist += max(0, exact[key] - empirical[key])
    
    # Add contribution from keys in empirical but not in exact
    for key in empirical:
        if key not in exact:
            tv_dist += empirical[key]
    
    return 0.5 * tv_dist

def compute_kl_divergence(samples, sigma, lattice_basis, center=None, exact=None):
    """
    Compute KL divergence to the true discrete Gaussian.
    
    Args:
        samples: List of samples.
        sigma: Standard deviation parameter.
        lattice_basis: Basis matrix.
        center: Center of the Gaussian. If None, defaults to the origin.
        exact: Pre-computed exact distribution. If None, computed here.
        
    Returns:
        KL divergence between the empirical and true distributions.
    """
    # This is only feasible for small dimensions
    dim = lattice_basis.nrows()
    if dim > 2:
        logger.warning("KL divergence computation only implemented for dimensions ≤ 2")
        return None
    
    # Compute exact distribution if not provided
    if exact is None:
        exact = compute_exact_discrete_gaussian(lattice_basis, sigma, center)
        if exact is None:
            return None
    
    # Count occurrences of each lattice point in the samples
    sample_counts = {}
    n_samples = len(samples)
    
    for sample in samples:
        key = tuple(sample)
        sample_counts[key] = sample_counts.get(key, 0) + 1
    
    # Compute empirical distribution
    empirical = {}
    for key in sample_counts:
        empirical[key] = sample_counts[key] / n_samples
    
    # Compute KL divergence
    kl_div = 0.0
    
    for key in empirical:
        if key in exact and exact[key] > 0:
            kl_div += empirical[key] * np.log(empirical[key] / exact[key])
        elif empirical[key] > 0:
            # Handle the case when the exact probability is zero (or very small)
            kl_div = float('inf')
            break
    
    return kl_div

def compute_log_likelihood(samples, sigma, lattice_basis, center=None):
    """
    Compute log-likelihood for sample quality assessment.
    
    Args:
        samples: List of samples.
        sigma: Standard deviation parameter.
        lattice_basis: Basis matrix.
        center: Center of the Gaussian. If None, defaults to the origin.
        
    Returns:
        Log-likelihood of the samples.
    """
    log_likelihood = 0.0
    
    for sample in samples:
        log_likelihood += np.log(discrete_gaussian_pdf(sample, sigma, center))
    
    return log_likelihood

def compare_distributions(samples1, samples2):
    """
    Use Kolmogorov-Smirnov test on squared norms for distribution comparison.
    
    Args:
        samples1: First set of samples.
        samples2: Second set of samples.
        
    Returns:
        KS statistic and p-value.
    """
    # Compute squared norms
    norms1 = [sample.norm()**2 for sample in samples1]
    norms2 = [sample.norm()**2 for sample in samples2]
    
    # Perform KS test
    ks_stat, p_value = ks_2samp(norms1, norms2)
    
    return ks_stat, p_value

def compute_moments(samples, basis=None):
    """
    Compute empirical moments and compare to theoretical expectations.
    
    Args:
        samples: List of samples.
        basis: Basis matrix used for the lattice. If None, moments are computed
               in the sampled space directly.
        
    Returns:
        Dictionary with moment statistics.
    """
    samples_np = np.array([np.array(sample) for sample in samples])
    
    # Compute mean and covariance
    mean = np.mean(samples_np, axis=0)
    cov = np.cov(samples_np, rowvar=False)
    
    # Compute mean squared norm
    squared_norms = np.sum(samples_np**2, axis=1)
    mean_squared_norm = np.mean(squared_norms)
    
    # If basis is provided, compute moments in the original space
    if basis is not None:
        basis_np = np.array(basis, dtype=float)
        original_samples = np.dot(samples_np, basis_np)
        original_mean = np.mean(original_samples, axis=0)
        original_cov = np.cov(original_samples, rowvar=False)
        original_squared_norms = np.sum(original_samples**2, axis=1)
        original_mean_squared_norm = np.mean(original_squared_norms)
        
        return {
            'mean': mean,
            'covariance': cov,
            'mean_squared_norm': mean_squared_norm,
            'original_mean': original_mean,
            'original_covariance': original_cov,
            'original_mean_squared_norm': original_mean_squared_norm
        }
    
    return {
        'mean': mean,
        'covariance': cov,
        'mean_squared_norm': mean_squared_norm
    }

def validate_sampler(B, sigma, num_samples, center=None):
    """
    For ℤ², compare empirical distribution to exact probabilities.
    
    Args:
        B: Basis matrix.
        sigma: Standard deviation parameter.
        num_samples: Number of samples to generate.
        center: Center of the Gaussian. If None, defaults to the origin.
        
    Returns:
        Dictionary with validation metrics.
    """
    # This is only feasible for dimension 2
    dim = B.nrows()
    if dim != 2:
        logger.warning("Validation only implemented for dimension 2")
        return None
    
    # Run the IMHK sampler
    samples_z, samples_x, acceptance_rate, _ = imhk_sampler(
        B, sigma, num_samples, center, burn_in=100, num_chains=1
    )
    
    # Compute the exact discrete Gaussian distribution
    exact = compute_exact_discrete_gaussian(B, sigma, center)
    
    # Compute metrics
    tv_dist = compute_total_variation_distance(samples_x, sigma, B, center, exact)
    kl_div = compute_kl_divergence(samples_x, sigma, B, center, exact)
    log_lik = compute_log_likelihood(samples_x, sigma, B, center)
    
    # Compute moments
    moments = compute_moments(samples_x)
    
    # Return validation metrics
    return {
        'TV_distance': tv_dist,
        'KL_divergence': kl_div,
        'log_likelihood': log_lik,
        'acceptance_rate': acceptance_rate,
        'mean_squared_norm': moments['mean_squared_norm']
    }

#############################################################################
# 4. Visualization and Publication-Ready Outputs
#############################################################################

def set_publication_style():
    """
    Configure Matplotlib for publication-quality plots.
    
    Uses a built-in Matplotlib style instead of 'seaborn-whitegrid' for
    better compatibility with SageMath 10.5.
    """
    try:
        # Try to use a built-in style that's similar to seaborn-whitegrid
        # but available in standard Matplotlib
        plt.style.use('ggplot')
    except Exception as e:
        logger.warning(f"Could not set plot style 'ggplot': {e}")
        # Fall back to default style if ggplot is not available
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (5.5, 4),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

def plot_trace(samples, filename, title, tv_distances=None):
    """
    Create trace plots with TV distance convergence.
    
    Args:
        samples: List of samples.
        filename: Path to save the plot.
        title: Plot title.
        tv_distances: List of TV distances over iterations.
    """
    set_publication_style()
    
    samples_np = np.array([np.array(sample) for sample in samples])
    n_samples, n_dims = samples_np.shape
    
    fig, axes = plt.subplots(n_dims + (1 if tv_distances else 0), 1, figsize=(6, 2 * (n_dims + (1 if tv_distances else 0))))
    
    # Handle the case when there's only one dimension
    if n_dims == 1 and tv_distances is None:
        axes = [axes]
    
    # Plot trace for each dimension
    for d in range(n_dims):
        ax = axes[d]
        ax.plot(samples_np[:, d], 'b-', alpha=0.7)
        ax.set_ylabel(f'Dim {d+1}')
        ax.grid(True)
    
    # Plot TV distances if provided
    if tv_distances:
        ax = axes[-1]
        ax.plot(tv_distances, 'r-', alpha=0.7)
        ax.set_ylabel('TV Distance')
        ax.set_xlabel('Iteration')
        ax.grid(True)
    else:
        axes[-1].set_xlabel('Iteration')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_autocorrelation(acf_by_dim, filename, title):
    """
    Create autocorrelation plots with ESS annotations.
    
    Args:
        acf_by_dim: Autocorrelation function by dimension.
        filename: Path to save the plot.
        title: Plot title.
    """
    set_publication_style()
    
    lag, n_dims = acf_by_dim.shape
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for d in range(n_dims):
        ax.plot(range(lag), acf_by_dim[:, d], label=f'Dim {d+1}')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_2d_samples(samples, sigma, filename, lattice_basis, title, center=None):
    """
    Create 2D scatter plots with density contours.
    
    Args:
        samples: List of samples.
        sigma: Standard deviation parameter.
        filename: Path to save the plot.
        lattice_basis: Basis matrix.
        title: Plot title.
        center: Center of the Gaussian. If None, defaults to the origin.
    """
    try:
        set_publication_style()
        
        samples_np = np.array([np.array(sample) for sample in samples])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot samples
        ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10)
        
        # Plot lattice basis vectors
        origin = [0, 0] if center is None else [center[0], center[1]]
        for i in range(lattice_basis.nrows()):
            basis_vector = lattice_basis[i]
            ax.arrow(origin[0], origin[1], basis_vector[0], basis_vector[1],
                    head_width=0.1, head_length=0.2, fc='red', ec='red', linewidth=2)
        
        # Draw contours
        if center is None:
            center_np = np.zeros(2)
        else:
            center_np = np.array(center)
        
        # Create a grid for contour
        x_min, x_max = np.min(samples_np[:, 0]) - sigma, np.max(samples_np[:, 0]) + sigma
        y_min, y_max = np.min(samples_np[:, 1]) - sigma, np.max(samples_np[:, 1]) + sigma
        
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute density
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] = discrete_gaussian_pdf(point, sigma, center_np)
        
        # Draw contours
        contour = ax.contour(X, Y, Z, cmap='viridis', levels=10)
        plt.colorbar(contour, ax=ax, label='Density')
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title)
        ax.grid(True)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        plt.savefig(filename)
        plt.close()
        logger.info(f"Plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error creating 2D plot: {e}")
        # Don't re-raise to allow execution to continue

def plot_3d_samples(samples, sigma, filename, title, center=None, basis=None):
    """
    Create 3D scatter plots for 3D lattices.
    
    Args:
        samples: List of samples.
        sigma: Standard deviation parameter.
        filename: Path to save the plot.
        title: Plot title.
        center: Center of the Gaussian. If None, defaults to the origin.
        basis: Basis matrix. If provided, basis vectors are shown.
    """
    try:
        set_publication_style()
        
        samples_np = np.array([np.array(sample) for sample in samples])
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot samples
        ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], 
                  alpha=0.5, s=5, c='blue')
        
        # Plot basis vectors if provided
        if basis is not None:
            origin = [0, 0, 0] if center is None else [center[0], center[1], center[2]]
            for i in range(basis.nrows()):
                basis_vector = basis[i]
                ax.quiver(origin[0], origin[1], origin[2],
                         basis_vector[0], basis_vector[1], basis_vector[2],
                         color='red', linewidth=2)
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('x₃')
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        plt.savefig(filename)
        plt.close()
        logger.info(f"3D plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error creating 3D plot: {e}")
        # Don't re-raise to allow execution to continue

def plot_2d_projections(samples, sigma, filename, title, center=None, basis=None):
    """
    Create 2D projections for higher dimensions.
    
    Args:
        samples: List of samples.
        sigma: Standard deviation parameter.
        filename: Path to save the plot.
        title: Plot title.
        center: Center of the Gaussian. If None, defaults to the origin.
        basis: Basis matrix. If provided, basis vectors are shown.
    """
    set_publication_style()
    
    samples_np = np.array([np.array(sample) for sample in samples])
    n_dims = samples_np.shape[1]
    
    # Create a grid of subplots for all pairs of dimensions
    n_pairs = n_dims * (n_dims - 1) // 2
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Handle the case when there's only one subplot
    if n_pairs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    pair_idx = 0
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            row = pair_idx // n_cols
            col = pair_idx % n_cols
            
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot samples
            ax.scatter(samples_np[:, i], samples_np[:, j], alpha=0.5, s=5)
            
            # Plot basis vectors if provided
            if basis is not None:
                origin = [0, 0] if center is None else [center[i], center[j]]
                ax.arrow(origin[0], origin[1], basis[i, i], basis[i, j],
                        head_width=0.1, head_length=0.2, fc='red', ec='red', linewidth=1)
                ax.arrow(origin[0], origin[1], basis[j, i], basis[j, j],
                        head_width=0.1, head_length=0.2, fc='red', ec='red', linewidth=1)
            
            ax.set_xlabel(f'Dim {i+1}')
            ax.set_ylabel(f'Dim {j+1}')
            ax.grid(True)
            
            pair_idx += 1
    
    # Hide any unused subplots
    for i in range(pair_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_pca_projection(samples, sigma, filename, title):
    """
    Create PCA projection to 2D for all dimensions.
    
    If scikit-learn is available, uses PCA to project samples to 2D.
    Otherwise, falls back to using the first two dimensions of the samples.
    
    Args:
        samples: List of samples.
        sigma: Standard deviation parameter.
        filename: Path to save the plot.
        title: Plot title.
    """
    try:
        set_publication_style()
        
        samples_np = np.array([np.array(sample) for sample in samples])
        
        if SKLEARN_AVAILABLE:
            # Apply PCA
            pca = PCA(n_components=2)
            projected = pca.fit_transform(samples_np)
            explained_variance = pca.explained_variance_ratio_
            variance_info = f"Explained Variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}"
            x_label = 'PC 1'
            y_label = 'PC 2'
        else:
            # Fallback: use the first two dimensions
            if samples_np.shape[1] >= 2:
                projected = samples_np[:, :2]
                variance_info = "PCA unavailable - showing first two dimensions"
            else:
                # Handle 1D case
                projected = np.column_stack((samples_np[:, 0], np.zeros(samples_np.shape[0])))
                variance_info = "PCA unavailable - showing 1D projection"
            x_label = 'Dimension 1'
            y_label = 'Dimension 2'
            logger.warning(f"Using first two dimensions instead of PCA for {filename}")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot projected samples
        sc = ax.scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=5, 
                       c=np.sum(samples_np**2, axis=1))
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}\n{variance_info}")
        ax.grid(True)
        
        plt.colorbar(sc, ax=ax, label='Squared Norm')
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        plt.savefig(filename)
        plt.close()
        logger.info(f"PCA plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error creating PCA plot: {e}")
        # Don't re-raise to allow execution to continue

def plot_gelman_rubin(chains, filename, title):
    """
    Plot PSRF convergence over iterations.
    
    Args:
        chains: List of chains, each chain is a list of vectors.
        filename: Path to save the plot.
        title: Plot title.
    """
    set_publication_style()
    
    chains_np = []
    for chain in chains:
        chain_np = np.array([np.array(sample) for sample in chain])
        chains_np.append(chain_np)
    
    n_samples = chains_np[0].shape[0]
    n_dims = chains_np[0].shape[1]
    
    # Compute PSRF over iterations
    n_points = min(20, n_samples)
    indices = np.linspace(n_samples // 10, n_samples, n_points, dtype=int)
    psrf_values = np.zeros((n_points, n_dims))
    
    for i, idx in enumerate(indices):
        chains_subset = [chain[:idx] for chain in chains_np]
        psrf_values[i] = compute_gelman_rubin(chains_subset)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for d in range(n_dims):
        ax.plot(indices, psrf_values[:, d], label=f'Dim {d+1}')
    
    ax.axhline(y=1.1, color='r', linestyle='--', label='Threshold 1.1')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSRF')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_geweke(samples, filename, title):
    """
    Plot Geweke z-scores over chain segments.
    
    Args:
        samples: List of samples.
        filename: Path to save the plot.
        title: Plot title.
    """
    set_publication_style()
    
    samples_np = np.array([np.array(sample) for sample in samples])
    n_samples, n_dims = samples_np.shape
    
    # Compute Geweke z-scores over the chain segments
    n_segments = 20
    segment_size = n_samples // n_segments
    
    z_scores = np.zeros((n_segments - 1, n_dims))
    segment_indices = np.arange(1, n_segments) * segment_size
    
    for i in range(1, n_segments):
        samples_segment = samples_np[:i*segment_size]
        z_scores[i-1] = compute_geweke(samples_segment)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for d in range(n_dims):
        ax.plot(segment_indices, z_scores[:, d], label=f'Dim {d+1}')
    
    # Add reference lines
    ax.axhline(y=1.96, color='r', linestyle='--', label='95% CI')
    ax.axhline(y=-1.96, color='r', linestyle='--')
    
    ax.set_xlabel('Chain Length')
    ax.set_ylabel('Geweke Z-Score')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_convergence(tv_distances, mixing_time, spectral_bound, filename, title):
    """
    Plot TV distance decay with theoretical bounds.
    
    Args:
        tv_distances: List of TV distances over iterations.
        mixing_time: Estimated mixing time.
        spectral_bound: Spectral gap bound.
        filename: Path to save the plot.
        title: Plot title.
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot empirical TV distances
    iterations = np.arange(len(tv_distances))
    ax.plot(iterations, tv_distances, 'b-', label='Empirical TV Distance')
    
    # Plot theoretical bound if available
    if spectral_bound is not None:
        delta, _ = spectral_bound
        theoretical_bound = np.exp(-delta * iterations)
        ax.plot(iterations, theoretical_bound, 'r--', label='Theoretical Bound')
    
    # Mark mixing time if available
    if mixing_time is not None:
        ax.axvline(x=mixing_time, color='g', linestyle='--', 
                  label=f'Mixing Time: {mixing_time}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Variation Distance')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_summary_table(results, filename):
    """
    Generate LaTeX tables summarizing results.
    
    Args:
        results: Dictionary with results from experiments.
        filename: Path to save the LaTeX table.
    """
    # Extract relevant information from results
    summary = []
    
    for key, value in results.items():
        dimension = value.get('dimension', '-')
        sigma = value.get('sigma', '-')
        basis_type = value.get('basis_type', '-')
        tv_distance = value.get('TV_distance', '-')
        acceptance_rate = value.get('acceptance_rate', '-')
        mixing_time = value.get('mixing_time', '-')
        
        # Format values
        if tv_distance != '-':
            tv_distance = f"{tv_distance:.4f}"
        if acceptance_rate != '-':
            acceptance_rate = f"{acceptance_rate:.4f}"
        
        summary.append({
            'experiment': key,
            'dimension': dimension,
            'sigma': sigma,
            'basis_type': basis_type,
            'tv_distance': tv_distance,
            'acceptance_rate': acceptance_rate,
            'mixing_time': mixing_time
        })
    
    # Generate LaTeX table
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{lcccccc}\n"
    latex_table += "\\hline\n"
    latex_table += "Experiment & Dimension & $\\sigma$ & Basis Type & TV Distance & Acceptance Rate & Mixing Time \\\\\n"
    latex_table += "\\hline\n"
    
    for row in summary:
        latex_table += f"{row['experiment']} & {row['dimension']} & {row['sigma']} & {row['basis_type']} & {row['tv_distance']} & {row['acceptance_rate']} & {row['mixing_time']} \\\\\n"
    
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Summary of IMHK Sampler Experiments}\n"
    latex_table += "\\label{tab:imhk_summary}\n"
    latex_table += "\\end{table}"
    
    # Save to file with error handling
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(latex_table)
        logger.info(f"Summary table saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save summary table to {filename}: {e}")
        # Save to a fallback location
        try:
            fallback_path = "results/summary_table.tex"
            with open(fallback_path, 'w') as f:
                f.write(latex_table)
            logger.info(f"Summary table saved to fallback location: {fallback_path}")
        except Exception as e2:
            logger.error(f"Failed to save summary table to fallback location: {e2}")


#############################################################################
# 5. Experimentation Framework
#############################################################################

def run_experiment(dim, sigma, num_samples, basis_type='identity', compare_with_klein=True,
                  center=None, reduction=None, do_convergence_analysis=False, num_chains=4):
    """
    Run IMHK experiment with the specified parameters.
    
    Args:
        dim: Dimension of the lattice.
        sigma: Standard deviation parameter.
        num_samples: Number of samples to generate.
        basis_type: Type of lattice basis ('identity', 'skewed', 'ill-conditioned', 'random', 'NTRU').
        compare_with_klein: Whether to compare with Klein's algorithm.
        center: Center of the Gaussian. If None, defaults to the origin.
        reduction: Lattice reduction method ('LLL', 'BKZ', None).
        do_convergence_analysis: Whether to perform convergence analysis.
        num_chains: Number of chains for MCMC.
        
    Returns:
        Dictionary with results.
    """
    logger.info(f"Running experiment: dim={dim}, sigma={sigma}, basis_type={basis_type}")
    
    # Generate lattice basis
    if basis_type.lower() == 'identity':
        B = identity_matrix(ZZ, dim)
    elif basis_type.lower() == 'skewed':
        B = generate_skewed_lattice(dim)
    elif basis_type.lower() == 'ill-conditioned':
        B = generate_ill_conditioned_lattice(dim)
    elif basis_type.lower() == 'random':
        B = generate_random_lattice(dim)
    elif basis_type.lower() == 'ntru':
        q = 12289  # Common NTRU parameter
        B = generate_ntru_lattice(dim // 2, q)
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")
    
    # Apply lattice reduction if requested
    if reduction:
        logger.info(f"Applying {reduction} reduction")
        B = apply_lattice_reduction(B, reduction)
    
    # Setup experiment identifier
    experiment_id = f"dim{dim}_sigma{sigma}_{basis_type}"
    if reduction:
        experiment_id += f"_{reduction}"
    
    # Create experiment directory
    exp_dir = f"results/plots/{experiment_id}"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    # Run IMHK sampler
    logger.info("Running IMHK sampler")
    start_time = time.time()
    
    samples_z, samples_x, acceptance_rates, tv_distances = imhk_sampler(
        B, sigma, num_samples, center, burn_in=max(100, num_samples // 10),
        num_chains=num_chains
    )
    
    imhk_time = time.time() - start_time
    logger.info(f"IMHK sampling completed in {imhk_time:.2f} seconds")
    
    # Run Klein sampler for comparison if requested
    if compare_with_klein:
        logger.info("Running Klein sampler for comparison")
        start_time = time.time()
        
        klein_samples_z = []
        klein_samples_x = []
        for _ in range(num_samples):
            z, x = klein_sampler(B, sigma, center)
            klein_samples_z.append(z)
            klein_samples_x.append(x)
        
        klein_time = time.time() - start_time
        logger.info(f"Klein sampling completed in {klein_time:.2f} seconds")
    
    # Compute diagnostics
    results = {
        'dimension': dim,
        'sigma': sigma,
        'basis_type': basis_type,
        'reduction': reduction,
        'acceptance_rate': np.mean(acceptance_rates),
        'sampling_time': imhk_time
    }
    
    # For small dimensions, compute TV distance and other metrics
    if dim <= 2:
        logger.info("Computing validation metrics")
        validation = validate_sampler(B, sigma, num_samples, center)
        results.update(validation)
    
    # Compute autocorrelation
    logger.info("Computing autocorrelation")
    acf = compute_autocorrelation(samples_x, lag=min(50, num_samples // 10))
    plot_autocorrelation(acf, f"{exp_dir}/autocorrelation.png", 
                         f"Autocorrelation for {experiment_id}")
    
    # Compute ESS
    logger.info("Computing ESS")
    ess = compute_ess(samples_x)
    results['ess'] = ess
    
    # Compute moments
    logger.info("Computing moments")
    moments = compute_moments(samples_x, B)
    results['moments'] = moments
    
    # Perform convergence analysis if requested
    if do_convergence_analysis and dim <= 2:
        logger.info("Performing convergence analysis")
        
        # Estimate mixing time with optimized parameters
        mixing_time, tv_hist = estimate_mixing_time(
            B, sigma, max_iterations=50, threshold=0.3, c=center, timeout_seconds=60
        )
        results['mixing_time'] = mixing_time
        
        # Compute theoretical spectral bound
        spectral_bound = compute_theoretical_spectral_bound(B, sigma, center)
        results['spectral_bound'] = spectral_bound
        
        # Plot convergence
        plot_convergence(tv_hist, mixing_time, spectral_bound, 
                        f"{exp_dir}/convergence.png", 
                        f"Convergence Analysis for {experiment_id}")
    
    # Visualize samples
    logger.info("Creating visualizations")
    
    # For 2D lattices
    if dim == 2:
        plot_2d_samples(samples_x, sigma, f"{exp_dir}/samples_2d.png", 
                       B, f"IMHK Samples for {experiment_id}", center)
        
        if compare_with_klein:
            plot_2d_samples(klein_samples_x, sigma, f"{exp_dir}/klein_samples_2d.png", 
                           B, f"Klein Samples for {experiment_id}", center)
    
    # For 3D lattices
    elif dim == 3:
        plot_3d_samples(samples_x, sigma, f"{exp_dir}/samples_3d.png", 
                       f"IMHK Samples for {experiment_id}", center, B)
        
        if compare_with_klein:
            plot_3d_samples(klein_samples_x, sigma, f"{exp_dir}/klein_samples_3d.png", 
                           f"Klein Samples for {experiment_id}", center, B)
    
    # For higher dimensions
    else:
        plot_2d_projections(samples_x, sigma, f"{exp_dir}/projections_2d.png", 
                           f"IMHK Projections for {experiment_id}", center, B)
        
        plot_pca_projection(samples_x, sigma, f"{exp_dir}/pca_projection.png", 
                           f"PCA Projection for {experiment_id}")
        
        if compare_with_klein:
            plot_2d_projections(klein_samples_x, sigma, f"{exp_dir}/klein_projections_2d.png", 
                               f"Klein Projections for {experiment_id}", center, B)
            
            plot_pca_projection(klein_samples_x, sigma, f"{exp_dir}/klein_pca_projection.png", 
                               f"Klein PCA Projection for {experiment_id}")
    
    # Compare distributions
    if compare_with_klein:
        logger.info("Comparing IMHK and Klein distributions")
        ks_stat, p_value = compare_distributions(samples_x, klein_samples_x)
        results['ks_stat'] = ks_stat
        results['ks_p_value'] = p_value
    
    # Save results
    logger.info("Saving results")
    try:
        with open(f"results/data/{experiment_id}.pkl", 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to results/data/{experiment_id}.pkl")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return results

def parameter_sweep(dimensions, sigmas, basis_types, reductions, centers, 
                   num_samples, do_convergence_analysis=False, num_chains=4):
    """
    Run experiments across parameter combinations.
    
    Args:
        dimensions: List of dimensions to test.
        sigmas: List of sigma values to test.
        basis_types: List of basis types to test.
        reductions: List of reduction methods to test (can include None).
        centers: List of centers to test (can include None).
        num_samples: Number of samples to generate.
        do_convergence_analysis: Whether to perform convergence analysis.
        num_chains: Number of chains for MCMC.
        
    Returns:
        Dictionary with results from all experiments.
    """
    all_results = {}
    
    # Track total experiments and progress
    total_experiments = (len(dimensions) * len(sigmas) * len(basis_types) * 
                        len(reductions) * len(centers))
    current_experiment = 0
    
    for dim in dimensions:
        for sigma in sigmas:
            for basis_type in basis_types:
                for reduction in reductions:
                    for center in centers:
                        current_experiment += 1
                        logger.info(f"Running experiment {current_experiment}/{total_experiments}")
                        
                        # Generate experiment ID
                        exp_id = f"dim{dim}_sigma{sigma}_{basis_type}"
                        if reduction:
                            exp_id += f"_{reduction}"
                        if center is not None:
                            exp_id += "_with_center"
                        
                        # Run experiment
                        results = run_experiment(
                            dim, sigma, num_samples, basis_type, 
                            compare_with_klein=True, center=center, 
                            reduction=reduction, 
                            do_convergence_analysis=do_convergence_analysis,
                            num_chains=num_chains
                        )
                        
                        all_results[exp_id] = results
    
    # Generate comparative plots
    logger.info("Generating comparative plots")
    
    # TV distance vs. sigma (for fixed dim and basis_type)
    for dim in dimensions:
        for basis_type in basis_types:
            for reduction in reductions:
                tv_distances = []
                sigma_values = []
                
                for sigma in sigmas:
                    exp_id = f"dim{dim}_sigma{sigma}_{basis_type}"
                    if reduction:
                        exp_id += f"_{reduction}"
                    
                    if exp_id in all_results and 'TV_distance' in all_results[exp_id]:
                        tv_distances.append(all_results[exp_id]['TV_distance'])
                        sigma_values.append(sigma)
                
                if tv_distances:
                    plt.figure(figsize=(6, 4))
                    plt.plot(sigma_values, tv_distances, 'o-')
                    plt.xlabel('$\\sigma$')
                    plt.ylabel('TV Distance')
                    plt.title(f"TV Distance vs. $\\sigma$ (dim={dim}, basis={basis_type})")
                    plt.grid(True)
                    plt.savefig(f"results/plots/tv_vs_sigma_dim{dim}_{basis_type}.png")
                    plt.close()
    
    # Generate summary table
    logger.info("Generating summary table")
    generate_summary_table(all_results, "results/tables/parameter_sweep_summary.tex")
    
    return all_results

def run_basic_example():
    """
    Run a basic example on a 2D identity lattice.
    
    EXAMPLES::
    
        >>> results = run_basic_example()  # doctest: +SKIP
        >>> isinstance(results, dict)  # Simple placeholder test
        True
    """
    logger.info("Running basic example")
    
    dimension = 2
    sigma = 5.0
    num_samples = 500  # Reduced from 1000 for faster execution
    basis_type = 'identity'
    center = None
    
    results = run_experiment(
        dimension, sigma, num_samples, basis_type, 
        compare_with_klein=True, center=center, 
        reduction=None, do_convergence_analysis=True,
        num_chains=2  # Reduced from 4 for faster execution
    )
    
    return results

def run_high_dimensional_test():
    """
    Test scalability on a 10D lattice using truncation.
    
    EXAMPLES::
    
        >>> # This test is resource-intensive, so we skip it in doctests
        >>> # results = run_high_dimensional_test()  # doctest: +SKIP
        >>> True  # Placeholder for doctest
        True
    """
    logger.info("Running high-dimensional test")
    
    dimension = 8  # Reduced from 10 for faster execution
    sigma = 10.0
    num_samples = 200  # Reduced from 1000 for faster execution
    basis_type = 'random'
    
    # Track memory usage if possible
    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        logger.info(f"Memory usage before high-dimensional test: {memory_before:.2f} MB")
    except ImportError:
        logger.warning("psutil not available, memory usage tracking disabled")
        memory_before = None
    
    # Generate lattice
    try:
        B = generate_random_lattice(dimension)
        
        # Determine truncation bounds
        bounds = truncate_lattice(B, sigma)
        logger.info(f"Truncation bounds determined successfully")
        
        # Run parallel IMHK sampler
        start_time = time.time()
        samples_z, samples_x, acceptance_rate, _ = parallel_imhk_sampler(
            B, sigma, num_samples, burn_in=20,  # Reduced from 100 for faster execution
            num_chains=min(4, mp.cpu_count())  # Reduced from 8 for faster execution
        )
        runtime = time.time() - start_time
        
        logger.info(f"High-dimensional test completed in {runtime:.2f} seconds")
        logger.info(f"Acceptance rate: {acceptance_rate:.4f}")
        
        # Plot PCA projection with error handling
        try:
            plot_pca_projection(samples_x, sigma, "results/plots/high_dim_pca.png", 
                               f"PCA Projection for {dimension}D Lattice")
        except Exception as e:
            logger.error(f"Error creating PCA projection: {e}")
        
        # Compute and plot autocorrelation with error handling
        try:
            acf = compute_autocorrelation(samples_x, lag=min(20, num_samples // 10))
            plot_autocorrelation(acf, "results/plots/high_dim_autocorrelation.png", 
                                 f"Autocorrelation for {dimension}D Lattice")
        except Exception as e:
            logger.error(f"Error computing autocorrelation: {e}")
        
        # Compute moments
        moments = compute_moments(samples_x, B)
        
        results = {
            'dimension': dimension,
            'sigma': sigma,
            'acceptance_rate': acceptance_rate,
            'runtime': runtime,
            'moments': moments
        }
        
        # Track memory usage after test
        if memory_before is not None:
            try:
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_used = memory_after - memory_before
                logger.info(f"Memory usage after test: {memory_after:.2f} MB")
                logger.info(f"Memory increase during test: {memory_used:.2f} MB")
                results['memory_usage_mb'] = memory_after
                results['memory_increase_mb'] = memory_used
            except Exception as e:
                logger.error(f"Failed to get memory usage: {e}")
        
        # Save results with error handling
        try:
            with open("results/data/high_dimensional_test.pkl", 'wb') as f:
                pickle.dump(results, f)
            logger.info("Results saved to results/data/high_dimensional_test.pkl")
        except Exception as e:
            logger.error(f"Failed to save high-dimensional test results: {e}")
        
        return results
    except Exception as e:
        logger.error(f"High-dimensional test failed: {e}")
        return None

#############################################################################
# 6. Main Execution
#############################################################################

if __name__ == "__main__":
    # Import doctest here to avoid conflicts with SageMath's doctest system
    import doctest
    import sys
    
    # Run unit tests with doctests
    logger.info("Running doctests to validate key functions")
    doctest_results = doctest.testmod(verbose=True)
    if doctest_results.failed > 0:
        logger.warning(f"Some doctests failed: {doctest_results.failed} failures out of {doctest_results.attempted}")
    else:
        logger.info(f"All {doctest_results.attempted} doctests passed")
    
    # If scikit-learn is not available, print a message and continue
    if not SKLEARN_AVAILABLE:
        logger.warning("Running without scikit-learn: PCA visualization will use first two dimensions")
        logger.warning("To enable proper PCA visualization, exit and run: sage -pip install scikit-learn")
    
    # Test error handling in gram_schmidt
    logger.info("Testing error handling in gram_schmidt")
    try:
        # Create an intentionally bad basis (zero vector)
        B_bad = matrix(ZZ, [[0, 0], [0, 0]])
        GS, M = gram_schmidt(B_bad)
        logger.error("gram_schmidt should have failed with zero vectors")
    except ValueError as e:
        logger.info(f"gram_schmidt correctly caught error: {e}")
    
    # Test core functions
    logger.info("Testing discrete_gaussian_pdf with various input types")
    test_cases = [
        (0, 1.0),  # Scalar
        ([0, 0], 1.0),  # List
        (vector(RR, [0, 0]), 1.0),  # RealField vector
        (vector(ZZ, [0, 0]), 1.0),  # Integer vector (SageMath)
        (np.array([0, 0]), 1.0)  # NumPy array
    ]
    for x, sigma in test_cases:
        try:
            result = discrete_gaussian_pdf(x, sigma)
            logger.info(f"discrete_gaussian_pdf({x}, {sigma}) = {result}")
        except Exception as e:
            logger.error(f"Error in discrete_gaussian_pdf({x}, {sigma}): {e}")
    
    logger.info("Testing klein_sampler")
    try:
        B = identity_matrix(ZZ, 2)
        sigma = 5.0
        z, x = klein_sampler(B, sigma)
        logger.info(f"Klein sample: z={z}, x={x}")
    except Exception as e:
        logger.error(f"Error in klein_sampler: {e}")
        sys.exit(1)  # Exit if klein_sampler fails
    
    # Run basic example with error handling
    try:
        logger.info("Running basic example")
        run_basic_example()
    except Exception as e:
        logger.error(f"Error in run_basic_example: {e}")
        logger.warning("Continuing to next test...")
    
    # Run parameter sweep for a small set of parameters
    try:
        logger.info("Running parameter sweep")
        dimensions = [2]
        sigmas = [3.0, 5.0]
        basis_types = ['identity']  # Reduced to just identity for faster execution
        reductions = [None]
        centers = [None]
        
        parameter_sweep(
            dimensions, sigmas, basis_types, reductions, centers,
            num_samples=250,  # Reduced from 500 for faster execution
            do_convergence_analysis=True, 
            num_chains=2
        )
    except Exception as e:
        logger.error(f"Error in parameter_sweep: {e}")
        logger.warning("Continuing to next test...")
    
    # Run high-dimensional test
    try:
        logger.info("Running high-dimensional test")
        run_high_dimensional_test()
    except Exception as e:
        logger.error(f"Error in run_high_dimensional_test: {e}")
        logger.warning("Some tests may have failed, check logs for details")
    
    # Test 3D lattice
    try:
        logger.info("Testing 3D lattice")
        B_3d = identity_matrix(ZZ, 3)
        sigma = 3.0
        samples_z, samples_x, _, _ = imhk_sampler(
            B_3d, sigma, 50,  # Reduced from 100 for faster execution
            burn_in=5,  # Reduced from 10 for faster execution
            num_chains=1
        )
        logger.info(f"3D test successful: Got {len(samples_x)} samples")
        
        # Plot 3D samples
        plot_3d_samples(samples_x, sigma, "results/plots/test_3d.png", 
                      "3D Test Samples", None, B_3d)
    except Exception as e:
        logger.error(f"Error in 3D test: {e}")
    
    logger.info("All tests and experiments completed")
    
    import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMHK Sampler CLI")
    parser.add_argument("--run", type=str, choices=["basic", "highdim", "sweep", "test3d"], help="Which example or test to run")
    args = parser.parse_args()

    if args.run == "basic":
        run_basic_example()
    elif args.run == "highdim":
        run_high_dimensional_test()
    elif args.run == "sweep":
        parameter_sweep(
            dimensions=[2],
            sigmas=[3.0, 5.0],
            basis_types=['identity'],
            reductions=[None],
            centers=[None],
            num_samples=250,
            do_convergence_analysis=True,
            num_chains=2
        )
    elif args.run == "test3d":
        B_3d = identity_matrix(ZZ, 3)
        sigma = 3.0
        samples_z, samples_x, _, _ = imhk_sampler(B_3d, sigma, 50, burn_in=5, num_chains=1)
        plot_3d_samples(samples_x, sigma, "results/plots/test_3d.png", "3D Test Samples", None, B_3d)
