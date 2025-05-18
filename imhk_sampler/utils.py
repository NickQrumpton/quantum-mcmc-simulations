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
    
    Args:
        x: The point to evaluate (can be a vector or scalar)
        sigma: The standard deviation of the distribution
        center: The center of the distribution (default: origin)
        
    Returns:
        The unnormalized probability density exp(-||x - center||² / (2σ²))
        
    Note:
        This returns the unnormalized density. The normalization factor
        can be computed separately when needed.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    # Handle scalar case
    if isinstance(x, (int, float, Integer)):
        x = vector([x])
    elif isinstance(x, (list, tuple, np.ndarray)):
        x = vector(x)
    
    # Handle center
    if center is None:
        center = vector([0] * len(x))
    elif isinstance(center, (int, float, Integer)):
        center = vector([center])
    elif isinstance(center, (list, tuple, np.ndarray)):
        center = vector(center)
    
    # Check dimensions
    if len(x) != len(center):
        raise ValueError(f"Dimension mismatch: x has length {len(x)}, center has length {len(center)}")
    
    try:
        # Compute ||x - center||²
        diff = x - center
        norm_squared = sum(float(d)**2 for d in diff)
        
        # Compute exp(-||x - center||² / (2σ²))
        return exp(-norm_squared / (2 * sigma**2))
        
    except Exception as e:
        logger.error(f"Error computing discrete Gaussian PDF: {e}")
        raise


def create_lattice_basis(dim, basis_type='identity'):
    """
    Create a lattice basis of the specified type for cryptographic experiments.
    
    Cryptographic Relevance:
    Different lattice types model various security scenarios:
    - Identity: Standard orthogonal basis (baseline)
    - q-ary: Models lattices from LWE-based constructions
    - NTRU: NTRU lattices with Falcon parameters for NIST standardization
    - PrimeCyclotomic: Prime cyclotomic lattices (Mitaka-style)
    
    Args:
        dim: Dimension of the lattice (must be ≥ 2)
        basis_type: Type of basis to create (default: 'identity')
                   Options: 'identity', 'q-ary', 'NTRU', 'PrimeCyclotomic'
        
    Returns:
        A SageMath matrix representing the lattice basis, or
        a tuple (polynomial_modulus, prime_modulus) for structured lattices
        
    Raises:
        ValueError: If dimension is invalid or basis_type is unknown
    """
    if not isinstance(dim, (int, Integer)) or dim < 2:
        raise ValueError(f"Dimension must be an integer ≥ 2, got {dim}")
    
    supported_types = ['identity', 'q-ary', 'NTRU', 'PrimeCyclotomic']
    if basis_type not in supported_types:
        raise ValueError(f"Unknown basis type '{basis_type}'. Supported types: {supported_types}")
    
    logger.info(f"Creating {basis_type} lattice basis of dimension {dim}")
    
    if basis_type == 'identity':
        # Standard orthogonal basis
        return identity_matrix(ZZ, dim)
    
    elif basis_type == 'q-ary':
        # q-ary lattice as used in LWE-based cryptography
        # Uses prime modulus based on dimension for cryptographic strength
        q = next_prime(2 ** (int(dim / 2)))
        logger.info(f"Creating q-ary lattice with q = {q}")
        
        # Create a random matrix A and construct the q-ary lattice
        # L_q(A) = {v ∈ Z^n : Av ≡ 0 (mod q)}
        from sage.all import randint
        B = matrix(ZZ, dim, dim)
        
        # Standard q-ary lattice construction
        for i in range(dim):
            B[i, i] = q
            
        # Random linear constraints
        for i in range(int(dim/2)):
            for j in range(int(dim/2), dim):
                B[i, j] = randint(0, q - 1)
                
        return B
    
    elif basis_type == 'NTRU':
        # NTRU lattice with Falcon parameters for NIST post-quantum standards
        q = 12289  # Falcon standard modulus
        N = 512 if dim <= 512 else 1024  # Falcon-512 or Falcon-1024
        
        logger.info(f"Creating NTRU lattice with N = {N}, q = {q}")
        
        # For NTRU, we return the parameters rather than a matrix
        # The sampler will handle the polynomial representation
        R = PolynomialRing(ZZ, 'x')
        x = R.gen()
        poly_mod = x**N + 1  # Cyclotomic polynomial for NTRU
        
        return (poly_mod, q)
    
    elif basis_type == 'PrimeCyclotomic':
        # Prime cyclotomic lattice (Mitaka-style construction)
        # Uses specific parameters for cryptographic applications
        m = 683  # Prime for cyclotomic construction
        q = 1367  # Prime modulus
        
        logger.info(f"Creating Prime Cyclotomic lattice with m = {m}, q = {q}")
        
        R = PolynomialRing(ZZ, 'x')
        x = R.gen()
        poly_mod = x**m - 1  # m-th cyclotomic polynomial
        
        return (poly_mod, q)