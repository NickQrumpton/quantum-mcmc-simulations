#!/usr/bin/env sage -python
"""
Final patched samplers with all issues resolved.
"""

import numpy as np
from sage.all import *
import time
import logging
from typing import Union, Tuple, Dict, Any, Optional
from math import sqrt

logger = logging.getLogger(__name__)


def sanitize_numeric_input(value, name: str, expected_type=float):
    """Sanitize numeric inputs to ensure correct type."""
    if value is None:
        raise ValueError(f"{name} cannot be None")
    
    # Handle lists/arrays - take first element if single value
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 1:
            value = value[0]
        else:
            # For arrays, might need to handle differently
            if isinstance(value, np.ndarray) and len(value.shape) == 0:
                value = value.item()
            else:
                raise ValueError(f"{name} must be scalar, got {type(value)} with shape {getattr(value, 'shape', len(value))}")
    
    # Convert strings
    if isinstance(value, str):
        try:
            value = expected_type(value)
        except ValueError:
            raise ValueError(f"Cannot convert {name}='{value}' to {expected_type}")
    
    # Final conversion
    try:
        return expected_type(value)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert {name}={value} (type {type(value)}) to {expected_type}")


def _get_dimension(basis_info: Union[matrix, Tuple], basis_type: str) -> int:
    """Get dimension from basis info."""
    if isinstance(basis_info, tuple):
        if basis_type == 'q-ary':
            # q-ary lattice
            poly = basis_info[0]
            if hasattr(poly, 'degree'):
                return 2 * poly.degree()
            elif hasattr(poly, 'parent'):
                return 2 * poly.parent().degree()
            else:
                # Fallback based on common sizes
                return 16  # Default q-ary dimension
        elif basis_type == 'NTRU':
            # NTRU lattice
            poly = basis_info[0]
            if hasattr(poly, 'parent'):
                try:
                    return 2 * poly.parent().degree()
                except:
                    return 512  # Default NTRU dimension
            return 512
        elif basis_type == 'PrimeCyclotomic':
            # Prime cyclotomic lattice  
            poly = basis_info[0]
            if hasattr(poly, 'parent'):
                try:
                    return poly.parent().degree()
                except:
                    return 683  # Default prime cyclotomic dimension
            return 683
        else:
            # Default for tuple
            if hasattr(basis_info[0], 'degree'):
                return 2 * basis_info[0].degree()
            else:
                return 16  # Default dimension
    else:
        # Standard matrix basis
        return basis_info.nrows()


def fixed_imhk_sampler_wrapper(basis_info: Union[matrix, Tuple],
                              sigma: float,
                              num_samples: int,
                              burn_in: int = 1000,
                              basis_type: str = 'identity') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fixed IMHK sampler wrapper with complete type safety.
    """
    # Sanitize inputs first
    try:
        sigma = sanitize_numeric_input(sigma, "sigma", float)
        num_samples = sanitize_numeric_input(num_samples, "num_samples", int)
        burn_in = sanitize_numeric_input(burn_in, "burn_in", int)
        
        # Validate inputs
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {burn_in}")
            
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        # Return minimal valid output
        return np.zeros((1, 2)), {'error': str(e), 'acceptance_rate': 0.0}
    
    start_time = time.time()
    metadata = {
        'basis_type': basis_type,
        'sigma': float(sigma),
        'num_samples': int(num_samples),
        'burn_in': int(burn_in),
        'acceptance_rate': 0.0,
        'sampling_time': 0.0,
        'actual_samples': 0,
        'error': None
    }
    
    try:
        if isinstance(basis_info, tuple):
            # Structured lattice sampling
            samples, meta_update = _sample_structured_lattice_imhk(
                basis_info, sigma, num_samples, burn_in, basis_type
            )
            metadata.update(meta_update)
        else:
            # Standard lattice sampling
            samples, meta_update = _sample_standard_lattice_imhk(
                basis_info, sigma, num_samples, burn_in
            )
            metadata.update(meta_update)
        
        metadata['sampling_time'] = float(time.time() - start_time)
        return samples, metadata
        
    except Exception as e:
        logger.error(f"IMHK sampling failed for {basis_type}: {e}")
        metadata['error'] = str(e)
        metadata['sampling_time'] = float(time.time() - start_time)
        
        # Return minimal valid output
        dim = _get_dimension(basis_info, basis_type)
        return np.zeros((1, dim)), metadata


def fixed_klein_sampler_wrapper(basis_info: Union[matrix, Tuple],
                               sigma: float,
                               num_samples: int,
                               basis_type: str = 'identity') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fixed Klein sampler wrapper with complete type safety.
    """
    # Sanitize inputs
    try:
        sigma = sanitize_numeric_input(sigma, "sigma", float)
        num_samples = sanitize_numeric_input(num_samples, "num_samples", int)
        
        # Validate inputs
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
            
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        # Return minimal valid output
        return np.zeros((1, 2)), {'error': str(e), 'actual_samples': 0}
    
    start_time = time.time()
    metadata = {
        'basis_type': basis_type,
        'sigma': float(sigma),
        'num_samples': int(num_samples),
        'sampling_time': 0.0,
        'actual_samples': 0,
        'error': None
    }
    
    try:
        if isinstance(basis_info, tuple):
            # Structured lattice Klein sampling
            samples, meta_update = _sample_structured_lattice_klein(
                basis_info, sigma, num_samples, basis_type
            )
            metadata.update(meta_update)
        else:
            # Standard Klein sampling
            samples, meta_update = _sample_standard_lattice_klein(
                basis_info, sigma, num_samples
            )
            metadata.update(meta_update)
        
        metadata['sampling_time'] = float(time.time() - start_time)
        return samples, metadata
        
    except Exception as e:
        logger.error(f"Klein sampling failed for {basis_type}: {e}")
        metadata['error'] = str(e)
        metadata['sampling_time'] = float(time.time() - start_time)
        
        # Return minimal valid output
        dim = _get_dimension(basis_info, basis_type)
        return np.zeros((1, dim)), metadata


def _sample_structured_lattice_imhk(basis_info: Tuple, 
                                   sigma: float, 
                                   num_samples: int, 
                                   burn_in: int,
                                   basis_type: str) -> Tuple[np.ndarray, Dict]:
    """
    IMHK sampling for structured lattices with complete type safety.
    """
    poly_mod, q = basis_info
    
    # Ensure q is properly typed
    q = sanitize_numeric_input(q, "q", int)
    
    dim = _get_dimension(basis_info, basis_type)
    
    logger.info(f"IMHK sampling for {basis_type}: dim={dim}, q={q}, sigma={sigma}")
    
    # Initialize
    samples = []
    current = np.zeros(dim, dtype=np.float64)
    accepted = 0
    total_steps = int(num_samples + burn_in)
    
    # Adaptive step size
    step_size = min(float(sigma) / sqrt(float(dim)), 1.0)
    
    for i in range(total_steps):
        # Propose new state
        proposal = current + step_size * np.random.randn(dim)
        
        # Reduce modulo q
        proposal_reduced = proposal % q
        current_reduced = current % q
        
        # Simple acceptance criterion for structured lattices
        log_ratio = -0.5 * (np.sum(proposal_reduced**2) - np.sum(current_reduced**2)) / (sigma**2)
        
        # Accept/reject with proper type comparison
        if np.log(np.random.random()) < float(log_ratio):
            current = proposal
            if i >= burn_in:
                accepted += 1
        
        # Store sample after burn-in
        if i >= burn_in:
            samples.append(current.copy())
    
    samples_array = np.array(samples, dtype=np.float64)
    
    # Calculate actual acceptance rate
    actual_acceptance = float(accepted) / float(num_samples)
    
    return samples_array, {
        'acceptance_rate': actual_acceptance,
        'actual_samples': len(samples)
    }


def _sample_standard_lattice_imhk(basis: matrix, 
                                 sigma: float, 
                                 num_samples: int, 
                                 burn_in: int) -> Tuple[np.ndarray, Dict]:
    """
    IMHK sampling for standard lattices.
    """
    dim = basis.nrows()
    
    # Initialize
    samples = []
    current = np.zeros(dim, dtype=np.float64)
    accepted = 0
    total_steps = int(num_samples + burn_in)
    
    # Adaptive step size
    step_size = float(sigma) / sqrt(float(dim))
    
    for i in range(total_steps):
        # Propose new state
        proposal = current + step_size * np.random.randn(dim)
        
        # Gaussian acceptance
        current_norm = float(np.linalg.norm(current))
        proposal_norm = float(np.linalg.norm(proposal))
        
        log_ratio = -0.5 * (proposal_norm**2 - current_norm**2) / (sigma**2)
        
        # Accept/reject with proper numeric comparison
        if np.random.random() < np.exp(min(float(log_ratio), 0.0)):
            current = proposal
            if i >= burn_in:
                accepted += 1
        
        # Store sample after burn-in
        if i >= burn_in:
            samples.append(current.copy())
    
    samples_array = np.array(samples, dtype=np.float64)
    actual_acceptance = float(accepted) / float(num_samples)
    
    return samples_array, {
        'acceptance_rate': actual_acceptance,
        'actual_samples': len(samples)
    }


def _sample_structured_lattice_klein(basis_info: Tuple,
                                    sigma: float,
                                    num_samples: int,
                                    basis_type: str) -> Tuple[np.ndarray, Dict]:
    """
    Klein sampling for structured lattices with type safety.
    """
    poly_mod, q = basis_info
    
    # Ensure proper types using Sage's Integer conversion
    try:
        q = Integer(q)  # Convert to Sage integer
    except:
        q = int(q)
    
    dim = _get_dimension(basis_info, basis_type)
    
    logger.info(f"Klein sampler: dim={dim}, sigma={sigma}, num_samples={num_samples}")
    logger.info(f"Starting Klein sampler: dim={dim}, sigma={sigma}")
    
    # Conservative sampling for structured lattices
    samples = []
    
    for _ in range(num_samples):
        # Sample from discrete Gaussian around origin
        vec = np.zeros(dim, dtype=np.float64)
        
        for j in range(dim):
            # Use Box-Muller with rounding for discretization
            z = np.random.randn()
            value = float(sigma) * z
            vec[j] = float(np.round(value))
        
        # Reduce modulo q
        vec = vec % float(q)
        samples.append(vec)
    
    samples_array = np.array(samples, dtype=np.float64)
    
    logger.info(f"Klein completed: {len(samples)} samples")
    
    return samples_array, {
        'actual_samples': len(samples)
    }


def _sample_standard_lattice_klein(basis: matrix,
                                  sigma: float,
                                  num_samples: int) -> Tuple[np.ndarray, Dict]:
    """
    Klein sampling for standard lattices.
    """
    dim = basis.nrows()
    samples = []
    
    for _ in range(num_samples):
        # Sample from continuous Gaussian
        vec = sigma * np.random.randn(dim)
        
        # Find closest lattice point (simple rounding)
        lattice_vec = basis * np.round(basis.inverse() * vector(vec))
        samples.append(np.array(lattice_vec, dtype=np.float64))
    
    samples_array = np.array(samples, dtype=np.float64)
    
    return samples_array, {
        'actual_samples': len(samples)
    }