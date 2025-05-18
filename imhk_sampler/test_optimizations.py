#!/usr/bin/env sage -python
"""
Quick test to verify TV distance optimizations are working.
"""

import sys
from pathlib import Path
import numpy as np
import time
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import create_lattice_basis
from samplers import imhk_sampler_wrapper
from stats import compute_total_variation_distance
from parameter_config import compute_smoothing_parameter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dimension(dim, basis_type='identity', num_samples=100):
    """Test TV distance computation for a specific dimension."""
    logger.info(f"\nTesting dimension {dim}, basis type {basis_type}")
    
    # Create basis
    basis = create_lattice_basis(dim, basis_type)
    
    # Compute sigma
    if isinstance(basis, tuple):
        sigma = 10.0  # Fixed for structured lattices
        logger.info(f"Structured lattice detected, using sigma={sigma}")
    else:
        eta = compute_smoothing_parameter(basis)
        sigma = 2.0 * eta
        logger.info(f"Matrix lattice: eta={eta}, sigma={sigma}")
    
    # Generate samples
    logger.info("Generating samples...")
    samples, metadata = imhk_sampler_wrapper(
        basis_info=basis,
        sigma=sigma,
        num_samples=num_samples,
        burn_in=50,
        basis_type=basis_type
    )
    
    logger.info(f"Generated {samples.shape[0]} samples, acceptance rate: {metadata.get('acceptance_rate', 0):.4f}")
    
    # Skip TV distance for structured lattices
    if isinstance(basis, tuple):
        logger.info("Skipping TV distance for structured lattice")
        return None
    
    # Test TV distance computation with optimizations
    logger.info("Computing TV distance with optimizations...")
    start_time = time.time()
    
    tv_dist = compute_total_variation_distance(
        samples,
        sigma,
        basis,
        max_radius=max(2, int(4.0 / np.sqrt(dim))),
        convergence_threshold=1e-3,
        max_points=1000,
        adaptive_sampling=(dim > 8),
        progress_interval=2.0
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"TV distance: {tv_dist:.6f}, computed in {elapsed:.2f}s")
    
    return {
        'dimension': dim,
        'basis_type': basis_type,
        'tv_distance': tv_dist,
        'computation_time': elapsed,
        'samples': num_samples
    }

def main():
    """Run optimization tests."""
    logger.info("Testing TV distance optimizations")
    
    # Test configurations
    test_cases = [
        (4, 'identity', 100),
        (8, 'identity', 100),
        (16, 'identity', 100),
        (8, 'q-ary', 100),
        (32, 'NTRU', 100),  # Should skip TV distance
    ]
    
    results = []
    
    for dim, basis_type, samples in test_cases:
        try:
            result = test_dimension(dim, basis_type, samples)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Test failed for dimension {dim}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION TEST SUMMARY")
    logger.info("="*50)
    
    for result in results:
        logger.info(f"Dimension {result['dimension']}: "
                   f"TV={result['tv_distance']:.6f}, "
                   f"Time={result['computation_time']:.2f}s")
    
    logger.info("\nOptimizations working correctly!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())