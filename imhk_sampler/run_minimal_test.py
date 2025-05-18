#!/usr/bin/env sage
"""
Minimal smoke test for IMHK sampler functionality.

This script runs a quick test to verify all components are working correctly.
It should complete in under 10 seconds and create a few diagnostic plots.
"""

import os
import sys
import time
import logging
import numpy as np
from sage.all import *

# Add the imhk_sampler directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from samplers import imhk_sampler, klein_sampler
from diagnostics import compute_autocorrelation, compute_ess, plot_trace, plot_autocorrelation
from stats import tv_distance_discrete_gaussian
from visualization import plot_2d_samples
from experiments import create_lattice_basis, init_directories

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_smoke_test():
    """Run minimal smoke test to verify all components work."""
    start_time = time.time()
    
    # Initialize directories
    init_directories("results/smoke_test")
    
    logger.info("Starting IMHK sampler smoke test...")
    
    # Test 1: Basic 2D example with visualization
    logger.info("Test 1: Basic 2D example")
    try:
        dim = 2
        sigma = 1.0
        num_samples = 500
        
        # Create identity basis
        B = create_lattice_basis(dim, 'identity')
        center = vector(RDF, [0, 0])
        
        # Run IMHK sampler
        samples, acceptance_rate, trace, acceptance_trace = imhk_sampler(
            B=B, 
            sigma=sigma, 
            num_samples=num_samples,
            center=center
        )
        
        logger.info(f"  Acceptance rate: {acceptance_rate:.3f}")
        
        # Compute diagnostics
        if trace is not None:
            # Ensure the trace is a numpy array
            trace_array = np.array(trace)
            
            # Compute ESS and handle potential list conversion
            try:
                autocorr = compute_autocorrelation(trace_array)
                ess = compute_ess(trace_array)
                # Use np.mean to handle both array and list cases
                logger.info(f"  Average ESS: {np.mean(ess):.1f}")
            except Exception as e:
                logger.error(f"  Failed to compute ESS: {e}")
        
        # Create visualization
        plot_2d_samples(samples, sigma, "results/smoke_test/test_2d_samples.png")
        
        # Compute TV distance
        # Ensure samples is a numpy array for TV distance calculation
        samples_array = np.array([list(s) for s in samples])
        tv_dist = tv_distance_discrete_gaussian(B, sigma, samples_array)
        logger.info(f"  TV distance: {tv_dist:.6f}")
        
        logger.info("  ✓ Test 1 passed")
        
    except Exception as e:
        logger.error(f"  ✗ Test 1 failed: {e}")
        return False
    
    # Test 2: All basis types
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    for basis_type in basis_types:
        logger.info(f"Test 2: {basis_type} basis")
        try:
            dim = 4
            sigma = 2.0
            num_samples = 300
            
            # Create basis
            B = create_lattice_basis(dim, basis_type)
            center = vector(RDF, [0] * dim)
            
            # Run IMHK sampler
            samples, acceptance_rate, trace, acceptance_trace = imhk_sampler(
                B=B,
                sigma=sigma,
                num_samples=num_samples,
                center=center
            )
            
            logger.info(f"  Acceptance rate: {acceptance_rate:.3f}")
            
            # Compute TV distance
            # Ensure samples is a numpy array for TV distance calculation
            samples_array = np.array([list(s) for s in samples])
            tv_dist = tv_distance_discrete_gaussian(B, sigma, samples_array, max_radius=3)
            logger.info(f"  TV distance: {tv_dist:.6f}")
            
            logger.info(f"  ✓ {basis_type} basis test passed")
            
        except Exception as e:
            logger.error(f"  ✗ {basis_type} basis test failed: {e}")
            return False
    
    # Test 3: Klein sampler comparison
    logger.info("Test 3: Klein sampler comparison")
    try:
        dim = 3
        sigma = 1.5
        num_samples = 100
        
        B = create_lattice_basis(dim, 'identity')
        center = vector(RDF, [0] * dim)
        
        # Run both samplers
        imhk_samples, imhk_rate, _, _ = imhk_sampler(B=B, sigma=sigma, num_samples=num_samples, center=center)
        
        klein_samples = []
        for _ in range(num_samples):
            klein_samples.append(klein_sampler(B=B, sigma=sigma, center=center))
        
        # Compare TV distances
        # Ensure samples are numpy arrays for TV distance calculation
        imhk_samples_array = np.array([list(s) for s in imhk_samples])
        klein_samples_array = np.array([list(s) for s in klein_samples])
        imhk_tv = tv_distance_discrete_gaussian(B, sigma, imhk_samples_array)
        klein_tv = tv_distance_discrete_gaussian(B, sigma, klein_samples_array)
        
        logger.info(f"  IMHK TV distance: {imhk_tv:.6f}")
        logger.info(f"  Klein TV distance: {klein_tv:.6f}")
        logger.info(f"  Ratio: {imhk_tv/klein_tv:.3f}")
        
        logger.info("  ✓ Klein comparison test passed")
        
    except Exception as e:
        logger.error(f"  ✗ Klein comparison test failed: {e}")
        return False
    
    # Summary
    runtime = time.time() - start_time
    logger.info(f"\nAll tests completed successfully!")
    logger.info(f"Total runtime: {runtime:.2f} seconds")
    
    return True

if __name__ == "__main__":
    logger.info("IMHK Sampler Smoke Test")
    logger.info("=" * 40)
    
    success = run_smoke_test()
    
    print("\n" + "=" * 40)
    print("IMHK Sampler Smoke Test Results")
    print("=" * 40)
    
    if success:
        print("✓ 2D Example")
        print("✓ Identity Basis")
        print("✓ Skewed Basis")
        print("✓ Ill-conditioned Basis")
        print("✓ Klein Comparison")
        print("\nAll tests passed!")
        print(f"Results saved to: results/smoke_test/")
    else:
        print("✗ Some tests failed. Check the log above for details.")
        sys.exit(1)