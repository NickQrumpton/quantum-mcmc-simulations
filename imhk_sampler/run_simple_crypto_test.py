#!/usr/bin/env sage -python
"""
Simple crypto test that works independently of the experiment framework.
"""

import sys
from pathlib import Path
import numpy as np
import logging
import time
from math import sqrt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_crypto_basis(basis_type, dim=16):
    """Test a single cryptographic basis type."""
    logger.info(f"\nTesting {basis_type} basis (dimension {dim})")
    
    try:
        # Import required modules
        from utils import create_lattice_basis
        from samplers import imhk_sampler_wrapper
        from parameter_config import compute_smoothing_parameter
        
        # Create lattice basis
        logger.info(f"Creating {basis_type} lattice...")
        basis_info = create_lattice_basis(dim, basis_type)
        
        # Determine appropriate sigma
        if isinstance(basis_info, tuple):
            # Structured lattice (NTRU, PrimeCyclotomic)
            poly_mod, q = basis_info
            sigma = float(sqrt(q) / 20)
            logger.info(f"Structured lattice: degree={poly_mod.degree()}, q={q}, sigma={sigma:.4f}")
        else:
            # Matrix lattice (identity, q-ary)
            eta = compute_smoothing_parameter(basis_info)
            sigma = max(2.0 * eta, 1.0)  # Ensure reasonable minimum
            logger.info(f"Matrix lattice: eta={eta:.6f}, sigma={sigma:.4f}")
        
        # Run IMHK sampler
        logger.info("Running IMHK sampler...")
        start_time = time.time()
        
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=100,
            burn_in=50,
            basis_type=basis_type
        )
        
        elapsed = time.time() - start_time
        
        # Report results
        logger.info(f"Sampling completed in {elapsed:.2f}s")
        logger.info(f"Generated {samples.shape[0]} samples, shape: {samples.shape}")
        logger.info(f"Acceptance rate: {metadata.get('acceptance_rate', 0):.4f}")
        
        # Test TV distance for matrix lattices only
        if not isinstance(basis_info, tuple) and dim <= 8:
            logger.info("Testing TV distance computation...")
            try:
                from stats import compute_total_variation_distance
                
                tv_dist = compute_total_variation_distance(
                    samples[:50],  # Use limited samples
                    sigma,
                    basis_info,
                    max_radius=2   # Small radius for speed
                )
                logger.info(f"TV distance: {tv_dist:.6f}")
            except Exception as e:
                logger.warning(f"TV distance computation failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple cryptographic tests."""
    logger.info("Simple Cryptographic Basis Test")
    logger.info("="*50)
    
    # Test configurations
    test_cases = [
        ('identity', 8),    # Baseline
        ('q-ary', 8),       # LWE-style
        ('q-ary', 16),      # Larger q-ary
        ('NTRU', 16),       # NTRU lattice
        ('PrimeCyclotomic', 16),  # Prime cyclotomic
    ]
    
    results = []
    total_time = time.time()
    
    for basis_type, dim in test_cases:
        success = test_crypto_basis(basis_type, dim)
        results.append((basis_type, dim, success))
    
    total_elapsed = time.time() - total_time
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    for basis_type, dim, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{basis_type:15} (dim={dim:2}): {status}")
    
    logger.info(f"\nTotal time: {total_elapsed:.2f}s")
    
    # Recommendations
    passed = sum(1 for _, _, s in results if s)
    total = len(results)
    
    logger.info(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        logger.info("\nAll cryptographic bases working correctly!")
        logger.info("\nNext steps:")
        logger.info("1. Run with higher dimensions: --dimensions 32 64")
        logger.info("2. Run full experiments: sage run_smoke_test_crypto.py")
        logger.info("3. Generate publication results: sage generate_crypto_publication_results.py")
    else:
        logger.warning("\nSome tests failed. Check error messages above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())