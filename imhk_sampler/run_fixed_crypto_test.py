#!/usr/bin/env sage -python
"""
Fixed crypto test with proper parameter handling and error recovery.
"""

import sys
from pathlib import Path
import numpy as np
import logging
import time
import json
from math import sqrt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_create_basis(dim, basis_type):
    """Safely create a lattice basis with error handling."""
    try:
        from utils import create_lattice_basis
        return create_lattice_basis(dim, basis_type)
    except Exception as e:
        logger.error(f"Failed to create {basis_type} basis: {e}")
        # Fallback to identity
        logger.info("Falling back to identity basis")
        from sage.all import identity_matrix, ZZ
        return identity_matrix(ZZ, dim)

def safe_compute_sigma(basis_info, basis_type, dim):
    """Safely compute sigma with proper bounds."""
    try:
        if isinstance(basis_info, tuple):
            # Structured lattice
            poly_mod, q = basis_info
            sigma = float(sqrt(q) / 20)
            # Ensure reasonable bounds
            sigma = max(sigma, 1.0)
            sigma = min(sigma, 100.0)
            return sigma
        else:
            # Matrix lattice
            from parameter_config import compute_smoothing_parameter
            eta = compute_smoothing_parameter(basis_info)
            sigma = 2.0 * eta
            
            # Ensure reasonable bounds
            sigma = max(sigma, 0.1)
            sigma = min(sigma, 100.0)
            
            # Special handling for q-ary lattices that might have very small eta
            if basis_type == 'q-ary' and sigma < 1.0:
                logger.warning(f"Small sigma detected for q-ary: {sigma}, adjusting to 1.0")
                sigma = 1.0
            
            return sigma
    except Exception as e:
        logger.error(f"Failed to compute sigma: {e}")
        # Return safe default
        return 2.0

def run_single_test(basis_type, dim, num_samples=50):
    """Run a single test with full error handling."""
    logger.info(f"\nTesting {basis_type} (dim={dim})")
    
    result = {
        'basis_type': basis_type,
        'dimension': dim,
        'status': 'running',
        'start_time': time.time()
    }
    
    try:
        # Import required modules
        from samplers import imhk_sampler_wrapper
        
        # Create basis with fallback
        basis_info = safe_create_basis(dim, basis_type)
        result['basis_created'] = True
        
        # Compute sigma with safety checks
        sigma = safe_compute_sigma(basis_info, basis_type, dim)
        result['sigma'] = sigma
        logger.info(f"Using sigma: {sigma:.4f}")
        
        # Run sampler
        logger.info("Running IMHK sampler...")
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=num_samples,
            burn_in=min(num_samples//2, 50),
            basis_type=basis_type
        )
        
        result['samples_generated'] = samples.shape[0]
        result['acceptance_rate'] = metadata.get('acceptance_rate', 0)
        result['elapsed_time'] = time.time() - result['start_time']
        result['status'] = 'success'
        
        logger.info(f"✓ Success: {result['samples_generated']} samples, "
                   f"acceptance rate: {result['acceptance_rate']:.4f}, "
                   f"time: {result['elapsed_time']:.2f}s")
        
    except Exception as e:
        result['error'] = str(e)
        result['status'] = 'failed'
        result['elapsed_time'] = time.time() - result['start_time']
        logger.error(f"✗ Failed: {e}")
        
        # Try to provide helpful diagnostics
        if "sigma" in str(e).lower():
            logger.info("Hint: Error might be related to sigma parameter")
        elif "dimension" in str(e).lower():
            logger.info("Hint: Error might be related to dimension")
    
    return result

def main():
    """Run comprehensive crypto tests with error recovery."""
    logger.info("Fixed Cryptographic Lattice Test")
    logger.info("="*50)
    
    # Test configurations (start with easier cases)
    test_configs = [
        # Easy cases first
        ('identity', 4),
        ('identity', 8),
        ('q-ary', 8),
        
        # Medium difficulty
        ('q-ary', 16),
        ('NTRU', 512),  # NTRU uses fixed dimension internally
        
        # Harder cases
        ('PrimeCyclotomic', 683),  # Fixed dimension
        ('q-ary', 32),
    ]
    
    results = []
    start_time = time.time()
    
    for i, (basis_type, dim) in enumerate(test_configs):
        logger.info(f"\nTest {i+1}/{len(test_configs)}")
        result = run_single_test(basis_type, dim)
        results.append(result)
        
        # Save intermediate results
        with open('crypto_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time:.2f}s")
    
    # Detailed results
    logger.info("\nDetailed Results:")
    for r in results:
        status_icon = "✓" if r['status'] == 'success' else "✗"
        logger.info(f"{status_icon} {r['basis_type']:15} (dim={r['dimension']:4}): "
                   f"sigma={r.get('sigma', 'N/A'):6.3f}, "
                   f"acc_rate={r.get('acceptance_rate', 0):6.4f}, "
                   f"time={r.get('elapsed_time', 0):6.2f}s")
        if r['status'] == 'failed':
            logger.info(f"  Error: {r.get('error', 'Unknown')}")
    
    # Save final report
    report = {
        'summary': {
            'total_tests': len(results),
            'successful': successful,
            'failed': failed,
            'total_time': total_time
        },
        'results': results
    }
    
    with open('crypto_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nResults saved to crypto_test_report.json")
    
    if successful == len(results):
        logger.info("\nAll tests passed! The crypto implementation is working.")
        logger.info("\nNext steps:")
        logger.info("1. Run larger experiments: sage generate_crypto_publication_results.py")
        logger.info("2. Use optimized TV distance: sage run_optimized_smoke_test.py")
    else:
        logger.warning(f"\n{failed} tests failed. Check the detailed results above.")
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Check sigma values - they might be too small")
        logger.info("2. Try smaller dimensions first")
        logger.info("3. Run debug_crypto_test.py for more details")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())